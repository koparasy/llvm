//===- DetectHostDeviceClosureLayoutMismatch.cpp - closure ABI check ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Warns when a SYCL kernel's captured-closure layout diverges between the host
// module and the device module of a merged cir.offload.container.
//
// A SYCL kernel lambda is compiled twice: once for the host and once for the
// SPIR-V device. Preprocessor divergence (`#ifdef __SYCL_DEVICE_ONLY__`) or
// divergent `[=]` implicit captures can make the two closures disagree in a
// field's type/layout while keeping the same total size. The
// `sizeof(KernelType) == Info.KernelSize` static_assert in `sycl/queue.hpp`
// only checks total size, so an equal-size / different-type divergence compiles
// clean and produces silent wrong results at runtime (observed: a kernel that
// should write 111 wrote 0). Only the joined host+device container sees both
// layouts, so only a container pass can catch it.
//
// The pass reuses the KernelBindingTable, which already recovers the
// host<->device kernel correspondence (by mangled name) and the launch sites.
// For each SYCL binding it:
//   - recovers the HOST closure record from a launch site's HostKernelRef
//     closure operand (the HostKernelRef wraps a pointer to the user closure);
//   - recovers the DEVICE closure record from each device spir_kernel's
//     closure object, the alloca CIRGen tags `cir.sycl_kernel_object`;
//   - compares the two records' ordered field lists. A field holding a SYCL
//     special type (accessor, stream, ...) is reported as not checked: its
//     host and device representations differ by design.
// Records are matched by BINDING, never by type-name equality: the host and
// device closures carry different aliases (`anon.0` vs `__cir_lowered.anon.0`).
// Any divergence -- a differing field count, or a field whose type differs
// (ignoring a captured pointer's address space, a device-lowering artifact) --
// emits an MLIR warning anchored at the capture's source line and naming the
// field, flagging a host/device closure mismatch (undefined behavior). The CIR
// field types themselves are not printed: at the merge stage there is no Clang
// AST to recover the C++ spelling, and the lowered CIR types would be more
// confusing than helpful; the source line and capture name identify the
// culprit.
//
// Analysis + warning ONLY: the pass never mutates the module. It runs at the
// head of the cir-offload-merge combine pipeline, BEFORE the mem2reg/sccp that
// would promote the device closure alloca away, so the check sees the
// as-emitted closures.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Location.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLKernelParams.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

namespace mlir {
#define GEN_PASS_DEF_OFFLOADDETECTHOSTDEVICECLOSURELAYOUTMISMATCH
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace cir;

namespace {

// Whether two closure-field types are equivalent up to pointer address space.
// A captured pointer is emitted without an address space on the host but in the
// device global/generic address space (target_address_space(1)/(4)) on the
// device -- an expected lowering artifact present on every SYCL kernel, not a
// layout divergence. Recurse through pointers ignoring their address space so
// only a genuine type/size difference (e.g. host `float` vs device `int`)
// counts.
static bool sameIgnoringAddrSpace(mlir::Type a, mlir::Type b) {
  if (a == b)
    return true;
  auto pa = mlir::dyn_cast<cir::PointerType>(a);
  auto pb = mlir::dyn_cast<cir::PointerType>(b);
  if (pa && pb)
    return sameIgnoringAddrSpace(pa.getPointee(), pb.getPointee());
  return false;
}

// Peel the offset-0 reinterprets CIRGen inserts on a closure `this`
// (address-space / bitcast casts and base_class_addr views) so a closure
// operand resolves to the object pointer it names.
static mlir::Value peelClosurePtr(mlir::Value v) {
  for (;;) {
    if (auto bca = v.getDefiningOp<cir::BaseClassAddrOp>()) {
      v = bca.getSrcAddr();
      continue;
    }
    if (auto cast = v.getDefiningOp<cir::CastOp>()) {
      if (cast.getKind() == cir::CastKind::address_space ||
          cast.getKind() == cir::CastKind::bitcast) {
        v = cast.getSrc();
        continue;
      }
    }
    return v;
  }
}

// The HOST closure record for a launch, recovered from the HostKernelRef
// closure operand the KernelBindingTable stored (the HostKernelRefBase* passed
// to the runtime launch). CIRGen builds a HostKernelRef wrapper record that
// holds a POINTER to the user closure object; after peeling the base-class /
// address-space views the operand is a pointer to that wrapper, whose sole
// pointer-to-record member points at the closure. If the operand already points
// straight at a record with no pointer-to-record member (the handler-path
// lambda pointer), that record is itself the closure.
static cir::RecordType hostClosureRecord(mlir::Value closureOperand) {
  if (!closureOperand)
    return {};
  mlir::Value obj = peelClosurePtr(closureOperand);
  auto ptrTy = mlir::dyn_cast<cir::PointerType>(obj.getType());
  if (!ptrTy)
    return {};
  auto rec = mlir::dyn_cast<cir::RecordType>(ptrTy.getPointee());
  if (!rec)
    return {};
  // Direct path: the wrapper holds a pointer to the user closure record.
  for (mlir::Type member : rec.getMembers())
    if (auto memPtr = mlir::dyn_cast<cir::PointerType>(member))
      if (auto inner = mlir::dyn_cast<cir::RecordType>(memPtr.getPointee()))
        return inner;
  // Handler path: the operand is already the closure object pointer.
  return rec;
}

// Collect "file:line:col" strings for every FileLineColLoc reachable from
// `loc` (peeling FusedLoc/NameLoc), into `out`. Used to match a device kernel's
// lambda source location against the host HostKernel<lambda at ...> wrapper.
static void collectFileLineCols(mlir::Location loc,
                                llvm::SmallVectorImpl<std::string> &out) {
  if (auto flc = mlir::dyn_cast<mlir::FileLineColLoc>(loc)) {
    out.push_back((flc.getFilename().strref() + ":" +
                   llvm::Twine(flc.getLine()) + ":" +
                   llvm::Twine(flc.getColumn()))
                      .str());
    return;
  }
  if (auto fused = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
    for (mlir::Location sub : fused.getLocations())
      collectFileLineCols(sub, out);
    return;
  }
  if (auto named = mlir::dyn_cast<mlir::NameLoc>(loc))
    collectFileLineCols(named.getChildLoc(), out);
}

// Whether `fn` is a `detail::HostKernel<...>` constructor: the wrapper CIRGen
// builds around the user closure, whose second argument is a pointer to the
// closure record (KernelBindingTable recognizes it by its hostkernel.ctor
// role).
static bool isHostKernelCtor(cir::FuncOp fn) {
  llvm::StringRef name = fn.getSymName();
  return name.contains("6detail10HostKernelI") &&
         (name.contains("C1") || name.contains("C2"));
}

// Fallback host-closure recovery for when the launch site's closure operand was
// not recovered (the handler launch path, where the binding table reports
// "closure none"). CIRGen wraps the user closure in a
// `detail::HostKernel<(lambda at FILE:LINE:COL), ...>` whose type name embeds
// the lambda's source location and whose constructor takes a pointer to the
// closure record as its second argument. Match that wrapper to `deviceKernel`
// by the lambda source location (the device kernel and its arguments carry the
// same loc), then read the closure record off the constructor's second param.
// The location also appears in wrappers around the lambda, e.g.
// HostKernel<RoundedRangeKernel<(lambda at ...)>>, so prefer the wrapper whose
// closure record is named like the device one, else the innermost wrapper.
static cir::RecordType
hostClosureRecordFromHostKernel(mlir::ModuleOp hostModule,
                                cir::FuncOp deviceKernel,
                                cir::RecordType deviceRec) {
  // Source locations of the device kernel's lambda (its own loc and each
  // argument's), any of which the host wrapper name should embed.
  llvm::SmallVector<std::string, 4> needles;
  collectFileLineCols(deviceKernel.getLoc(), needles);
  if (!deviceKernel.getBody().empty())
    for (mlir::BlockArgument arg :
         deviceKernel.getBody().front().getArguments())
      collectFileLineCols(arg.getLoc(), needles);
  if (needles.empty())
    return {};

  llvm::StringRef deviceName;
  if (deviceRec && deviceRec.getName())
    deviceName = deviceRec.getName().getValue();
  deviceName.consume_front("__cir_lowered.");

  cir::RecordType found;
  size_t foundWrapperLen = 0;
  bool exact = false;
  hostModule.walk([&](cir::FuncOp fn) {
    if (exact || !isHostKernelCtor(fn))
      return;
    llvm::ArrayRef<mlir::Type> inputs = fn.getFunctionType().getInputs();
    if (inputs.size() < 2)
      return;
    auto wrapperPtr = mlir::dyn_cast<cir::PointerType>(inputs[0]);
    if (!wrapperPtr)
      return;
    auto wrapperRec = mlir::dyn_cast<cir::RecordType>(wrapperPtr.getPointee());
    if (!wrapperRec || !wrapperRec.getName())
      return;
    llvm::StringRef wrapperName = wrapperRec.getName().getValue();
    if (llvm::none_of(needles, [&](const std::string &n) {
          return wrapperName.contains(n);
        }))
      return;
    // The constructor's second argument is a pointer/reference to the closure.
    auto closurePtr = mlir::dyn_cast<cir::PointerType>(inputs[1]);
    if (!closurePtr)
      return;
    auto rec = mlir::dyn_cast<cir::RecordType>(closurePtr.getPointee());
    if (!rec)
      return;
    if (rec.getName() && !deviceName.empty() &&
        rec.getName().getValue() == deviceName) {
      found = rec;
      exact = true;
      return;
    }
    if (!found || wrapperName.size() < foundWrapperLen) {
      found = rec;
      foundWrapperLen = wrapperName.size();
    }
  });
  return found;
}

// The DEVICE closure record of a spir_kernel: the type of the closure object
// CIRGen tags with `cir.sycl_kernel_object`.
static cir::RecordType deviceClosureRecord(cir::FuncOp kernel) {
  mlir::Value obj = cir::getSYCLKernelObject(kernel);
  if (!obj)
    return {};
  auto alloca = obj.getDefiningOp<cir::AllocaOp>();
  return alloca ? mlir::dyn_cast<cir::RecordType>(alloca.getAllocaType())
                : cir::RecordType();
}

// Closure fields holding (or containing) a SYCL special type such as an
// accessor or stream, read off the kernel parameters CIRGen decomposed them
// into. Their host and device representations differ by design (a host
// accessor holds its runtime implementation, the device one its pointer and
// ranges), so a type comparison would flag every accessor kernel.
static llvm::StringSet<> specialTypeFields(cir::FuncOp kernel) {
  llvm::StringSet<> fields;
  for (unsigned i = 0, e = kernel.getNumArguments(); i != e; ++i) {
    std::optional<cir::SYCLKernelParam> param =
        cir::getSYCLKernelParam(kernel, i);
    if (param && param->kind != "pointer" && param->kind != "scalar" &&
        param->kind != "std_layout")
      fields.insert(param->field);
  }
  return fields;
}

struct OffloadDetectHostDeviceClosureLayoutMismatchPass
    : public impl::OffloadDetectHostDeviceClosureLayoutMismatchBase<
          OffloadDetectHostDeviceClosureLayoutMismatchPass> {
  void runOnOperation() override;
};

void OffloadDetectHostDeviceClosureLayoutMismatchPass::runOnOperation() {
  mlir::ModuleOp container = getOperation();
  if (!cir::isOffloadContainer(container))
    return;

  // Analysis only: nothing below mutates the IR.
  markAllAnalysesPreserved();

  cir::KernelBindingTable &table = getAnalysis<cir::KernelBindingTable>();

  for (const auto &entry : table) {
    const cir::KernelBinding &binding = entry.second;
    if (binding.kind != cir::BindingKind::SYCLName)
      continue;

    // Recover the host closure record from any launch site of this binding
    // (the direct launch path exposes the HostKernelRef closure operand).
    cir::RecordType hostRecFromLaunch;
    for (const cir::LaunchSite &site : binding.launchSites) {
      hostRecFromLaunch = hostClosureRecord(site.getSYCLClosure());
      if (hostRecFromLaunch)
        break;
    }

    for (cir::FuncOp kernel : binding.deviceKernels) {
      if (kernel.getCallingConv() != cir::CallingConv::SpirKernel)
        continue;
      if (kernel.getBody().empty())
        continue;

      // When the launch site did not expose the closure (the handler launch
      // path reports "closure none"), fall back to the host-side
      // HostKernel<lambda at ...> wrapper, matched to this kernel by lambda
      // source location.
      cir::RecordType devRec = deviceClosureRecord(kernel);
      if (!devRec)
        continue;

      cir::RecordType hostRec = hostRecFromLaunch;
      if (!hostRec)
        hostRec = hostClosureRecordFromHostKernel(
            cir::getOffloadContainerHostModule(container), kernel, devRec);
      if (!hostRec)
        continue;
      llvm::ArrayRef<mlir::Type> hostFields = hostRec.getMembers();
      llvm::ArrayRef<mlir::Type> devFields = devRec.getMembers();

      // Per-field get_member ops of the device closure: their names label the
      // diagnostic and their locations anchor it at the source line of the
      // divergent capture (a real file:line:col, since CIRGen emits the .cir
      // with debug locations) rather than at the whole kernel.
      llvm::DenseMap<uint64_t, cir::GetMemberOp> fieldOps;
      kernel.walk([&](cir::GetMemberOp gm) {
        if (gm.getAddr().getType().getPointee() == devRec)
          fieldOps.try_emplace(gm.getIndex(), gm);
      });
      auto nameOf = [&](uint64_t i) -> std::string {
        auto it = fieldOps.find(i);
        if (it != fieldOps.end())
          return "'" + it->second.getName().str() + "'";
        return "#" + std::to_string(i);
      };
      // The location to anchor a field-`i` diagnostic at: the field's
      // get_member op when known, else the kernel itself.
      auto locOf = [&](uint64_t i) -> mlir::Location {
        auto it = fieldOps.find(i);
        if (it != fieldOps.end())
          return it->second.getLoc();
        return kernel.getLoc();
      };

      if (hostFields.size() != devFields.size()) {
        mlir::emitWarning(kernel.getLoc())
            << "SYCL kernel captured-closure layout differs between host and "
               "device (field count differs); likely a #ifdef "
               "__SYCL_DEVICE_ONLY__ or divergent [=] capture -- undefined "
               "behavior";
      }

      // TODO: compare a host special-type field against the device parameters
      // it was decomposed into instead of skipping it.
      llvm::StringSet<> special = specialTypeFields(kernel);
      size_t n = std::min(hostFields.size(), devFields.size());
      for (size_t i = 0; i < n; ++i) {
        if (sameIgnoringAddrSpace(hostFields[i], devFields[i]))
          continue;
        auto it = fieldOps.find(i);
        if (it != fieldOps.end() && special.contains(it->second.getName())) {
          mlir::emitRemark(locOf(i))
              << "SYCL kernel captured field " << nameOf(i)
              << " not checked for a host/device layout difference: it holds "
                 "a SYCL special type whose host and device representations "
                 "differ";
          continue;
        }
        mlir::emitWarning(locOf(i))
            << "SYCL kernel captured-closure layout differs between host and "
               "device (captured field "
            << nameOf(i)
            << "); likely a #ifdef __SYCL_DEVICE_ONLY__ or divergent [=] "
               "capture -- undefined behavior";
      }
    }
  }
}

} // namespace

std::unique_ptr<Pass>
mlir::createOffloadDetectHostDeviceClosureLayoutMismatchPass() {
  return std::make_unique<OffloadDetectHostDeviceClosureLayoutMismatchPass>();
}
