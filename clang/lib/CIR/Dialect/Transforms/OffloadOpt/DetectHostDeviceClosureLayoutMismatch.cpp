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
//     prologue (the record the kernel reconstructs via cir.get_member stores);
//   - compares the two records' ordered field lists.
// Records are matched by BINDING, never by type-name equality: the host and
// device closures carry different aliases (`anon.0` vs `__cir_lowered.anon.0`).
// Any divergence -- a differing field count, or a field whose type differs --
// emits an MLIR warning naming the kernel, the field index/name, and both
// types, flagging it as a host/device closure ABI mismatch (undefined
// behavior).
//
// Analysis + warning ONLY: the pass never mutates the module and is not part of
// the default cir-offload-merge pipeline, so it must be requested explicitly.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
#define GEN_PASS_DEF_OFFLOADDETECTHOSTDEVICECLOSURELAYOUTMISMATCH
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace cir;

namespace {

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

// The DEVICE closure record a spir_kernel reconstructs in its prologue: the
// sole named record `cir.alloca` whose address feeds a `cir.get_member` (the
// per-field store the decomposed kernel arguments populate). A scalar spill
// slot (`C.addr`) is not a record, so it is skipped.
static cir::RecordType deviceClosureRecord(cir::FuncOp kernel) {
  cir::RecordType found;
  kernel.walk([&](cir::AllocaOp alloca) {
    if (found)
      return;
    auto rec = mlir::dyn_cast<cir::RecordType>(alloca.getAllocaType());
    if (!rec || !rec.getName())
      return;
    for (mlir::Operation *user : alloca.getResult().getUsers()) {
      if (mlir::isa<cir::GetMemberOp>(user)) {
        found = rec;
        return;
      }
      // A closure `this` is address-space cast before member access.
      if (auto cast = mlir::dyn_cast<cir::CastOp>(user))
        for (mlir::Operation *cu : cast.getResult().getUsers())
          if (mlir::isa<cir::GetMemberOp>(cu)) {
            found = rec;
            return;
          }
    }
  });
  return found;
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

    // Recover the host closure record from any launch site of this binding.
    cir::RecordType hostRec;
    for (const cir::LaunchSite &site : binding.launchSites) {
      hostRec = hostClosureRecord(site.getSYCLClosure());
      if (hostRec)
        break;
    }
    if (!hostRec)
      continue;
    llvm::ArrayRef<mlir::Type> hostFields = hostRec.getMembers();

    for (cir::FuncOp kernel : binding.deviceKernels) {
      if (kernel.getCallingConv() != cir::CallingConv::SpirKernel)
        continue;
      if (kernel.getBody().empty())
        continue;

      cir::RecordType devRec = deviceClosureRecord(kernel);
      if (!devRec)
        continue;
      llvm::ArrayRef<mlir::Type> devFields = devRec.getMembers();

      // Source-level names for the device closure fields, for the diagnostic.
      llvm::DenseMap<uint64_t, llvm::StringRef> fieldNames;
      kernel.walk([&](cir::GetMemberOp gm) {
        if (gm.getAddr().getType().getPointee() == devRec)
          fieldNames.try_emplace(gm.getIndex(), gm.getName());
      });
      auto nameOf = [&](uint64_t i) -> std::string {
        auto it = fieldNames.find(i);
        if (it != fieldNames.end())
          return "'" + it->second.str() + "'";
        return "#" + std::to_string(i);
      };

      if (hostFields.size() != devFields.size()) {
        kernel.emitWarning()
            << "device kernel '" << kernel.getName()
            << "': host/device closure ABI mismatch: field count differs (host "
            << hostFields.size() << " vs device " << devFields.size()
            << "); likely __SYCL_DEVICE_ONLY__ or divergent captures -- "
               "undefined behavior";
      }

      size_t n = std::min(hostFields.size(), devFields.size());
      for (size_t i = 0; i < n; ++i) {
        if (hostFields[i] == devFields[i])
          continue;
        kernel.emitWarning()
            << "device kernel '" << kernel.getName() << "': closure field "
            << nameOf(i) << " layout differs: host " << hostFields[i]
            << " vs device " << devFields[i]
            << "; host/device closure ABI mismatch (likely "
               "__SYCL_DEVICE_ONLY__ or divergent captures) -- undefined "
               "behavior";
      }
    }
  }
}

} // namespace

std::unique_ptr<Pass>
mlir::createOffloadDetectHostDeviceClosureLayoutMismatchPass() {
  return std::make_unique<OffloadDetectHostDeviceClosureLayoutMismatchPass>();
}
