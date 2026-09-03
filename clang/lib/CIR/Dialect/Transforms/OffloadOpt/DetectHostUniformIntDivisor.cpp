//===- DetectHostUniformIntDivisor.cpp - launch-uniform int-div remark ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Remarks on a SYCL device integer division / remainder whose DIVISOR is
// kernel-scope-uniform: a value that is invariant across the whole launch
// because it is derived only from captured SCALAR closure fields and
// compile-time constants. The motivating case (CMPLRLLVM-76652) is a pytorch
// kernel `c = lid / C_`, where `lid = item.get_local_id(0)` varies per
// work-item but `C_` is a captured scalar kernel argument -- the same value at
// every work-item of the launch. A division by such a uniform value can be
// strength-reduced to a magic-multiply whose magic/shift constants are a
// function of C_ only, computable once on the host.
//
// This is analysis + remark ONLY. It never mutates the module, adds no kernel
// argument, and does not rewrite the division (the auto-transform is Flavor B,
// deliberately out of scope). The point is to demonstrate the combine's unique
// host+device visibility: the front end cannot see that C_ is uniform across
// the launch (it sees one work-item's control flow), and the back end cannot
// see that C_ is a runtime value known on the host (it sees an opaque kernel
// argument). Only the joined host+device container sees both.
//
// The uniformity oracle mirrors the scalar-capture classification of
// PropagateSYCLKernelArgs.cpp: a spir_kernel scalar parameter that the prologue
// stores into a closure capture field (arg -> [spill slot ->] cir.get_member
// store) is a captured scalar, and hence uniform. From there a divisor is
// uniform iff every step back is either such a captured field, a cir.const, or
// a pure (memory-effect-free, non-call) arithmetic op over uniform operands. A
// chain that reaches a work-item/index builtin (a cir.call), a load from
// non-captured memory, or a parameter with no captured-scalar field is NOT
// uniform. A divisor that is itself a cir.const is skipped -- the backend
// already strength-reduces compile-time constants; the interesting case is a
// runtime-but-launch-uniform divisor like C_.
//
// A SYCL device kernel calls its functor's `operator()` (an always_inline
// helper) rather than containing the division directly, so this pass expects
// the CIR inliner to have folded operator() into the kernel first; run it after
// `-inline` (the lit fixtures model both the pre- and post-inline shapes). It
// is not part of the default cir-offload-merge pipeline; request it explicitly.
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
#define GEN_PASS_DECL_OFFLOADDETECTHOSTUNIFORMINTDIVISOR
#define GEN_PASS_DEF_OFFLOADDETECTHOSTUNIFORMINTDIVISOR
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace cir;

namespace {

// Peel value-preserving pointer reinterprets (address-space / bitcast casts) so
// two references to the same closure record compare equal regardless of the
// address-space cast the kernel inserts before handing the closure to
// operator().
static mlir::Value peelPtrCasts(mlir::Value v) {
  while (auto cast = v.getDefiningOp<cir::CastOp>()) {
    if (cast.getKind() != cir::CastKind::address_space &&
        cast.getKind() != cir::CastKind::bitcast)
      break;
    v = cast.getSrc();
  }
  return v;
}

// The single closure capture field a spir_kernel SCALAR parameter is stored
// into, together with the closure record it is a member of, or an empty field
// if the parameter reaches no field or more than one. Shares the forward-walk
// shape with PropagateSYCLKernelArgs.cpp's mapper (arg -> [spill slot ->]
// cir.get_member store), and additionally reports the get_member base so a
// later divisor load can be matched to the very closure the parameter feeds.
struct CapturedField {
  llvm::StringRef name;
  mlir::Value closure; // the record the field lives in (casts peeled)
};

static CapturedField closureFieldForArg(mlir::Value arg) {
  CapturedField result;
  llvm::SmallVector<mlir::Value, 4> worklist{arg};
  llvm::SmallPtrSet<mlir::Value, 8> seen{arg};
  while (!worklist.empty()) {
    mlir::Value v = worklist.pop_back_val();
    for (mlir::Operation *user : v.getUsers()) {
      if (auto st = mlir::dyn_cast<cir::StoreOp>(user)) {
        if (st.getValue() != v)
          continue;
        if (auto gm = st.getAddr().getDefiningOp<cir::GetMemberOp>()) {
          if (!result.name.empty() && result.name != gm.getName())
            return {}; // Two distinct fields: ambiguous, give up.
          result.name = gm.getName();
          result.closure = peelPtrCasts(gm.getAddr());
          continue;
        }
        // A spill into a stack slot: follow the reloads of that slot.
        for (mlir::Operation *slotUser : st.getAddr().getUsers())
          if (auto load = mlir::dyn_cast<cir::LoadOp>(slotUser))
            if (seen.insert(load.getResult()).second)
              worklist.push_back(load.getResult());
        continue;
      }
      if (auto cast = mlir::dyn_cast<cir::CastOp>(user)) {
        if ((cast.getKind() == cir::CastKind::integral ||
             cast.getKind() == cir::CastKind::bitcast) &&
            seen.insert(cast.getResult()).second)
          worklist.push_back(cast.getResult());
      }
    }
  }
  return result;
}

// Uniformity oracle over the def-use chain of a divisor. `fields` maps each
// captured-scalar field name to the closure record it belongs to; `outField`
// receives the first captured field the chain reaches, to name in the remark.
static bool isUniform(mlir::Value v, const llvm::StringMap<mlir::Value> &fields,
                      llvm::SmallPtrSetImpl<mlir::Value> &visited,
                      llvm::StringRef &outField, unsigned depth) {
  // A value's uniformity is a fixed property; a value re-reached on a second
  // operand path was already accepted (a rejected one short-circuits the whole
  // divisor), so treat a revisit as uniform. The depth cap guards pathological
  // chains.
  if (depth > 64)
    return false;
  if (!visited.insert(v).second)
    return true;

  // A kernel parameter is uniform iff it is a captured scalar (it maps to a
  // closure field). Any other block argument -- a non-captured parameter, or a
  // loop/region-carried value -- is conservatively non-uniform.
  if (mlir::isa<mlir::BlockArgument>(v)) {
    CapturedField cf = closureFieldForArg(v);
    if (cf.name.empty() || !fields.count(cf.name))
      return false;
    if (outField.empty())
      outField = cf.name;
    return true;
  }

  mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return false;

  // A load is uniform only when it reads a captured-scalar field of the closure
  // that parameter stores populated (the divisor `C_` in operator() after
  // inlining). Any other load -- a USM buffer, an accessor -- is memory and so
  // non-uniform.
  if (auto load = mlir::dyn_cast<cir::LoadOp>(def)) {
    auto gm = load.getAddr().getDefiningOp<cir::GetMemberOp>();
    if (!gm)
      return false;
    auto it = fields.find(gm.getName());
    if (it == fields.end() || peelPtrCasts(gm.getAddr()) != it->second)
      return false;
    if (outField.empty())
      outField = gm.getName();
    return true;
  }

  // A call (a work-item/index builtin such as get_local_id / __spirv_BuiltIn*,
  // or any other callee) is not a uniform source.
  if (mlir::isa<cir::CallOp>(def))
    return false;

  // A pure (memory-effect-free) op is a function of its operands only, so it is
  // uniform iff they all are. This covers cir.const (no operands -> uniform),
  // integral/bool casts, and integer arithmetic (add/sub/mul/shift/and/or/xor/
  // select/...). An impure or unknown op is conservatively non-uniform.
  if (!mlir::isMemoryEffectFree(def))
    return false;
  for (mlir::Value operand : def->getOperands())
    if (!isUniform(operand, fields, visited, outField, depth + 1))
      return false;
  return true;
}

struct OffloadDetectHostUniformIntDivisorPass
    : public impl::OffloadDetectHostUniformIntDivisorBase<
          OffloadDetectHostUniformIntDivisorPass> {
  void runOnOperation() override;
};

void OffloadDetectHostUniformIntDivisorPass::runOnOperation() {
  mlir::ModuleOp container = getOperation();
  if (!cir::isOffloadContainer(container))
    return;

  // Analysis only: nothing below mutates the IR.
  markAllAnalysesPreserved();

  auto scanKernel = [&](cir::FuncOp kernel) {
    if (kernel.getCallingConv() != cir::CallingConv::SpirKernel)
      return;
    if (kernel.getBody().empty())
      return;

    // Captured-scalar fields of this kernel's closure: each scalar parameter
    // the prologue stores into a closure member. The uniformity oracle treats
    // exactly these fields (and constants, through pure arithmetic) as uniform.
    llvm::StringMap<mlir::Value> fields;
    for (mlir::BlockArgument arg : kernel.getBody().front().getArguments()) {
      if (!mlir::isa<cir::IntType, cir::FPTypeInterface, cir::BoolType>(
              arg.getType()))
        continue;
      CapturedField cf = closureFieldForArg(arg);
      if (!cf.name.empty() && cf.closure)
        fields.try_emplace(cf.name, cf.closure);
    }
    if (fields.empty())
      return;

    kernel.getBody().walk([&](mlir::Operation *op) {
      if (!mlir::isa<cir::DivOp, cir::RemOp>(op))
        return;
      mlir::Value divisor = op->getOperand(1);
      // A compile-time-constant divisor is already handled by the backend's
      // constant strength reduction; the interesting case is a runtime value
      // that is nonetheless launch-uniform.
      if (divisor.getDefiningOp<cir::ConstantOp>())
        return;

      llvm::SmallPtrSet<mlir::Value, 16> visited;
      llvm::StringRef field;
      if (!isUniform(divisor, fields, visited, field, 0) || field.empty())
        return;

      op->emitRemark()
          << "device kernel '" << kernel.getName() << "': integer "
          << (mlir::isa<cir::DivOp>(op) ? "division" : "remainder")
          << " by launch-uniform capture '" << field
          << "'; consider host-precomputing a reciprocal magic-multiply "
             "(CMPLRLLVM-76652)";
    });
  };

  // Survey mode: walk every device-side spir_kernel directly, without a host
  // binding. Used to scan a device-only module for the pattern's frequency.
  if (scanAllKernels) {
    for (mlir::ModuleOp mod : cir::getOffloadContainerDeviceModules(container))
      for (cir::FuncOp kernel : mod.getOps<cir::FuncOp>())
        scanKernel(kernel);
    return;
  }

  cir::KernelBindingTable &table = getAnalysis<cir::KernelBindingTable>();
  for (const auto &entry : table) {
    const cir::KernelBinding &binding = entry.second;
    // Every SYCL device kernel is a candidate, not only TU-local/optimizable
    // ones. The transform-soundness gate the sibling opts use (in-place edits
    // are only safe when every launch is visible) does not apply to a remark:
    // the host may precompute the reciprocal of a launch-uniform divisor
    // regardless of whether other TUs also launch the kernel -- and the
    // motivating case (CMPLRLLVM-76652) is a named functor, which is not
    // TU-local. The divisor's uniformity is a property of the kernel body and
    // its captured scalars alone.
    if (binding.kind != cir::BindingKind::SYCLName)
      continue;
    for (cir::FuncOp kernel : binding.deviceKernels)
      scanKernel(kernel);
  }
}

} // namespace

std::unique_ptr<Pass> mlir::createOffloadDetectHostUniformIntDivisorPass() {
  return std::make_unique<OffloadDetectHostUniformIntDivisorPass>();
}
