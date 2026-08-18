//===- PropagateSYCLKernelArgs.cpp - SYCL scalar-capture const prop -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Propagates a SYCL captured SCALAR that is a compile-time constant at every
// visible launch into the device kernel body.
//
// A spir_kernel scalar parameter is bound to a closure capture field (the
// prologue spills the by-value argument to a stack slot, reloads it, and stores
// it into a cir.get_member of the closure record). KernelBindingTable already
// resolved, on the host side, which scalar capture fields hold the SAME
// compile-time constant at every launch site -- see KernelBinding::
// syclConstArgs. This pass maps each device scalar parameter to its field and,
// when that field has an agreed constant, replaces the parameter's uses with a
// materialized cir.const of that value.
//
// The kernel signature/ABI is untouched: the host still passes the argument,
// the argument simply becomes unused on the device side. Soundness: every
// visible launch passes that constant and the kernel is TU-local, so the
// substitution is behavior-preserving. This is the scalar analogue of the
// noalias inference pass (InferKernelParamNoAlias.cpp) and, like it, annotates
// the shared device kernel in place -- never clones it.
//
// The headline win (a captured flag that gates a branch folding away) is
// realized later: the captured value is stored into the closure record, read
// back inside the always_inline operator(), and only collapses once the device
// backend inlines operator() and promotes the closure (SROA). In-combine this
// pass makes the closure-field store a constant; a follow-up device-module
// mem2reg/SCCP then folds any code that reaches the constant through a
// promotable scalar slot.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_OFFLOADPROPAGATESYCLKERNELARGS
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace cir;

namespace {

// The single closure capture field a spir_kernel parameter is stored into, or
// empty if it reaches no field or more than one. The prologue routes the
// by-value scalar argument through a spill slot (before mem2reg) and, for a
// bool, a bitcast/integral reinterpret, into a cir.get_member store; walk that
// forward. Shares the shape with InferKernelParamNoAlias.cpp's mapper, minus
// the address-space cast that only pointer captures carry.
static llvm::StringRef closureFieldForArg(mlir::Value arg) {
  llvm::StringRef field;
  llvm::SmallVector<mlir::Value, 4> worklist{arg};
  llvm::SmallPtrSet<mlir::Value, 8> seen{arg};
  while (!worklist.empty()) {
    mlir::Value v = worklist.pop_back_val();
    for (mlir::Operation *user : v.getUsers()) {
      if (auto st = mlir::dyn_cast<cir::StoreOp>(user)) {
        if (st.getValue() != v)
          continue;
        if (auto gm = st.getAddr().getDefiningOp<cir::GetMemberOp>()) {
          if (!field.empty() && field != gm.getName())
            return {}; // Two distinct fields: ambiguous, give up.
          field = gm.getName();
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
  return field;
}

struct OffloadPropagateSYCLKernelArgsPass
    : public impl::OffloadPropagateSYCLKernelArgsBase<
          OffloadPropagateSYCLKernelArgsPass> {
  void runOnOperation() override;
};

void OffloadPropagateSYCLKernelArgsPass::runOnOperation() {
  mlir::ModuleOp container = getOperation();
  if (!cir::isOffloadContainer(container))
    return;
  cir::KernelBindingTable &table = getAnalysis<cir::KernelBindingTable>();
  bool changed = false;

  // Replace each scalar kernel parameter bound to an agreed-constant capture
  // field with a materialized cir.const. Returns whether anything was replaced.
  auto applyConstArgs = [&](llvm::ArrayRef<cir::FuncOp> kernels,
                            const llvm::StringMap<mlir::TypedAttr> &consts) {
    bool did = false;
    for (cir::FuncOp kernel : kernels) {
      if (kernel.getCallingConv() != cir::CallingConv::SpirKernel)
        continue;
      if (kernel.getBody().empty())
        continue;
      for (mlir::BlockArgument arg : kernel.getBody().front().getArguments()) {
        // Only scalar parameters carry a by-value capture; a pointer/accessor
        // parameter is handled by the noalias pass, not here.
        if (!mlir::isa<cir::IntType, cir::FPTypeInterface, cir::BoolType>(
                arg.getType()))
          continue;
        if (arg.use_empty())
          continue;
        llvm::StringRef field = closureFieldForArg(arg);
        if (field.empty())
          continue;
        mlir::TypedAttr value = consts.lookup(field);
        // A field with no agreed constant, or a constant whose type does not
        // match the parameter (an unexpected reinterpret), is left alone.
        if (!value || value.getType() != arg.getType())
          continue;

        mlir::OpBuilder builder(&kernel.getBody().front(),
                                kernel.getBody().front().begin());
        auto constant =
            cir::ConstantOp::create(builder, kernel.getLoc(), value);
        arg.replaceAllUsesWith(constant);
        did = true;
      }
    }
    return did;
  };

  unsigned specTag = 0;
  for (const auto &entry : table) {
    const cir::KernelBinding &binding = entry.second;

    // An empty const-arg set means host-side recovery found no agreed constant
    // across the launches this TU sees.
    if (binding.kind != cir::BindingKind::SYCLName ||
        binding.launchSites.empty() || binding.syclConstArgs.empty())
      continue;

    if (binding.isTULocalOptimizable) {
      // Every launch is visible and the kernel is TU-local: safe to fold the
      // constant into the shared device kernel in place.
      changed |= applyConstArgs(binding.deviceKernels, binding.syclConstArgs);
    } else {
      // Residual: the shared kernel is cross-TU / by-name reachable, so another
      // TU could launch it with a different argument. Specialize by clone --
      // redirect this TU's launches to a fresh kernel and fold the constant
      // only into the clone, leaving the generic original for unseen launches.
      auto spec = cir::specializeSYCLKernelForLaunches(
          container, binding, binding.launchSites,
          "constprop." + std::to_string(specTag++));
      if (spec) {
        applyConstArgs(spec->deviceKernels, binding.syclConstArgs);
        changed = true; // The clone + redirect mutated the IR regardless.
      }
    }
  }

  if (!changed)
    markAllAnalysesPreserved();
}

} // namespace

std::unique_ptr<Pass> mlir::createOffloadPropagateSYCLKernelArgsPass() {
  return std::make_unique<OffloadPropagateSYCLKernelArgsPass>();
}
