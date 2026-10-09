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
// A spir_kernel scalar parameter is bound to the closure capture field CIRGen
// recorded for it in `cir.sycl_kernel_param` (see SYCLKernelParams.h).
// KernelBindingTable already
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

#include "mlir/IR/Diagnostics.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLKernelParams.h"

namespace mlir {
#define GEN_PASS_DECL_OFFLOADPROPAGATESYCLKERNELARGS
#define GEN_PASS_DEF_OFFLOADPROPAGATESYCLKERNELARGS
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace cir;

namespace {

struct OffloadPropagateSYCLKernelArgsPass
    : public impl::OffloadPropagateSYCLKernelArgsBase<
          OffloadPropagateSYCLKernelArgsPass> {
  void runOnOperation() override;
};

void OffloadPropagateSYCLKernelArgsPass::runOnOperation() {
  mlir::ModuleOp container = getOperation();
  if (!cir::isOffloadContainer(container))
    return;
  std::optional<cir::HostFactsMode> mode = cir::parseHostFactsMode(hostFacts);
  if (!mode) {
    container.emitError() << "unknown host-facts mode '" << hostFacts
                          << "' (expected legacy, ipo or both)";
    return signalPassFailure();
  }
  // The legacy table is the cached analysis the other container passes share;
  // the other modes build their own.
  cir::HostFactsOptions options;
  options.mode = *mode;
  options.assumeClosedTU = hostFactsClosedTU;
  options.bottomUp = hostFactsBottomUp;
  options.subobjects = hostFactsSubobjects;
  std::optional<cir::KernelBindingTable> ownTable;
  if (*mode != cir::HostFactsMode::Legacy)
    ownTable.emplace(container, options);
  cir::KernelBindingTable &table =
      ownTable ? *ownTable : getAnalysis<cir::KernelBindingTable>();
  if (*mode == cir::HostFactsMode::Both) {
    // Legacy is the result; report where the two sources differ.
    unsigned disagreements = 0;
    for (const cir::HostFactsConstComparison &cmp :
         table.getHostFactsConstComparisons()) {
      mlir::Location loc = container.getLoc();
      if (const cir::KernelBinding *binding = table.lookup(cmp.kernel))
        if (!binding->deviceKernels.empty())
          loc = binding->deviceKernels.front()->getLoc();
      if (cmp.disagrees()) {
        ++disagreements;
        mlir::emitRemark(loc) << cir::formatHostFactsDisagreement(cmp);
      } else if (hostFactsCensus) {
        mlir::emitRemark(loc)
            << "host-facts: " << cir::formatHostFactsComparison(cmp);
      }
    }
    mlir::emitRemark(container.getLoc())
        << "host-facts-disagreements: " << disagreements << " of "
        << table.getHostFactsConstComparisons().size()
        << " scalar capture fields " << cir::formatHostFactsOptions(options);
  }
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
        llvm::StringRef field = cir::getSYCLCapturedField(
            kernel, arg.getArgNumber(), /*wantPointer=*/false);
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
