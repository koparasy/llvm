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
// KernelBindingTable already resolved, on the host side, which scalar capture
// fields hold the SAME compile-time constant at every launch site -- see
// KernelBinding::syclConstArgs, from SYCLHostFacts (the stored-value queries
// of the points-to analysis). This pass maps each device scalar parameter to
// its field and, when that field has an agreed constant, replaces the
// parameter's uses with a materialized cir.const of that value.
//
// The kernel signature/ABI is untouched: the host still passes the argument,
// the argument simply becomes unused on the device side. Soundness: every
// launch of this TU passes that constant and no other TU launches the kernel
// (KernelBinding::stampsInPlace), so the substitution is behavior-preserving.
// This is the scalar analogue of the noalias inference pass
// (InferKernelParamNoAlias.cpp) and follows its rules: in place for a kernel
// no other TU launches (or a kernel copy of an internalized launch path);
// nothing for a kernel other TUs can launch (a remark per field). One remark
// per decision (category cir-offload-kernel-arg-constant).
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
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLHostFacts.h"
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
    return markAllAnalysesPreserved();
  std::optional<cir::HostFactsMode> mode = cir::parseHostFactsMode(hostFacts);
  if (!mode) {
    container.emitError() << "unknown host-facts mode '" << hostFacts
                          << "' (expected ipo or none)";
    return signalPassFailure();
  }
  if (*mode == cir::HostFactsMode::None)
    return markAllAnalysesPreserved();
  // The table and the host facts the container passes share
  // (SYCLHostFactsCache).
  cir::HostFactsOptions options;
  options.mode = *mode;
  options.assumeClosedTU = hostFactsClosedTU;
  options.bottomUp = hostFactsBottomUp;
  options.subobjects = hostFactsSubobjects;
  options.budgetScale = hostFactsBudgetScale;
  options.relevanceSlice = hostFactsSlice;
  options.maxSeconds = hostFactsMaxSeconds;
  options.maxMemoryMB = hostFactsMaxMemoryMB;
  options.internalize = hostFactsInternalize;
  cir::SYCLHostFactsCache &factsCache = getAnalysis<cir::SYCLHostFactsCache>();
  unsigned builtBefore = factsCache.getNumBuilt();
  unsigned tablesBefore = factsCache.getNumTablesBuilt();
  const cir::KernelBindingTable &table = factsCache.getTable(options);
  numHostFactsBuilt += factsCache.getNumBuilt() - builtBefore;
  numTablesBuilt += factsCache.getNumTablesBuilt() - tablesBefore;
  // The budgets that tripped while the analysis was built (once: the passes
  // share it).
  if (std::optional<std::string> trips =
          factsCache.get(options).takeBudgetReport())
    mlir::emitRemark(container.getLoc())
        << "host-facts budget trips: " << *trips << " "
        << cir::formatHostFactsOptions(options);
  bool changed = false;

  // Replace each scalar kernel parameter bound to an agreed-constant capture
  // field with a materialized cir.const. Returns whether anything was
  // replaced. With `notApplied`, replace nothing and remark why instead.
  auto applyConstArgs = [&](llvm::ArrayRef<cir::FuncOp> kernels,
                            const llvm::StringMap<mlir::TypedAttr> &consts,
                            llvm::StringRef notApplied = "") {
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
        std::string text;
        llvm::raw_string_ostream os(text);
        os << "kernel-arg-constant: kernel @" << kernel.getName() << " arg "
           << arg.getArgNumber() << " ('" << field << "'): " << value;
        if (!notApplied.empty()) {
          os << " not propagated (" << notApplied << ")";
          cir::emitOffloadRemark(
              kernel.getLoc(), cir::OffloadRemarkKind::Missed,
              "PropagateSYCLKernelArgs", cir::kKernelArgConstantRemarkCategory,
              kernel.getName(), text);
          continue;
        }

        mlir::OpBuilder builder(&kernel.getBody().front(),
                                kernel.getBody().front().begin());
        auto constant =
            cir::ConstantOp::create(builder, kernel.getLoc(), value);
        arg.replaceAllUsesWith(constant);
        did = true;
        cir::noteInternalizedSYCLKernelUse(kernel);
        os << cir::describeInternalizedSYCLKernel(kernel);
        cir::emitOffloadRemark(kernel.getLoc(), cir::OffloadRemarkKind::Passed,
                               "PropagateSYCLKernelArgs",
                               cir::kKernelArgConstantRemarkCategory,
                               kernel.getName(), text);
      }
    }
    return did;
  };

  std::string sharedNote = cir::getSharedSYCLKernelNote(hostFactsInternalize);
  for (const auto &entry : table) {
    const cir::KernelBinding &binding = entry.second;

    // An empty const-arg set means host-side recovery found no agreed constant
    // across the launches this TU sees.
    if (binding.kind != cir::BindingKind::SYCLName ||
        binding.launchSites.empty() || binding.syclConstArgs.empty())
      continue;

    if (binding.stampsInPlace(hostFactsClosedTU)) {
      // No other TU launches the kernel (or closed-TU is assumed): safe to
      // fold the constant into the shared device kernel in place.
      changed |= applyConstArgs(binding.deviceKernels, binding.syclConstArgs);
    } else {
      // Another TU can launch the kernel with another value: not in place;
      // the kernel copies of the internalized paths get the facts.
      applyConstArgs(binding.deviceKernels, binding.syclConstArgs,
                     hostFactsClosedTU
                         ? llvm::StringRef("the kernel is registered: it may "
                                           "be launched by name")
                         : llvm::StringRef(sharedNote));
    }
  }

  if (!changed)
    markAllAnalysesPreserved();
  else
    // SOUNDNESS: see SYCLHostFactsCache: the device-side changes leave the
    // host facts and the table true.
    markAnalysesPreserved<cir::SYCLHostFactsCache>();
}

} // namespace

std::unique_ptr<Pass> mlir::createOffloadPropagateSYCLKernelArgsPass() {
  return std::make_unique<OffloadPropagateSYCLKernelArgsPass>();
}
