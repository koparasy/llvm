//===- TestCopyKernel.cpp - Exercise the copyKernel primitive -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test pass that drives cir::specializeSYCLKernelForLaunch ("copyKernel"): for
// every SYCL binding with a launch site, it mints one specialized clone and
// redirects the FIRST launch site to it, leaving the original generic. This
// exercises the primitive through cir-opt + FileCheck so the minted clone and
// the redirected marker can be inspected structurally, before any optimization
// consumes it.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"

using namespace mlir;

namespace {
struct CopyKernelPass
    : public PassWrapper<CopyKernelPass,
                         OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CopyKernelPass)

  StringRef getArgument() const final { return "test-copy-kernel"; }
  StringRef getDescription() const final {
    return "Specialize each launched SYCL kernel by cloning it and redirecting "
           "its first launch site (exercises copyKernel).";
  }
  void runOnOperation() override {
    mlir::ModuleOp container = getOperation();
    if (!cir::isOffloadContainer(container))
      return;
    cir::KernelBindingTable &table = getAnalysis<cir::KernelBindingTable>();

    unsigned tag = 0;
    for (const auto &entry : table) {
      const cir::KernelBinding &binding = entry.second;
      if (binding.launchSites.empty())
        continue;
      cir::specializeSYCLKernelForLaunches(container, binding,
                                           binding.launchSites.front(),
                                           std::to_string(tag++));
    }
  }
};
} // namespace

namespace cir {
namespace test {
void registerCopyKernelPass() { PassRegistration<CopyKernelPass>(); }
} // namespace test
} // namespace cir
