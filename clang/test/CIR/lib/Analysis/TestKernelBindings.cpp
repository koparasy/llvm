//===- TestKernelBindings.cpp - Print the kernel binding table ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test pass that constructs and prints a KernelBindingTable, so the analysis
// can be exercised through cir-opt + FileCheck (mirrors MLIR's test-print-*
// analysis passes).
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

using namespace mlir;

namespace {
struct PrintKernelBindingsPass
    : public PassWrapper<PrintKernelBindingsPass,
                         OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintKernelBindingsPass)

  PrintKernelBindingsPass() = default;
  PrintKernelBindingsPass(const PrintKernelBindingsPass &pass)
      : PassWrapper(pass) {}

  Option<std::string> hostFacts{
      *this, "host-facts",
      llvm::cl::desc("The host facts of the table: none (the bindings and "
                     "launch sites only, getAnalysis<KernelBindingTable>()) "
                     "or ipo (SYCLHostFacts)"),
      llvm::cl::init("none")};
  Option<bool> closedTU{
      *this, "closed-tu",
      llvm::cl::desc("Assume every caller of every function is in the module"),
      llvm::cl::init(false)};
  Option<bool> internalize{
      *this, "internalize",
      llvm::cl::desc("The launch paths were internalized "
                     "(HostFactsOptions::internalize)"),
      llvm::cl::init(false)};

  Option<bool> bottomUp{
      *this, "bu",
      llvm::cl::desc("Run the points-to analysis bottom-up "
                     "(the default; bu=0 runs it context-insensitively)"),
      llvm::cl::init(cir::HostFactsOptions().bottomUp)};
  Option<bool> subobjects{
      *this, "subobjects",
      llvm::cl::desc("Give class, union and array members "
                     "their own points-to nodes (the default)"),
      llvm::cl::init(cir::HostFactsOptions().subobjects)};

  Option<double> budgetScale{
      *this, "budget-scale",
      llvm::cl::desc("Multiply every budget of the points-to "
                     "analysis by this factor"),
      llvm::cl::init(1.0)};
  Option<bool> slice{
      *this, "slice",
      llvm::cl::desc("The relevance slice of the points-to "
                     "analysis"),
      llvm::cl::init(false)};
  Option<double> maxSeconds{
      *this, "max-seconds",
      llvm::cl::desc("The points-to analysis' time limit"),
      llvm::cl::init(cir::HostFactsOptions().maxSeconds)};

  StringRef getArgument() const final { return "test-print-kernel-bindings"; }
  StringRef getDescription() const final {
    return "Print the host<->device kernel bindings of an offload container.";
  }
  void runOnOperation() override {
    // The analysis only makes sense on an offload container module; leave any
    // other module alone.
    if (!cir::isOffloadContainer(getOperation()))
      return;
    std::optional<cir::HostFactsMode> mode = cir::parseHostFactsMode(hostFacts);
    if (!mode) {
      getOperation().emitError() << "unknown host-facts mode '" << hostFacts
                                 << "'";
      return signalPassFailure();
    }
    if (*mode == cir::HostFactsMode::None) {
      getAnalysis<cir::KernelBindingTable>().print(llvm::errs());
      return;
    }
    cir::HostFactsOptions options;
    options.mode = *mode;
    options.assumeClosedTU = closedTU;
    options.internalize = internalize;
    options.bottomUp = bottomUp;
    options.subobjects = subobjects;
    options.budgetScale = budgetScale;
    options.relevanceSlice = slice;
    options.maxSeconds = maxSeconds;
    cir::KernelBindingTable(getOperation(), options).print(llvm::errs());
  }
};
} // namespace

namespace cir {
namespace test {
void registerPrintKernelBindingsPass() {
  PassRegistration<PrintKernelBindingsPass>();
}
} // namespace test
} // namespace cir