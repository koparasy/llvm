//===- TestCIRPointsTo.cpp - Test the CIR points-to graphs ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builds the local points-to graph of every cir.func with a body and prints
// it, followed by the cell of every value tagged with a `test.ptr = "name"`
// attribute:
//
//   === @func
//   <FunctionGraph::print>
//   <name> -> <cell>                 for result 0 of a tagged op
//   <name>.region0#<i> -> <cell>     for the pointer arguments of a tagged
//                                    function
//
// Every graph is checked with verifyFunctionGraph (the -cir-points-to-verify
// self-check); a violation is an error on the offending op and fails the
// pass.
//
// With print-stats, a summary of the whole input follows: functions, live
// nodes, collapsed nodes, build time and the ops that went pessimistic.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/PointsTo/LocalBuilder.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToVerifier.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace cir::pointsto;

namespace {

struct TestCIRPointsToPass
    : public PassWrapper<TestCIRPointsToPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCIRPointsToPass)

  TestCIRPointsToPass() = default;
  TestCIRPointsToPass(const TestCIRPointsToPass &pass) : PassWrapper(pass) {}

  StringRef getArgument() const final { return "test-cir-points-to"; }
  StringRef getDescription() const final {
    return "Print the local CIR points-to graph of each function.";
  }

  Option<bool> printGraphs{*this, "print-graphs",
                           llvm::cl::desc("Print each function's graph"),
                           llvm::cl::init(true)};
  Option<bool> printStats{*this, "print-stats",
                          llvm::cl::desc("Print totals for the whole input"),
                          llvm::cl::init(false)};

  void runOnOperation() override;
};

void printCell(raw_ostream &os, StringRef name, std::optional<Cell> cell) {
  os << name << " -> ";
  Graph::printCell(os, cell ? *cell : Cell());
  os << "\n";
}

void printSlots(raw_ostream &os, StringRef name,
                const AggregateSlots &slots) {
  os << name << " -> {";
  for (auto [i, slot] : llvm::enumerate(slots)) {
    os << (i ? ", " : "") << slot.first << ": ";
    Graph::printCell(os, slot.second);
  }
  os << "}\n";
}

void TestCIRPointsToPass::runOnOperation() {
  raw_ostream &os = llvm::outs();
  SymbolTableCollection symbols;
  llvm::StringMap<unsigned> pessimisticOps, unhandledOps, unclassifiedOps;
  size_t numFuncs = 0, numNodes = 0, numCollapsed = 0, numOps = 0;
  llvm::TimeRecord buildTime;
  bool anyFailed = false;

  getOperation()->walk([&](cir::FuncOp func) {
    if (func.isDeclaration())
      return;
    llvm::TimeRecord start = llvm::TimeRecord::getCurrentTime(true);
    LocalBuilder builder(func, &symbols);
    std::unique_ptr<FunctionGraph> fg = builder.build();
    llvm::TimeRecord end = llvm::TimeRecord::getCurrentTime(false);
    end -= start;
    buildTime += end;

    // build() already verified the graph under -cir-points-to-verify.
    if (isPointsToVerifyEnabled() ? builder.hadVerifyErrors()
                                  : failed(verifyFunctionGraph(*fg)))
      anyFailed = true;

    ++numFuncs;
    func.walk([&](Operation *) { ++numOps; });
    for (const Node *node : fg->getGraph().nodes()) {
      ++numNodes;
      numCollapsed += node->isCollapsed();
    }
    for (const auto &entry : builder.getPessimisticOps())
      pessimisticOps[entry.getKey()] += entry.getValue();
    for (const auto &entry : builder.getUnhandledOps())
      unhandledOps[entry.getKey()] += entry.getValue();
    for (const auto &entry : builder.getUnclassifiedOps())
      unclassifiedOps[entry.getKey()] += entry.getValue();

    if (!printGraphs)
      return;
    os << "=== @" << func.getSymName() << "\n";
    fg->print(os);
    func.walk([&](Operation *op) {
      auto name = op->getAttrOfType<StringAttr>("test.ptr");
      if (!name)
        return;
      if (op == func.getOperation()) {
        for (BlockArgument arg : func.getArguments())
          if (std::optional<Cell> cell = fg->lookup(arg))
            printCell(os,
                      (name.getValue() + ".region0#" +
                       std::to_string(arg.getArgNumber()))
                          .str(),
                      cell);
        return;
      }
      if (op->getNumResults())
        if (const AggregateSlots *slots = fg->lookupSlots(op->getResult(0))) {
          printSlots(os, name.getValue(), *slots);
          return;
        }
      printCell(os, name.getValue(),
                op->getNumResults() ? fg->lookup(op->getResult(0))
                                    : std::nullopt);
    });
  });

  if (printStats) {
    auto printTop = [&](StringRef title, const llvm::StringMap<unsigned> &map) {
      SmallVector<std::pair<unsigned, StringRef>> sorted;
      for (const auto &entry : map)
        sorted.emplace_back(entry.getValue(), entry.getKey());
      llvm::sort(sorted, [](const auto &a, const auto &b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
      });
      os << title << ": " << sorted.size() << " op names\n";
      for (auto [count, name] : sorted)
        os << "  " << name << " " << count << "\n";
    };
    os << "--- stats\n"
       << "functions: " << numFuncs << "\n"
       << "ops: " << numOps << "\n"
       << "live nodes: " << numNodes << "\n"
       << "collapsed nodes: " << numCollapsed << "\n"
       << "build time (wall, s): " << buildTime.getWallTime() << "\n";
    printTop("pessimistic ops (touching pointers or memory)", pessimisticOps);
    printTop("ops sent to the pessimistic default", unhandledOps);
    printTop("unclassified ops", unclassifiedOps);
  }

  if (anyFailed)
    signalPassFailure();
}

} // namespace

namespace cir {
namespace test {
void registerTestCIRPointsToPass() { PassRegistration<TestCIRPointsToPass>(); }
} // namespace test
} // namespace cir
