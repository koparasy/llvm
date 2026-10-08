//===- TestCIRAliasAnalysis.cpp - Test CIR alias analysis results ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// -test-cir-alias-analysis and -test-cir-alias-analysis-modref print the
// alias (mod/ref) answer for every pair of values (op, value) tagged with
// `test.ptr`, in the format of MLIR's -test-alias-analysis.
//
// use-points-to=1 builds the points-to analysis (PointsToAnalysis, with the
// generic library models) of the module that holds the operation and
// registers CIRPointsToAliasAnalysis after CIRBasicAliasAnalysis.
// mode=bu runs the bottom-up mode (PointsToMode::BU).
// closed-tu=1 sets PointsToAnalysisOptions::assumeClosedTU, and sycl=1 adds
// the SYCL plug-in (addSYCLPointsToPlugin: the host API models and the
// device-write rule).
// points-to-only=1 prints the answers of the points-to analysis alone.
// explain=1 follows each MayAlias line with "  why: <explainAlias>".
//
// With use-points-to=1 every pair is also cross-checked (an assertion): when
// CIRBasicAliasAnalysis and the points-to analysis both give an answer that
// is not MayAlias, the answers must be the same. In particular a points-to
// NoAlias never meets a CIRBasicAliasAnalysis MustAlias.
//
// -test-cir-points-to-alias-sample runs on a module: it samples random pairs
// of pointer values of one function (pairs=<n>, seed=<s>) and prints the
// share of each answer and the time per query for CIRBasicAliasAnalysis
// alone, the points-to analysis alone, the default chain
// (registerCIRAliasAnalyses) and the chain with points-to.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/test/lib/Analysis/TestAliasAnalysis.h"
#include "clang/CIR/Dialect/Analysis/CIRAliasAnalysis.h"
#include "clang/CIR/Dialect/Analysis/CIRBasicAliasAnalysis.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/LocalBuilder.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToAnalysis.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToModels.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLPointsToModels.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <random>

using namespace mlir;
using namespace cir::pointsto;

namespace {

/// The points-to analysis of the module that holds an operation.
struct PointsToHolder {
  PointsToModelRegistry registry;
  std::unique_ptr<PointsToAnalysis> analysis;

  PointsToHolder(Operation *op, bool closedTU, bool sycl = false,
                 bool bottomUp = false) {
    registerGenericModels(registry);
    auto module = dyn_cast<ModuleOp>(op);
    if (!module)
      module = op->getParentOfType<ModuleOp>();
    PointsToAnalysisOptions options;
    options.assumeClosedTU = closedTU;
    if (bottomUp)
      options.mode = PointsToMode::BU;
    if (sycl)
      cir::addSYCLPointsToPlugin(options);
    analysis = std::make_unique<PointsToAnalysis>(module, &registry, options);
  }
};

/// The ops to test: the cir.func ops with a body in \p op if it is a module,
/// else \p op.
SmallVector<Operation *> getTestedOps(Operation *op) {
  SmallVector<Operation *> ops;
  if (!isa<ModuleOp>(op)) {
    ops.push_back(op);
    return ops;
  }
  op->walk([&](cir::FuncOp func) {
    if (!func.isDeclaration())
      ops.push_back(func);
  });
  return ops;
}

/// The values tagged in \p op, as TestAliasAnalysisBase collects them.
SmallVector<Value, 32> collectTaggedValues(Operation *op) {
  SmallVector<Value, 32> values;
  op->walk([&](Operation *op) {
    if (!op->getDiscardableAttr("test.ptr"))
      return;
    values.append(op->result_begin(), op->result_end());
    for (Region &region : op->getRegions())
      for (Block &block : region)
        values.append(block.args_begin(), block.args_end());
  });
  return values;
}

/// Assert that CIRBasicAliasAnalysis and the points-to analysis do not
/// contradict each other on (\p lhs, \p rhs).
void crossCheck(cir::CIRBasicAliasAnalysis &basic, const PointsToAnalysis &pt,
                Value lhs, Value rhs) {
  AliasResult basicResult = basic.alias(lhs, rhs);
  AliasResult ptResult = pt.alias(lhs, rhs);
  bool agree = basicResult.isMay() || ptResult.isMay() ||
               basicResult == ptResult;
  if (!agree) {
    llvm::errs() << "points-to cross-check failed: CIRBasicAliasAnalysis "
                 << basicResult << ", points-to " << ptResult << " ("
                 << pt.explainAlias(lhs, rhs) << ")\n  lhs: " << lhs
                 << "\n  rhs: " << rhs << "\n";
  }
  assert(agree && "points-to and CIRBasicAliasAnalysis contradict each other");
  (void)agree;
}

template <typename Derived>
struct PointsToOptions : public PassWrapper<Derived, OperationPass<>> {
  PointsToOptions() = default;
  PointsToOptions(const PointsToOptions &other)
      : PassWrapper<Derived, OperationPass<>>(other) {}

  // The points-to analysis makes CIR types even on an input without CIR ops
  // (the empty first chunk of a split input).
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cir::CIRDialect>();
  }

  Pass::Option<bool> usePointsTo{
      *this, "use-points-to",
      llvm::cl::desc("Register the points-to alias analysis last"),
      llvm::cl::init(false)};
  Pass::Option<bool> closedTU{
      *this, "closed-tu",
      llvm::cl::desc("Points-to: assume every caller is in the module"),
      llvm::cl::init(false)};
  Pass::Option<std::string> ptMode{
      *this, "mode",
      llvm::cl::desc("Points-to: ci (context-insensitive) or bu (bottom-up)"),
      llvm::cl::init("ci")};
  Pass::Option<bool> syclModels{
      *this, "sycl", llvm::cl::desc("Points-to: add the SYCL host API models"),
      llvm::cl::init(false)};
  Pass::Option<bool> pointsToOnly{
      *this, "points-to-only",
      llvm::cl::desc("Print the answers of the points-to analysis alone"),
      llvm::cl::init(false)};
  Pass::Option<bool> explainResults{
      *this, "explain",
      llvm::cl::desc("Explain every MayAlias answer of the points-to "
                     "analysis"),
      llvm::cl::init(false)};
};

//===----------------------------------------------------------------------===//
// Testing AliasResult
//===----------------------------------------------------------------------===//

struct TestCIRAliasAnalysisPass
    : public test::TestAliasAnalysisBase,
      PointsToOptions<TestCIRAliasAnalysisPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCIRAliasAnalysisPass)

  StringRef getArgument() const final { return "test-cir-alias-analysis"; }
  StringRef getDescription() const final {
    return "Test CIR alias analysis results.";
  }
  void runOnOperation() override {
    if (!usePointsTo && !pointsToOnly) {
      mlir::AliasAnalysis aliasAnalysis(getOperation());
      cir::registerCIRAliasAnalyses(aliasAnalysis, getOperation());
      runAliasAnalysisOnOperation(getOperation(), aliasAnalysis);
      return;
    }
    PointsToHolder holder(getOperation(), this->closedTU, this->syclModels,
                          this->ptMode == "bu");
    const PointsToAnalysis &pt = *holder.analysis;
    for (Operation *op : getTestedOps(getOperation())) {
      mlir::AliasAnalysis aliasAnalysis(op);
      cir::registerCIRAliasAnalyses(aliasAnalysis, op, &pt);
      cir::CIRBasicAliasAnalysis basic(op);
      llvm::errs() << "Testing : " << *op->getInherentAttr("sym_name") << "\n";
      SmallVector<Value, 32> values = collectTaggedValues(op);
      for (auto it = values.begin(), e = values.end(); it != e; ++it)
        for (auto *inner = values.begin(); inner != it; ++inner) {
          crossCheck(basic, pt, *inner, *it);
          AliasResult result = pointsToOnly ? pt.alias(*inner, *it)
                                            : aliasAnalysis.alias(*inner, *it);
          test::printAliasResult(result, *inner, *it);
          if (explainResults && result.isMay())
            llvm::errs() << "  why: " << pt.explainAlias(*inner, *it) << "\n";
        }
    }
  }
};

//===----------------------------------------------------------------------===//
// Testing ModRefResult
//===----------------------------------------------------------------------===//

struct TestCIRAliasAnalysisModRefPass
    : public test::TestAliasAnalysisModRefBase,
      PointsToOptions<TestCIRAliasAnalysisModRefPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCIRAliasAnalysisModRefPass)

  StringRef getArgument() const final {
    return "test-cir-alias-analysis-modref";
  }
  StringRef getDescription() const final {
    return "Test CIR alias analysis ModRef results.";
  }
  void runOnOperation() override {
    if (!usePointsTo && !pointsToOnly) {
      mlir::AliasAnalysis aliasAnalysis(getOperation());
      cir::registerCIRAliasAnalyses(aliasAnalysis, getOperation());
      runAliasAnalysisOnOperation(getOperation(), aliasAnalysis);
      return;
    }
    PointsToHolder holder(getOperation(), this->closedTU, this->syclModels,
                          this->ptMode == "bu");
    const PointsToAnalysis &pt = *holder.analysis;
    for (Operation *op : getTestedOps(getOperation())) {
      mlir::AliasAnalysis aliasAnalysis(op);
      cir::registerCIRAliasAnalyses(aliasAnalysis, op, &pt);
      llvm::errs() << "Testing : " << *op->getInherentAttr("sym_name") << "\n";
      SmallVector<Value, 32> values = collectTaggedValues(op);
      for (Value value : values)
        op->walk([&](Operation *tagged) {
          if (!tagged->getDiscardableAttr("test.ptr"))
            return;
          ModRefResult result = pointsToOnly
                                    ? pt.getModRef(tagged, value)
                                    : aliasAnalysis.getModRef(tagged, value);
          test::printModRefResult(result, tagged, value);
        });
    }
  }
};

//===----------------------------------------------------------------------===//
// Sampling random pairs
//===----------------------------------------------------------------------===//

struct TestCIRPointsToAliasSamplePass
    : public PassWrapper<TestCIRPointsToAliasSamplePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCIRPointsToAliasSamplePass)

  TestCIRPointsToAliasSamplePass() = default;
  TestCIRPointsToAliasSamplePass(const TestCIRPointsToAliasSamplePass &pass)
      : PassWrapper(pass) {}

  StringRef getArgument() const final {
    return "test-cir-points-to-alias-sample";
  }
  StringRef getDescription() const final {
    return "Compare CIR alias analyses on random pointer pairs.";
  }

  Option<unsigned> numPairs{*this, "pairs",
                            llvm::cl::desc("The number of pairs to sample"),
                            llvm::cl::init(10000)};
  Option<unsigned> seed{*this, "seed", llvm::cl::desc("The random seed"),
                        llvm::cl::init(1)};
  Option<bool> closedTU{
      *this, "closed-tu",
      llvm::cl::desc("Points-to: assume every caller is in the module"),
      llvm::cl::init(false)};
  Option<std::string> ptMode{
      *this, "mode",
      llvm::cl::desc("Points-to: ci (context-insensitive) or bu (bottom-up)"),
      llvm::cl::init("ci")};
  Option<std::string> showRule{
      *this, "show-rule",
      llvm::cl::desc("Print the first pairs (show=<n>) a points-to rule "
                     "decided, with their explanation"),
      llvm::cl::init("")};
  Option<unsigned> numShown{*this, "show",
                            llvm::cl::desc("The number of pairs show-rule "
                                           "prints"),
                            llvm::cl::init(5)};
  Option<bool> printRules{
      *this, "rules",
      llvm::cl::desc("Print the points-to rules that decided the pairs"),
      llvm::cl::init(true)};

  void runOnOperation() override;
};

/// A pointer value of a sampled function.
struct SampledValue {
  Value value;
  unsigned function;
};

void TestCIRPointsToAliasSamplePass::runOnOperation() {
  raw_ostream &os = llvm::outs();
  ModuleOp module = getOperation();
  // The innermost module with function bodies: the host module of an
  // offload container, or the module itself.
  ModuleOp target;
  module->walk([&](ModuleOp inner) {
    if (!target && llvm::any_of(inner.getOps<cir::FuncOp>(),
                                [](cir::FuncOp f) { return !f.isDeclaration(); }))
      target = inner;
  });
  if (!target) {
    os << "no function with a body\n";
    return;
  }

  llvm::TimeRecord start = llvm::TimeRecord::getCurrentTime(true);
  PointsToHolder holder(target, closedTU, /*sycl=*/false, ptMode == "bu");
  llvm::TimeRecord built = llvm::TimeRecord::getCurrentTime(false);
  built -= start;
  const PointsToAnalysis &pt = *holder.analysis;

  // Every pointer value (block arguments and op results) of every function.
  SmallVector<SmallVector<Value, 0>, 0> perFunction;
  SmallVector<SampledValue, 0> all;
  for (cir::FuncOp func : target.getOps<cir::FuncOp>()) {
    if (func.isDeclaration())
      continue;
    SmallVector<Value, 0> values;
    func.walk([&](Operation *op) {
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument arg : block.getArguments())
            if (isPointerValueType(arg.getType()))
              values.push_back(arg);
      if (op == func.getOperation())
        return;
      for (Value result : op->getResults())
        if (isPointerValueType(result.getType()))
          values.push_back(result);
    });
    if (values.size() < 2)
      continue;
    unsigned index = perFunction.size();
    for (Value value : values)
      all.push_back({value, index});
    perFunction.push_back(std::move(values));
  }
  if (all.empty()) {
    os << "no function with two pointer values\n";
    return;
  }

  // A value uniformly among all, then a different one of its function.
  std::mt19937_64 rng(seed);
  SmallVector<std::pair<Value, Value>, 0> pairs;
  pairs.reserve(numPairs);
  while (pairs.size() < numPairs) {
    const SampledValue &a =
        all[std::uniform_int_distribution<size_t>(0, all.size() - 1)(rng)];
    const SmallVector<Value, 0> &values = perFunction[a.function];
    Value b =
        values[std::uniform_int_distribution<size_t>(0, values.size() - 1)(
            rng)];
    if (a.value == b)
      continue;
    pairs.emplace_back(a.value, b);
  }

  struct Column {
    StringRef name;
    unsigned counts[4] = {0, 0, 0, 0};
    double seconds = 0;
  };
  auto kindIndex = [](AliasResult result) {
    if (result.isNo())
      return 0;
    if (result.isMay())
      return 1;
    if (result.isMust())
      return 2;
    return 3;
  };
  auto measure = [&](Column &column, auto &&query) {
    SmallVector<AliasResult, 0> results;
    results.reserve(pairs.size());
    llvm::TimeRecord begin = llvm::TimeRecord::getCurrentTime(true);
    for (auto [a, b] : pairs)
      results.push_back(query(a, b));
    llvm::TimeRecord time = llvm::TimeRecord::getCurrentTime(false);
    time -= begin;
    column.seconds = time.getWallTime();
    for (AliasResult result : results)
      ++column.counts[kindIndex(result)];
    return results;
  };

  Column basicCol{"CIRBasicAA"}, ptCol{"points-to"}, defaultCol{"default"},
      combinedCol{"combined"};
  cir::CIRBasicAliasAnalysis basic(target);
  SmallVector<AliasResult, 0> basicResults = measure(
      basicCol, [&](Value a, Value b) { return basic.alias(a, b); });
  SmallVector<AliasResult, 0> ptResults =
      measure(ptCol, [&](Value a, Value b) { return pt.alias(a, b); });
  mlir::AliasAnalysis defaultAA(target);
  cir::registerCIRAliasAnalyses(defaultAA, target);
  SmallVector<AliasResult, 0> defaultResults = measure(
      defaultCol, [&](Value a, Value b) { return defaultAA.alias(a, b); });
  mlir::AliasAnalysis combinedAA(target);
  cir::registerCIRAliasAnalyses(combinedAA, target, &pt);
  measure(combinedCol,
          [&](Value a, Value b) { return combinedAA.alias(a, b); });

  os << "=== points-to alias sample: " << pairs.size() << " pairs, seed "
     << seed << (closedTU ? ", closed-tu" : "")
     << (ptMode == "bu" ? ", bu" : "") << "\n";
  os << "functions sampled: " << perFunction.size()
     << ", pointer values: " << all.size() << "\n";
  os << llvm::format("points-to analysis: %.1f ms\n",
                     built.getWallTime() * 1000);
  os << llvm::format("%-12s %8s %8s %8s %8s %10s\n", "analysis", "NoAlias",
                     "MayAlias", "Must", "Partial", "ns/query");
  for (Column *column : {&basicCol, &ptCol, &defaultCol, &combinedCol}) {
    double n = pairs.size();
    os << llvm::format("%-12s %7.2f%% %7.2f%% %7.2f%% %7.2f%% %10.0f\n",
                       column->name.str().c_str(), 100 * column->counts[0] / n,
                       100 * column->counts[1] / n, 100 * column->counts[2] / n,
                       100 * column->counts[3] / n,
                       column->seconds * 1e9 / n);
  }

  // How the two analyses relate, pair by pair.
  unsigned contradictions = 0, defaultContradictions = 0, ptRefines = 0,
           basicRefines = 0;
  for (auto [i, pair] : llvm::enumerate(pairs)) {
    AliasResult b = basicResults[i], p = ptResults[i], d = defaultResults[i];
    if (!d.isMay() && !p.isMay() && d != p)
      ++defaultContradictions;
    if (!b.isMay() && !p.isMay() && b != p)
      ++contradictions;
    else if (b.isMay() && !p.isMay())
      ++ptRefines;
    else if (!b.isMay() && p.isMay())
      ++basicRefines;
  }
  os << "points-to refines a CIRBasicAA MayAlias: " << ptRefines
     << ", CIRBasicAA refines a points-to MayAlias: " << basicRefines
     << ", contradictions: " << contradictions
     << " (with the default chain: " << defaultContradictions << ")\n";

  if (printRules) {
    llvm::StringMap<unsigned> rules;
    for (auto [a, b] : pairs)
      ++rules[getAliasRuleName(pt.aliasWithRule(a, b).rule)];
    SmallVector<std::pair<StringRef, unsigned>, 0> sorted;
    for (const auto &entry : rules)
      sorted.emplace_back(entry.getKey(), entry.getValue());
    llvm::sort(sorted, [](const auto &x, const auto &y) {
      return x.second != y.second ? x.second > y.second : x.first < y.first;
    });
    os << "points-to rules:\n";
    for (auto [rule, count] : sorted)
      os << llvm::format("  %-26s %6u\n", rule.str().c_str(), count);
  }

  unsigned shown = 0;
  if (!showRule.empty())
    for (auto [a, b] : pairs) {
      if (shown == numShown)
        break;
      if (getAliasRuleName(pt.aliasWithRule(a, b).rule) != showRule)
        continue;
      ++shown;
      OpPrintingFlags flags;
      flags.skipRegions();
      os << "--- " << showRule << " in @"
         << a.getParentRegion()
                ->getParentOfType<cir::FuncOp>()
                .getSymName()
         << "\n  a: ";
      a.print(os, flags);
      os << "\n  b: ";
      b.print(os, flags);
      os << "\n  " << pt.explainAlias(a, b) << "\n";
    }
}

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace cir {
namespace test {
void registerTestCIRAliasAnalysisPass() {
  PassRegistration<TestCIRAliasAnalysisPass>();
  PassRegistration<TestCIRAliasAnalysisModRefPass>();
  PassRegistration<TestCIRPointsToAliasSamplePass>();
}
} // namespace test
} // namespace cir
