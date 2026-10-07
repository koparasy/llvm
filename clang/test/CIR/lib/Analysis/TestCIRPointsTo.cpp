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
// Calls get the generic library models (PointsToModels.h) unless models=0.
//
// With print-stats, a summary of the whole input follows: functions, live
// nodes, collapsed and External nodes, build time, the ops that went
// pessimistic, the call sites by kind and by model, and the declarations
// that got the unknown-call rule most often.
//
// With mode=ci, the context-insensitive interprocedural analysis
// (PointsToAnalysis) runs instead, once per module that holds functions with
// a body, and prints:
//
//   === TU graph
//   <Graph::print of the TU graph>
//   === @func formals=complete|incomplete
//   @func formal #<i> -> <cell or slots>
//   @func return -> <cell or slots>
//   @func <name> -> <cell or slots>   for each tagged value, as above
//   === calls
//   @caller <op> [@callee] -> direct @f | model <name> | unknown
//                             | indirect -> {@f, @g}
//                             | indirect-unresolved (<Kind>, "<detail>")
//                               [bound={@f, ...}]
//
// closed-tu=1 sets PointsToAnalysisOptions::assumeClosedTU and
// max-indirect-rounds=<n> PointsToAnalysisOptions::maxIndirectRounds.
// print-graphs=0 leaves out everything but the statistics, which end with one
// line per unresolved indirect call: its location, its reason, and the first
// reason the callee node got External or Incomplete.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/PointsTo/LocalBuilder.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToAnalysis.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToModels.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToVerifier.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Path.h"
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
  Option<bool> useModels{*this, "models",
                         llvm::cl::desc("Apply the generic library models"),
                         llvm::cl::init(true)};
  Option<std::string> mode{
      *this, "mode",
      llvm::cl::desc("local: the local graph of each function; ci: the "
                     "context-insensitive interprocedural analysis"),
      llvm::cl::init("local")};
  Option<bool> closedTU{*this, "closed-tu",
                        llvm::cl::desc("In ci mode, assume every caller of "
                                       "every function is in the module"),
                        llvm::cl::init(false)};
  Option<unsigned> maxIndirectRounds{
      *this, "max-indirect-rounds",
      llvm::cl::desc("In ci mode, the rounds of indirect-call resolution "
                     "that may bind new targets"),
      llvm::cl::init(PointsToAnalysisOptions().maxIndirectRounds)};

  void runOnOperation() override;
  void runLocal();
  void runCI();
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

std::string formatTargets(const TUCallSite &call) {
  std::string out = "{";
  for (auto [i, target] : llvm::enumerate(call.targets))
    out += (i ? ", @" : "@") + target.getValue().str();
  return out + "}";
}

std::string formatReason(const Reason *why) {
  std::string out = getReasonKindName(why->kind).str();
  if (!why->detail.empty())
    out += ", \"" + why->detail.str() + "\"";
  return out;
}

/// The op of \p why and the first op of its causes, as "name file:line".
std::string formatReasonSite(const Reason *why) {
  for (; why; why = why->cause) {
    if (!why->op)
      continue;
    std::string out = why->op->getName().getStringRef().str();
    if (auto loc = dyn_cast<FileLineColLoc>(why->op->getLoc()))
      out += " " + llvm::sys::path::filename(loc.getFilename().getValue())
                       .str() +
             ":" + std::to_string(loc.getLine());
    if (auto func = why->op->getParentOfType<cir::FuncOp>())
      out += " in @" + func.getSymName().str();
    else if (auto func = dyn_cast<cir::FuncOp>(why->op))
      out += " @" + func.getSymName().str();
    return out;
  }
  return "-";
}

/// The root of the chain of \p why: the last cause.
const Reason *rootReason(const Reason *why) {
  while (why && why->cause)
    why = why->cause;
  return why;
}

void TestCIRPointsToPass::runOnOperation() {
  if (mode == "ci")
    return runCI();
  if (mode != "local") {
    getOperation()->emitError() << "unknown test-cir-points-to mode '"
                                << mode << "'";
    return signalPassFailure();
  }
  runLocal();
}

/// Print the tagged values of \p func with \p prefix before each line.
template <typename LookupCell, typename LookupSlots>
static void printTagged(raw_ostream &os, cir::FuncOp func, StringRef prefix,
                        LookupCell lookupCell, LookupSlots lookupSlots) {
  func.walk([&](Operation *op) {
    auto name = op->getAttrOfType<StringAttr>("test.ptr");
    if (!name)
      return;
    std::string label = (prefix + name.getValue()).str();
    if (op == func.getOperation()) {
      for (BlockArgument arg : func.getArguments())
        if (std::optional<Cell> cell = lookupCell(arg))
          printCell(os, label + ".region0#" + std::to_string(arg.getArgNumber()),
                    cell);
      return;
    }
    if (op->getNumResults())
      if (const AggregateSlots *slots = lookupSlots(op->getResult(0))) {
        printSlots(os, label, *slots);
        return;
      }
    printCell(os, label,
              op->getNumResults() ? lookupCell(op->getResult(0)) : std::nullopt);
  });
}

void TestCIRPointsToPass::runCI() {
  raw_ostream &os = llvm::outs();
  PointsToModelRegistry registry;
  if (useModels)
    registerGenericModels(registry);
  PointsToAnalysisOptions options;
  options.assumeClosedTU = closedTU;
  options.maxIndirectRounds = maxIndirectRounds;
  bool anyFailed = false;

  SmallVector<ModuleOp> modules;
  getOperation()->walk([&](ModuleOp module) {
    if (llvm::any_of(module.getOps<cir::FuncOp>(),
                     [](cir::FuncOp func) { return !func.isDeclaration(); }))
      modules.push_back(module);
  });
  for (ModuleOp module : modules) {
    PointsToAnalysis analysis(module, &registry, options);
    if (analysis.hadVerifyErrors())
      anyFailed = true;
    const Graph &graph = analysis.getGraph();
    if (printGraphs) {
      os << "=== TU graph\n";
      graph.print(os);
      auto lookupCell = [&](Value value) -> std::optional<Cell> {
        if (Cell cell = analysis.cellOf(value))
          return cell;
        return std::nullopt;
      };
      auto lookupSlots = [&](Value value) { return analysis.slotsOf(value); };
      for (auto func : module.getOps<cir::FuncOp>()) {
        const TUFunction *fn = analysis.getFunction(func);
        if (!fn)
          continue;
        std::string prefix = ("@" + func.getSymName() + " ").str();
        os << "=== @" << func.getSymName() << " formals="
           << (fn->formalsComplete ? "complete" : "incomplete") << "\n";
        for (auto [i, formal] : llvm::enumerate(fn->formals)) {
          std::string label = prefix + "formal #" + std::to_string(i);
          if (formal)
            printCell(os, label, formal);
          else if (fn->formalSlots[i])
            printSlots(os, label, *fn->formalSlots[i]);
        }
        if (fn->returnCell)
          printCell(os, prefix + "return", fn->returnCell);
        if (fn->returnSlots)
          printSlots(os, prefix + "return", *fn->returnSlots);
        printTagged(os, func, prefix, lookupCell, lookupSlots);
      }
      os << "=== calls\n";
      for (auto func : module.getOps<cir::FuncOp>()) {
        const TUFunction *fn = analysis.getFunction(func);
        if (!fn)
          continue;
        for (const TUCallSite &call : fn->callSites) {
          os << "@" << func.getSymName() << " " << call.site.call->getName();
          if (call.site.callee)
            os << " " << call.site.callee;
          if (call.site.synthetic)
            os << " synthetic";
          os << " -> " << getCallResolutionName(call.resolution);
          if (call.resolution == CallResolution::Direct)
            os << " @" << cir::FuncOp(call.callee).getSymName();
          else if (call.resolution == CallResolution::Model)
            os << " " << call.site.modelled;
          else if (call.resolution == CallResolution::Indirect)
            os << " -> " << formatTargets(call);
          else if (call.resolution == CallResolution::IndirectUnresolved) {
            if (call.unresolvedWhy)
              os << " (" << formatReason(call.unresolvedWhy) << ")";
            if (!call.targets.empty())
              os << " bound=" << formatTargets(call);
          }
          os << "\n";
        }
      }
    }

    if (!printStats)
      continue;
    const PointsToAnalysisStats &stats = analysis.getStats();
    llvm::StringMap<unsigned> collapseReasons, resolutions;
    size_t numNodes = 0, numCollapsed = 0, numExternal = 0, numIncomplete = 0;
    for (const Node *node : graph.nodes()) {
      ++numNodes;
      if (node->hasFlags(NodeFlags::External))
        ++numExternal;
      if (node->hasFlags(NodeFlags::Incomplete))
        ++numIncomplete;
      if (!node->isCollapsed())
        continue;
      ++numCollapsed;
      std::string key = "<no reason>";
      if (const Reason *why = node->getReason(NodeFlags::Collapsed)) {
        key = getReasonKindName(why->kind).str() + "(" +
              (why->op ? why->op->getName().getStringRef().str() : "-");
        if (!why->detail.empty())
          key += ", \"" + why->detail.str() + "\"";
        key += ")";
      }
      ++collapseReasons[key];
    }
    for (const auto &fn : analysis.getFunctions())
      for (const TUCallSite &call : fn->callSites)
        ++resolutions[getCallResolutionName(call.resolution)];
    os << "--- ci stats\n"
       << "functions: " << stats.numFunctions << "\n"
       << "functions with complete formals: " << stats.numCompleteFunctions
       << "\n"
       << "local nodes: " << stats.numLocalNodes << "\n"
       << "TU live nodes: " << numNodes << "\n"
       << "collapsed nodes: " << numCollapsed << "\n"
       << "external nodes: " << numExternal << "\n"
       << "incomplete nodes: " << numIncomplete << "\n"
       << "direct calls: " << stats.numDirectCalls << "\n"
       << "bindings unified: " << stats.numBindings << "\n"
       << "indirect calls resolved: " << stats.numIndirectResolved << "\n"
       << "indirect calls unresolved: " << stats.numIndirectUnresolved << "\n"
       << "indirect targets bound: " << stats.numIndirectTargetsBound << "\n"
       << "indirect rounds that bound a target: " << stats.numIndirectRounds
       << "\n"
       << "indirect calls over the round budget: " << stats.numIndirectBudget
       << "\n"
       << "clobbering calls: " << stats.numClobberingCalls << "\n"
       << "copy closure settled: " << (stats.transfersSettled ? "yes" : "no")
       << "\n"
       << "local phase time (wall, s): " << stats.localSeconds << "\n"
       << "analysis time (wall, s): " << stats.totalSeconds << "\n";
    auto printMap = [&](StringRef title, const llvm::StringMap<unsigned> &map) {
      SmallVector<std::pair<unsigned, StringRef>> sorted;
      for (const auto &entry : map)
        sorted.emplace_back(entry.getValue(), entry.getKey());
      llvm::sort(sorted, [](const auto &a, const auto &b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
      });
      os << title << ": " << sorted.size() << "\n";
      for (auto [count, name] : sorted)
        os << "  " << name << " " << count << "\n";
    };
    printMap("call sites by resolution", resolutions);
    printMap("collapsed nodes by reason", collapseReasons);
    llvm::StringMap<unsigned> unresolvedReasons;
    for (const auto &fn : analysis.getFunctions())
      for (const TUCallSite &call : fn->callSites)
        if (call.resolution == CallResolution::IndirectUnresolved &&
            call.unresolvedWhy)
          ++unresolvedReasons[formatReason(call.unresolvedWhy)];
    printMap("unresolved indirect calls by reason", unresolvedReasons);
    os << "unresolved indirect calls:\n";
    for (const auto &fn : analysis.getFunctions())
      for (const TUCallSite &call : fn->callSites) {
        if (call.resolution != CallResolution::IndirectUnresolved)
          continue;
        os << "  " << formatReasonSite(call.unresolvedWhy) << ": "
           << (call.unresolvedWhy ? formatReason(call.unresolvedWhy) : "-");
        if (!call.targets.empty())
          os << " bound=" << formatTargets(call);
        if (call.site.calleePtr) {
          const Node *callee = call.site.calleePtr->getNode();
          for (NodeFlags flag : {NodeFlags::External, NodeFlags::Incomplete})
            if (const Reason *why = callee->getReason(flag)) {
              const Reason *root = rootReason(why);
              os << "; " << getFlagName(flag) << " from "
                 << formatReason(root) << " at " << formatReasonSite(root);
            }
          os << "; callee n" << callee->getId() << " origins="
             << callee->getOrigins().size();
        }
        os << "\n";
      }
  }
  if (anyFailed)
    signalPassFailure();
}

void TestCIRPointsToPass::runLocal() {
  raw_ostream &os = llvm::outs();
  SymbolTableCollection symbols;
  llvm::StringMap<unsigned> pessimisticOps, unhandledOps, unclassifiedOps,
      collapseReasons, callKinds, callModels, unknownDecls;
  size_t numFuncs = 0, numNodes = 0, numCollapsed = 0, numOps = 0,
         numExternal = 0, numCalls = 0;
  PointsToModelRegistry registry;
  if (useModels)
    registerGenericModels(registry);
  llvm::TimeRecord buildTime;
  bool anyFailed = false;

  getOperation()->walk([&](cir::FuncOp func) {
    if (func.isDeclaration())
      return;
    llvm::TimeRecord start = llvm::TimeRecord::getCurrentTime(true);
    LocalBuilder builder(func, &symbols, /*reasons=*/nullptr, &registry);
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
    for (const CallSiteRecord &site : fg->callSites) {
      ++numCalls;
      ++callKinds[getCallKindName(site.kind)];
      if (!site.modelled.empty())
        ++callModels[site.modelled];
      if (site.kind == CallKind::UnknownDecl)
        ++unknownDecls[site.callee ? site.callee.getValue() : "<missing>"];
    }
    for (const Node *node : fg->getGraph().nodes()) {
      ++numNodes;
      if (node->hasFlags(NodeFlags::External))
        ++numExternal;
      if (!node->isCollapsed())
        continue;
      ++numCollapsed;
      std::string key;
      if (const Reason *why = node->getReason(NodeFlags::Collapsed)) {
        key = getReasonKindName(why->kind).str();
        key += "(";
        key += why->op ? why->op->getName().getStringRef().str() : "-";
        if (!why->detail.empty())
          key += ", \"" + why->detail.str() + "\"";
        key += ")";
      } else {
        key = "<no reason>";
      }
      ++collapseReasons[key];
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
    auto printTop = [&](StringRef title, const llvm::StringMap<unsigned> &map,
                        StringRef unit = "op names", size_t limit = ~0u) {
      SmallVector<std::pair<unsigned, StringRef>> sorted;
      for (const auto &entry : map)
        sorted.emplace_back(entry.getValue(), entry.getKey());
      llvm::sort(sorted, [](const auto &a, const auto &b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
      });
      os << title << ": " << sorted.size() << " " << unit << "\n";
      for (auto [count, name] : llvm::ArrayRef(sorted).take_front(limit))
        os << "  " << name << " " << count << "\n";
    };
    os << "--- stats\n"
       << "functions: " << numFuncs << "\n"
       << "ops: " << numOps << "\n"
       << "live nodes: " << numNodes << "\n"
       << "collapsed nodes: " << numCollapsed << "\n"
       << "external nodes: " << numExternal << "\n"
       << "build time (wall, s): " << buildTime.getWallTime() << "\n";
    printTop("collapsed nodes by reason", collapseReasons, "reasons");
    printTop("pessimistic ops (touching pointers or memory)", pessimisticOps);
    printTop("ops sent to the pessimistic default", unhandledOps);
    printTop("unclassified ops", unclassifiedOps);
    os << "call sites: " << numCalls << "\n";
    printTop("call sites by kind", callKinds, "kinds");
    printTop("modelled call sites by model", callModels, "models");
    printTop("unknown declarations by call count", unknownDecls,
             "declarations", ~0u);
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
