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
//   @func <name> -> <cell> complete|incomplete
//                                     for each tagged pointer value (slots
//                                     for an aggregate), as above; with
//                                     explain=1 followed by " (<explain>)"
//                                     and one "    <- <reason>" line per
//                                     link of the reason chain
//   === calls
//   @caller <op> [@callee] -> direct @f | model <name> | unknown
//                             | indirect -> {@f, @g}
//                             | indirect-unresolved (<Kind>, "<detail>")
//                               [bound={@f, ...}]
//
// sycl=1 adds the SYCL plug-in (OffloadOpt's addSYCLPointsToPlugin: the
// host API models and the device-write rule) and, in ci mode, prints the
// kernel launches they record and the noalias verdicts of SYCLHostFacts,
// even with print-graphs=0:
//
//   === launches
//   launch @<func> <model> kernels={<name>, ...} closure=<cell>
//       geometry=<cell>
//     kernel <name>[ (no device function)]
//       field <name> accessor storage=<node> [<flags>] origins=<n>
//           host=<node [<flags>] or none> <complete | unknown (<why>)>
//       field <name> pointer target=<node> [<flags>] <complete | ...>
//       blocked: <why>          the candidates' poison, if any
//       same <a> <b>            for each pair with the same memory
//       no-alias <field> ...    the fields no-alias at this launch
//   kernel <name>: no-alias <field> ...   SYCLHostFacts::noAliasFields,
//                                         over every launch of the kernel
//     note <field>: <note>                with explain=1
//
// The fields are the noalias candidates of the kernel's device function
// (getSYCLNoAliasCandidates), by name; their identities and verdicts are
// SYCLHostFacts::identityOfClosureField and noAliasFields.
//
// mode=bu runs the bottom-up mode (PointsToMode::BU) and prints the same,
// with the calls annotated by how each body was bound:
//
//   ... [clone #<ctx>] | [merged (<Kind>, "<detail>")]
//       | [fallback (<Kind>, "<detail>")]
//
// then the call sites each function's graph inherited from copies, and the
// calling contexts:
//
//   === inherited calls
//   @<graph> <op> in @<func> path=#<ctx> -> <resolution as above>
//   === contexts
//   #<id> = <op> <file:line> in @<func> [< #<parent>]
//
// The stats then start with "--- bu stats" and add the copy counters.
// contexts=1 also prints each tagged value of a function in the contexts of
// the calls that copied its graph, and in all of its calling contexts
// (pointsToInContexts; see printInContexts).
// max-clone-nodes=<n> and max-total-clone-nodes=<n> set the budgets.
//
// subobjects=1 sets PointsToAnalysisOptions::subobjects (in local mode,
// LocalBuilder::setSubobjects); the stats then count the subobject links and
// the nodes with a parent.
//
// closed-tu=1 sets PointsToAnalysisOptions::assumeClosedTU and
// max-indirect-rounds=<n> PointsToAnalysisOptions::maxIndirectRounds.
// values=1 (ci and bu modes) prints, even with print-graphs=0, what every
// cir.load tagged with `test.load = "name"` may read (StoredValues.h:
// PointsToAnalysis::valuesAtInContexts, the top-down union in bu mode):
//
//   === values
//   <name>: {<const>, ...} defs=<n> complete|incomplete[ (<Kind>, "<detail>")]
//       [contexts=<n>]
//
// max-query-visits=<n> sets the stored-value query budget.
//
// print-graphs=0 leaves out everything but the statistics, which end with one
// line per unresolved indirect call: its location, its reason, and the first
// reason the callee node got External or Incomplete. The statistics also
// count the pointer values with a complete and an incomplete points-to set,
// the Summary nodes by reason, and list the most frequent explanations of
// the incomplete ones (top=<n>, default 10), with and without locations.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/PointsTo/LocalBuilder.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToAnalysis.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToModels.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToVerifier.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/StoredValues.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLHostFacts.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLKernelParams.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLPointsToModels.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
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
  Option<bool> syclModels{*this, "sycl",
                          llvm::cl::desc("Add the SYCL host API models and, in "
                                         "ci mode, print the kernel launches"),
                          llvm::cl::init(false)};
  Option<bool> closedTU{*this, "closed-tu",
                        llvm::cl::desc("In ci mode, assume every caller of "
                                       "every function is in the module"),
                        llvm::cl::init(false)};
  Option<bool> explainSets{
      *this, "explain",
      llvm::cl::desc("In ci mode, explain the points-to set of each tagged "
                     "value and print its reason chain"),
      llvm::cl::init(false)};
  Option<unsigned> topExplanations{
      *this, "top",
      llvm::cl::desc("In ci mode with print-stats, the number of most "
                     "frequent explanations to list"),
      llvm::cl::init(10)};
  Option<unsigned> maxIndirectRounds{
      *this, "max-indirect-rounds",
      llvm::cl::desc("In ci mode, the rounds of indirect-call resolution "
                     "that may bind new targets"),
      llvm::cl::init(PointsToAnalysisOptions().maxIndirectRounds)};
  Option<bool> printContexts{
      *this, "contexts",
      llvm::cl::desc("In bu mode, also print each tagged pointer value in the "
                     "calling contexts of one and two calls "
                     "(PointsToAnalysis::pointsTo(value, callPath))"),
      llvm::cl::init(false)};
  Option<bool> subobjects{
      *this, "subobjects",
      llvm::cl::desc("Give every member of class, union or array type its own "
                     "child node (PointsToAnalysisOptions::subobjects)"),
      llvm::cl::init(false)};
  Option<unsigned> maxCloneNodes{
      *this, "max-clone-nodes",
      llvm::cl::desc("In bu mode, the most nodes of one function's graph"),
      llvm::cl::init(PointsToAnalysisOptions().maxCloneNodes)};
  Option<unsigned> maxTotalCloneNodes{
      *this, "max-total-clone-nodes",
      llvm::cl::desc("In bu mode, the most nodes of all copies"),
      llvm::cl::init(PointsToAnalysisOptions().maxTotalCloneNodes)};

  Option<bool> printValues{
      *this, "values",
      llvm::cl::desc("Print what each cir.load tagged test.load may read"),
      llvm::cl::init(false)};
  Option<unsigned> maxQueryVisits{
      *this, "max-query-visits",
      llvm::cl::desc("The stored-value query budget"), llvm::cl::init(4096)};

  void runOnOperation() override;
  void runLocal();
  void runCI();
  void printLaunches(raw_ostream &os, ModuleOp module,
                     const PointsToAnalysis &analysis);
};

void printCell(raw_ostream &os, StringRef name, std::optional<Cell> cell,
               StringRef suffix = {}) {
  os << name << " -> ";
  Graph::printCell(os, cell ? *cell : Cell());
  os << suffix << "\n";
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

/// BU: each tagged pointer value of \p func as seen through the copies of
/// its graph at one call and at two calls:
///   @func <name> in [<call> file:line, ...] -> <cell>, ... complete|incomplete
void printInContexts(raw_ostream &os, const PointsToAnalysis &analysis,
                     cir::FuncOp func, const TUFunction &fn) {
  auto where = [](Operation *call) {
    std::string out = call->getName().getStringRef().str();
    if (auto loc = dyn_cast<FileLineColLoc>(call->getLoc()))
      out += " " + llvm::sys::path::filename(loc.getFilename().getValue())
                       .str() +
             ":" + std::to_string(loc.getLine());
    return out;
  };
  // The paths: the calls whose copy of fn's graph sits in the caller's own
  // graph, and one more call above each.
  SmallVector<SmallVector<Operation *, 2>, 4> paths;
  const CallPathTable &table = analysis.getCallPaths();
  for (const auto &clone : analysis.getClones()) {
    if (clone->callee != &fn || table.getPath(clone->ctx).size() != 1)
      continue;
    paths.push_back({clone->call});
    for (const auto &outer : analysis.getClones())
      if (outer->callee == clone->caller &&
          table.getPath(outer->ctx).size() == 1)
        paths.push_back({outer->call, clone->call});
  }
  func.walk([&](Operation *op) {
    auto name = op->getAttrOfType<StringAttr>("test.ptr");
    if (!name || op == func.getOperation() || op->getNumResults() != 1 ||
        !isPointerValueType(op->getResult(0).getType()))
      return;
    for (ArrayRef<Operation *> path : paths) {
      ObjectSet set = analysis.pointsTo(op->getResult(0), path);
      os << "@" << func.getSymName() << " " << name.getValue() << " in [";
      llvm::interleaveComma(path, os,
                            [&](Operation *call) { os << where(call); });
      os << "] ->";
      for (Cell cell : set.cells) {
        os << " ";
        Graph::printCell(os, cell);
      }
      os << (set.complete ? " complete" : " incomplete") << "\n";
    }
    // The top-down set: every calling context of a covered function.
    unsigned numContexts = 0;
    ObjectSet all = analysis.pointsToInContexts(op->getResult(0), &numContexts);
    os << "@" << func.getSymName() << " " << name.getValue() << " in "
       << (analysis.isCovered(func) ? "" : "uncovered, ") << numContexts
       << " contexts ->";
    for (Cell cell : all.cells) {
      os << " ";
      Graph::printCell(os, cell);
    }
    os << (all.complete ? " complete" : " incomplete") << "\n";
  });
}

/// " -> <resolution>" of \p call, then (BU) how each body was bound.
void printResolution(raw_ostream &os, const PointsToAnalysis &analysis,
                     const TUCallSite &call) {
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
  if (analysis.getMode() != PointsToMode::BU)
    return;
  if (call.fallback) {
    os << " [fallback (" << formatReason(call.notClonedWhy) << ")]";
    return;
  }
  for (const TUCallSite::Binding &binding : call.bindings) {
    os << " [";
    if (call.resolution != CallResolution::Direct)
      os << "@" << cir::FuncOp(binding.callee).getSymName() << " ";
    if (binding.clone >= 0)
      os << "clone #" << analysis.getClones()[binding.clone]->ctx;
    else
      os << "merged ("
         << (call.notClonedWhy ? formatReason(call.notClonedWhy)
                               : std::string("-"))
         << ")";
    os << "]";
  }
}

void TestCIRPointsToPass::runOnOperation() {
  if (mode == "ci" || mode == "bu")
    return runCI();
  if (mode != "local") {
    getOperation()->emitError() << "unknown test-cir-points-to mode '"
                                << mode << "'";
    return signalPassFailure();
  }
  runLocal();
}

/// Print the tagged values of \p func with \p prefix before each line. The
/// line of a pointer value ends with \p suffix(value).
template <typename LookupCell, typename LookupSlots, typename Suffix>
static void printTagged(raw_ostream &os, cir::FuncOp func, StringRef prefix,
                        LookupCell lookupCell, LookupSlots lookupSlots,
                        Suffix suffix) {
  func.walk([&](Operation *op) {
    auto name = op->getAttrOfType<StringAttr>("test.ptr");
    if (!name)
      return;
    std::string label = (prefix + name.getValue()).str();
    if (op == func.getOperation()) {
      for (BlockArgument arg : func.getArguments())
        if (std::optional<Cell> cell = lookupCell(arg))
          printCell(os, label + ".region0#" + std::to_string(arg.getArgNumber()),
                    cell, suffix(arg));
      return;
    }
    if (op->getNumResults())
      if (const AggregateSlots *slots = lookupSlots(op->getResult(0))) {
        printSlots(os, label, *slots);
        return;
      }
    if (!op->getNumResults())
      return printCell(os, label, std::nullopt);
    printCell(os, label, lookupCell(op->getResult(0)),
              suffix(op->getResult(0)));
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
  options.subobjects = subobjects;
  bool bottomUp = mode == "bu";
  if (bottomUp) {
    options.mode = PointsToMode::BU;
    options.maxCloneNodes = maxCloneNodes;
    options.maxTotalCloneNodes = maxTotalCloneNodes;
  }
  if (syclModels)
    cir::addSYCLPointsToPlugin(options);
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
        auto suffix = [&](Value value) {
          ObjectSet set = analysis.pointsTo(value);
          std::string out = set.complete ? " complete" : " incomplete";
          if (!explainSets)
            return out;
          out += " (" + explain(set) + ")";
          const Reason *why = set.why;
          if (set.complete && !set.cells.empty())
            why = set.cells.front().getNode()->getReason(NodeFlags::Summary);
          for (const std::string &link : explainChain(why))
            out += "\n    <- " + link;
          return out;
        };
        printTagged(os, func, prefix, lookupCell, lookupSlots, suffix);
        if (bottomUp && printContexts)
          printInContexts(os, analysis, func, *fn);
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
          printResolution(os, analysis, call);
          os << "\n";
        }
      }
      if (bottomUp) {
        os << "=== inherited calls\n";
        for (auto func : module.getOps<cir::FuncOp>()) {
          const TUFunction *fn = analysis.getFunction(func);
          if (!fn)
            continue;
          for (const auto &call : fn->inheritedCalls) {
            os << "@" << func.getSymName() << " "
               << call->site.call->getName() << " in @"
               << call->site.call->getParentOfType<cir::FuncOp>().getSymName()
               << " prefix=#" << call->prefix;
            printResolution(os, analysis, *call);
            os << "\n";
          }
        }
        os << "=== contexts\n";
        const CallPathTable &paths = analysis.getCallPaths();
        for (uint32_t id = 1; id <= paths.size(); ++id) {
          Operation *call = paths.getCall(id);
          os << "#" << id << " = " << call->getName();
          if (auto loc = dyn_cast<FileLineColLoc>(call->getLoc()))
            os << " " << llvm::sys::path::filename(loc.getFilename().getValue())
               << ":" << loc.getLine();
          if (auto f = call->getParentOfType<cir::FuncOp>())
            os << " in @" << f.getSymName();
          if (uint32_t parent = paths.getParent(id))
            os << " < #" << parent;
          os << "\n";
        }
      }
    }

    if (printValues) {
      os << "=== values\n";
      analysis.getStoredValueQuery().setMaxQueryVisits(maxQueryVisits);
      module.walk([&](cir::LoadOp load) {
        auto name = load->getAttrOfType<StringAttr>("test.load");
        if (!name)
          return;
        unsigned numContexts = 0;
        StoredValues values = analysis.valuesAtInContexts(
            load.getAddr(), std::nullopt, load.getType(), &numContexts);
        os << name.getValue() << ": {";
        llvm::interleaveComma(values.consts, os,
                              [&](TypedAttr c) { os << c; });
        os << "} defs=" << values.defs.size()
           << (values.complete ? " complete" : " incomplete");
        if (!values.complete)
          os << " (" << (values.why ? formatReason(values.why) : "-") << ")";
        if (numContexts)
          os << " contexts=" << numContexts;
        if (explainSets && !values.complete)
          for (const std::string &link : explainChain(values.why))
            os << "\n    <- " << link;
        os << "\n";
      });
      if (printStats) {
        // Every load of an analysed function, by answer.
        size_t numLoads = 0, numConstant = 0, numComplete = 0, numCtx = 0;
        llvm::StringMap<unsigned> whyIncomplete;
        llvm::TimeRecord start = llvm::TimeRecord::getCurrentTime(true);
        for (const auto &fn : analysis.getFunctions())
          fn->func->walk([&](cir::LoadOp load) {
            ++numLoads;
            unsigned contexts = 0;
            StoredValues values = analysis.valuesAtInContexts(
                load.getAddr(), std::nullopt, load.getType(), &contexts);
            numCtx += contexts > 0;
            if (!values.complete) {
              ++whyIncomplete[values.why ? formatReason(values.why) : "-"];
              return;
            }
            ++numComplete;
            if (values.defs.empty() && values.consts.size() == 1)
              ++numConstant;
          });
        llvm::TimeRecord end = llvm::TimeRecord::getCurrentTime(false);
        end -= start;
        const StoredValueQuery::Stats &qs =
            analysis.getStoredValueQuery().getStats();
        os << "--- value stats\n"
           << "loads: " << numLoads << "\n"
           << "loads with a complete answer: " << numComplete << "\n"
           << "loads proven constant: " << numConstant << "\n"
           << "loads answered from calling contexts: " << numCtx << "\n"
           << "indexed writes: "
           << analysis.getStoredValueQuery().getIndex().getWrites().size()
           << " (entries "
           << analysis.getStoredValueQuery().getIndex().getNumEntries()
           << ")\n"
           << "queries: " << qs.numQueries << " visits: " << qs.numVisits
           << " budget hits: " << qs.numBudgetHits << "\n"
           << "value query time (wall, s): " << end.getWallTime() << "\n";
        SmallVector<std::pair<unsigned, StringRef>> sorted;
        for (const auto &entry : whyIncomplete)
          sorted.emplace_back(entry.getValue(), entry.getKey());
        llvm::sort(sorted, [](const auto &a, const auto &b) {
          return a.first != b.first ? a.first > b.first : a.second < b.second;
        });
        os << "ops written with an unknown value:";
        for (const auto &entry :
             analysis.getStoredValueQuery().getIndex().getUnknownWriteOps())
          os << " " << entry.getKey() << "=" << entry.getValue();
        os << "\n";
        os << "incomplete loads by reason: " << sorted.size() << "\n";
        for (auto [count, name] : llvm::ArrayRef(sorted).take_front(topExplanations))
          os << "  " << name << " " << count << "\n";
      }
    }

    if (syclModels)
      printLaunches(os, module, analysis);

    if (!printStats)
      continue;
    const PointsToAnalysisStats &stats = analysis.getStats();
    llvm::StringMap<unsigned> collapseReasons, resolutions;
    size_t numNodes = 0, numCollapsed = 0, numExternal = 0, numIncomplete = 0;
    size_t numSubobjectLinks = 0, numChildren = 0, numCollapsedChildren = 0;
    for (const Node *node : graph.nodes()) {
      ++numNodes;
      numSubobjectLinks += node->getNumSubobjects();
      if (!graph.getParents(node).empty()) {
        ++numChildren;
        numCollapsedChildren += node->isCollapsed();
      }
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
    os << "--- " << getPointsToModeName(options.mode) << " stats\n"
       << "functions: " << stats.numFunctions << "\n"
       << "functions with complete formals: " << stats.numCompleteFunctions
       << "\n"
       << "functions whose address escapes: " << stats.numEscapedFunctions
       << "\n"
       << "local nodes: " << stats.numLocalNodes << "\n"
       << "TU live nodes: " << numNodes << "\n"
       << "collapsed nodes: " << numCollapsed << "\n"
       << "external nodes: " << numExternal << "\n"
       << "incomplete nodes: " << numIncomplete << "\n";
    if (options.subobjects)
      os << "subobject links: " << numSubobjectLinks << "\n"
         << "subobject nodes: " << numChildren << " (collapsed "
         << numCollapsedChildren << ")\n";
    os << "direct calls: " << stats.numDirectCalls << "\n"
       << "bindings unified: " << stats.numBindings << "\n"
       << "indirect calls resolved: " << stats.numIndirectResolved << "\n"
       << "indirect calls unresolved: " << stats.numIndirectUnresolved << "\n"
       << "indirect targets bound: " << stats.numIndirectTargetsBound << "\n"
       << "indirect rounds that bound a target: " << stats.numIndirectRounds
       << "\n"
       << "indirect calls over the round budget: " << stats.numIndirectBudget
       << "\n"
       << "clobbering calls: " << stats.numClobberingCalls << "\n"
       << "calls a model replaced: " << stats.numReplacedCalls << "\n"
       << "functions skipped (only replaced calls): "
       << stats.numSkippedFunctions << "\n"
       << "kernel launches recorded: " << analysis.getLaunches().size()
       << "\n"
       << "copy closure settled: " << (stats.transfersSettled ? "yes" : "no")
       << "\n"
       << "local phase time (wall, s): " << stats.localSeconds << "\n"
       << "analysis time (wall, s): " << stats.totalSeconds << "\n";
    if (bottomUp) {
      std::string largest = "-";
      if (auto f = dyn_cast_if_present<cir::FuncOp>(stats.largestGraph))
        largest = ("@" + f.getSymName()).str();
      os << "copies of callee graphs: " << stats.numClones << "\n"
         << "copied nodes: " << stats.numClonedNodes << "\n"
         << "largest graph (nodes): " << stats.maxGraphNodes << " " << largest
         << "\n"
         << "calling contexts: " << stats.numCallPaths << "\n"
         << "inherited call sites: " << stats.numInheritedCalls << "\n"
         << "bindings inside an SCC: " << stats.numSCCBindings << "\n"
         << "bindings through a cycle: " << stats.numCycleBindings << "\n"
         << "bindings over budget: " << stats.numBudgetBindings << "\n"
         << "calls to a callee still being analysed: "
         << stats.numCycleFallbacks << "\n"
         << "flags projected to sources: " << stats.numProjectedFlags << "\n"
         << "source merges projected: " << stats.numPullBackMerges << "\n"
         << "launch records (all contexts): " << stats.numLaunchInstances
         << "\n";
    }
    auto printMap = [&](StringRef title, const llvm::StringMap<unsigned> &map,
                        size_t limit = ~size_t(0)) {
      SmallVector<std::pair<unsigned, StringRef>> sorted;
      for (const auto &entry : map)
        sorted.emplace_back(entry.getValue(), entry.getKey());
      llvm::sort(sorted, [](const auto &a, const auto &b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
      });
      os << title << ": " << sorted.size() << "\n";
      for (auto [count, name] : llvm::ArrayRef(sorted).take_front(limit))
        os << "  " << name << " " << count << "\n";
    };
    // The points-to set of every pointer value of the analysed functions.
    size_t numComplete = 0, numIncompleteValues = 0;
    llvm::StringMap<unsigned> explanations, rootReasons, summaryReasons;
    auto countValue = [&](Value value) {
      if (!isPointerValueType(value.getType()))
        return;
      ObjectSet set = analysis.pointsTo(value);
      if (set.complete) {
        ++numComplete;
        return;
      }
      ++numIncompleteValues;
      ++explanations[explain(set)];
      const Reason *root = getRootReason(set.why);
      std::string key = root ? formatReason(root) : std::string("-");
      if (root != set.why)
        key = "reachable from " + key;
      ++rootReasons[key];
    };
    for (const auto &fn : analysis.getFunctions())
      fn->func->walk([&](Operation *op) {
        for (Region &region : op->getRegions())
          for (Block &block : region)
            for (BlockArgument arg : block.getArguments())
              countValue(arg);
        for (Value result : op->getResults())
          countValue(result);
      });
    for (const Node *node : graph.nodes())
      if (const Reason *why = node->getReason(NodeFlags::Summary))
        ++summaryReasons[why->detail];
    os << "recursive functions: " << stats.numRecursiveFunctions << "\n"
       << "summary nodes: " << stats.numSummaryNodes << "\n"
       << "pointer values with a complete set: " << numComplete << "\n"
       << "pointer values with an incomplete set: " << numIncompleteValues
       << "\n";
    printMap("summary nodes by reason", summaryReasons);
    printMap("incomplete pointer values by root reason", rootReasons,
             topExplanations);
    printMap("incomplete pointer values by explanation", explanations,
             topExplanations);
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

void TestCIRPointsToPass::printLaunches(raw_ostream &os, ModuleOp module,
                                        const PointsToAnalysis &analysis) {
  if (analysis.getLaunches().empty())
    return;
  os << "=== launches\n";
  cir::SYCLHostFacts facts(analysis, getOperation());
  auto printNode = [&](const Node *node) {
    os << "n" << node->getId() << "+0 [" << toString(node->getFlags()) << "]";
  };
  // The kernels in launch order, for the per-kernel verdicts.
  llvm::SetVector<StringAttr> kernels;
  for (const LaunchRecord &launch : analysis.getLaunches()) {
    auto func = launch.call->getParentOfType<cir::FuncOp>();
    os << "launch @" << (func ? func.getSymName() : StringRef("?")) << " "
       << launch.model;
    // BU: a copy in a caller's graph.
    if (launch.ctx) {
      Operation *outer = analysis.getCallPaths().getPath(launch.ctx).front();
      auto graph = outer->getParentOfType<cir::FuncOp>();
      os << " in-graph-of=@" << (graph ? graph.getSymName() : StringRef("?"))
         << " ctx=#" << launch.ctx;
    }
    os << " kernels={";
    llvm::interleaveComma(launch.kernelNames, os,
                          [&](StringAttr name) { os << name.getValue(); });
    os << "} closure=";
    Graph::printCell(os, launch.closure ? *launch.closure : Cell());
    os << " geometry=";
    Graph::printCell(os, launch.geometry ? *launch.geometry : Cell());
    os << "\n";
    if (printValues) {
      cir::SYCLHostFacts::GeometryFact g = facts.geometryOfLaunch(launch);
      os << "  geometry " << cir::formatSyclGeometry(g.geometry);
      if (!g.note.empty())
        os << " (" << g.note << ")";
      os << "\n";
    }
    for (StringAttr name : launch.kernelNames) {
      kernels.insert(name);
      os << "  kernel " << name.getValue();
      ArrayRef<cir::FuncOp> device = facts.getDeviceKernels(name.getValue());
      if (device.empty()) {
        os << " (no device function)\n";
        continue;
      }
      os << "\n";
      cir::SYCLNoAliasCandidates candidates =
          cir::getSYCLNoAliasCandidates(device);
      cir::SYCLHostFacts::LaunchFacts here =
          facts.factsOfLaunch(launch, candidates);
      for (const auto &[field, id] : here.fields) {
        os << "    field " << field << " ";
        if (id.kind == cir::SYCLHostFacts::MemIdentity::Kind::Storage) {
          os << "accessor storage=";
          if (id.node) {
            printNode(id.node);
            os << " origins=" << id.node->getOrigins().size() << " host=";
            if (id.host)
              printNode(id.host);
            else
              os << "none";
          } else {
            os << "null";
          }
        } else {
          os << "pointer target=";
          if (id.node)
            printNode(id.node);
          else
            os << "null";
        }
        os << (id.complete ? " complete" : " unknown (" + id.what + ")")
           << "\n";
      }
      llvm::StringMap<mlir::Type> consts =
          facts.getConstCandidates(name.getValue());
      SmallVector<StringRef> constNames;
      for (const auto &c : consts)
        constNames.push_back(c.first());
      llvm::sort(constNames);
      for (StringRef field : printValues ? constNames : SmallVector<StringRef>()) {
        cir::SYCLHostFacts::ConstFact c =
            facts.constantOfLaunch(launch, field, consts.lookup(field));
        os << "    const " << field << " ";
        if (c.value)
          c.value.print(os);
        else
          os << "none (" << c.note << ")";
        os << "\n";
      }
      if (!candidates.poison.empty())
        os << "    blocked: " << candidates.poison << "\n";
      for (size_t a = 0; a < here.fields.size(); ++a)
        for (size_t b = a + 1; b < here.fields.size(); ++b)
          if (cir::SYCLHostFacts::sameMemory(here.fields[a].second,
                                             here.fields[b].second))
            os << "    same " << here.fields[a].first << " "
               << here.fields[b].first << "\n";
      os << "    no-alias";
      for (const auto &field : here.fields)
        if (here.decisions.lookup(field.first).ok)
          os << " " << field.first;
      os << "\n";
    }
  }
  for (StringAttr name : kernels) {
    if (facts.getDeviceKernels(name.getValue()).empty())
      continue;
    llvm::StringMap<cir::NoAliasDecision> decisions =
        facts.noAliasFields(name.getValue());
    SmallVector<StringRef> names;
    for (const auto &d : decisions)
      names.push_back(d.first());
    llvm::sort(names);
    os << "kernel " << name.getValue() << ": no-alias";
    for (StringRef f : names)
      if (decisions.lookup(f).ok)
        os << " " << f;
    os << "\n";
    if (explainSets)
      for (StringRef f : names)
        os << "  note " << f << ": " << decisions.lookup(f).note << "\n";
    if (!printValues)
      continue;
    llvm::StringMap<cir::SYCLHostFacts::ConstFact> consts =
        facts.constantFields(name.getValue());
    SmallVector<StringRef> constNames;
    for (const auto &c : consts)
      constNames.push_back(c.first());
    llvm::sort(constNames);
    for (StringRef f : constNames) {
      const cir::SYCLHostFacts::ConstFact &c = consts.find(f)->second;
      os << "kernel " << name.getValue() << ": const " << f << " ";
      if (c.value)
        c.value.print(os);
      else
        os << "none (" << c.note << ")";
      os << "\n";
    }
    cir::SYCLHostFacts::GeometryFact g = facts.geometry(name.getValue());
    os << "kernel " << name.getValue() << ": geometry "
       << cir::formatSyclGeometry(g.geometry);
    if (!g.note.empty())
      os << " (" << g.note << ")";
    os << "\n";
  }
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
  PointsToModelRegistry syclRegistry;
  const PointsToModelRegistry *models = &registry;
  if (syclModels) {
    cir::registerSYCLPointsToModels(syclRegistry);
    syclRegistry.setFallback(&registry);
    models = &syclRegistry;
  }
  llvm::TimeRecord buildTime;
  bool anyFailed = false;

  getOperation()->walk([&](cir::FuncOp func) {
    if (func.isDeclaration())
      return;
    llvm::TimeRecord start = llvm::TimeRecord::getCurrentTime(true);
    LocalBuilder builder(func, &symbols, /*reasons=*/nullptr, models);
    builder.setSubobjects(subobjects);
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
