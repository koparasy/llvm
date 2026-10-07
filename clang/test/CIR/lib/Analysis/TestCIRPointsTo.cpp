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
// sycl=1 adds the SYCL host API models (OffloadOpt's
// registerSYCLPointsToModels, through PointsToAnalysisOptions::extraModels)
// and, in ci mode, prints the kernel launches they record, even with
// print-graphs=0:
//
//   === launches
//   launch @<func> <model> kernels={<name>, ...} closure=<cell>
//       geometry=<cell>
//     kernel <name>[ (no device function)]
//       field <path> accessor storage=<cell> [<flags>] origins=<n>
//           host=<cell or none> <complete | unknown (<why>)>
//       field <path> pointer target=<cell> [<flags>] <complete | ...>
//       field <path> <not in closure | local accessor>
//       blocked: <why>
//       same <a> <b>            for each pair with the same identity
//       no-alias <field> ...    the fields distinct from every other one
//   kernel <name>: no-alias <field> ...   over every launch of the kernel
//
// The identity of an accessor field is the `$storage` ghost node of the
// accessor in the closure and the `$host` memory of that storage; of a
// pointer field, the object it points to. Two identities are the same if
// their nodes are; distinct if both are complete (no External, Incomplete or
// UnknownObj, and a storage made by a model) and their nodes and host memory
// differ; unknown otherwise. A captured struct that holds a pointer, or an
// accessor inside a captured struct, blocks the kernel (as
// KernelBindingTable does).
//
// closed-tu=1 sets PointsToAnalysisOptions::assumeClosedTU and
// max-indirect-rounds=<n> PointsToAnalysisOptions::maxIndirectRounds.
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
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLKernelParams.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLPointsToModels.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/MapVector.h"
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
  if (syclModels)
    options.extraModels = cir::registerSYCLPointsToModels;
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

    if (syclModels)
      printLaunches(os, module, analysis);

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
       << "functions whose address escapes: " << stats.numEscapedFunctions
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
       << "calls a model replaced: " << stats.numReplacedCalls << "\n"
       << "functions skipped (only replaced calls): "
       << stats.numSkippedFunctions << "\n"
       << "kernel launches recorded: " << analysis.getLaunches().size()
       << "\n"
       << "copy closure settled: " << (stats.transfersSettled ? "yes" : "no")
       << "\n"
       << "local phase time (wall, s): " << stats.localSeconds << "\n"
       << "analysis time (wall, s): " << stats.totalSeconds << "\n";
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

namespace {
/// The memory identity of one closure field (see the file comment).
struct FieldIdentity {
  std::string path;
  enum Kind { Storage, Pointer, Skipped } kind = Skipped;
  const Node *node = nullptr; // the storage, or the pointer's target
  const Node *host = nullptr; // the storage's host memory
  bool complete = false;
  std::string why;
};

enum class Verdict { Same, Distinct, Unknown };
} // namespace

static bool isCompleteNode(const Node *node) {
  return node && !hasAny(node->getFlags(), NodeFlags::External |
                                               NodeFlags::Incomplete |
                                               NodeFlags::UnknownObj);
}

/// One reason a node is not complete, for the field lines.
static std::string incompleteWhy(const Node *node) {
  if (!node)
    return "no node";
  for (NodeFlags flag :
       {NodeFlags::External, NodeFlags::Incomplete, NodeFlags::UnknownObj})
    if (node->hasFlags(flag)) {
      std::string out = getFlagName(flag).str();
      if (const Reason *why = node->getReason(flag))
        if (const Reason *root = getRootReason(why))
          out += ": " + formatReason(root) + " at " + formatReasonSite(root);
      return out;
    }
  return "complete";
}

static Verdict compareIdentities(const FieldIdentity &a,
                                 const FieldIdentity &b) {
  if (a.kind == FieldIdentity::Skipped || b.kind == FieldIdentity::Skipped)
    return Verdict::Distinct; // Not a candidate: never blocks a pair.
  if (!a.node || !b.node)
    return Verdict::Unknown;
  if (a.node == b.node)
    return Verdict::Same;
  if (!a.complete || !b.complete)
    return Verdict::Unknown;
  // The memory the device may access through each field: an accessor's host
  // memory, else its fresh storage; a pointer's target.
  const Node *ma = a.kind == FieldIdentity::Storage && a.host ? a.host : a.node;
  const Node *mb = b.kind == FieldIdentity::Storage && b.host ? b.host : b.node;
  return ma == mb ? Verdict::Same : Verdict::Distinct;
}

void TestCIRPointsToPass::printLaunches(raw_ostream &os, ModuleOp module,
                                        const PointsToAnalysis &analysis) {
  if (analysis.getLaunches().empty())
    return;
  os << "=== launches\n";
  const Graph &graph = analysis.getGraph();
  mlir::DataLayout layout(module);
  // The no-alias fields of each kernel, ANDed over its launches.
  llvm::MapVector<StringAttr, std::optional<llvm::SmallVector<std::string>>>
      perKernel;
  for (const LaunchRecord &launch : analysis.getLaunches()) {
    auto func = launch.call->getParentOfType<cir::FuncOp>();
    os << "launch @" << (func ? func.getSymName() : StringRef("?")) << " "
       << launch.model << " kernels={";
    llvm::interleaveComma(launch.kernelNames, os,
                          [&](StringAttr name) { os << name.getValue(); });
    os << "} closure=";
    Graph::printCell(os, launch.closure ? *launch.closure : Cell());
    os << " geometry=";
    Graph::printCell(os, launch.geometry ? *launch.geometry : Cell());
    os << "\n";
    for (StringAttr name : launch.kernelNames) {
      cir::FuncOp kernel;
      getOperation()->walk([&](cir::FuncOp f) {
        if (f.getSymName() == name.getValue() && !f.isDeclaration() &&
            f.getNumArguments() && cir::getSYCLKernelParam(f, 0))
          kernel = f;
      });
      os << "  kernel " << name.getValue();
      if (!kernel) {
        os << " (no device function)\n";
        perKernel[name] = llvm::SmallVector<std::string>();
        continue;
      }
      os << "\n";
      SmallVector<FieldIdentity> fields;
      std::string blocked;
      for (unsigned i = 0, e = kernel.getNumArguments(); i != e; ++i) {
        std::optional<cir::SYCLKernelParam> param =
            cir::getSYCLKernelParam(kernel, i);
        if (!param)
          continue;
        Type argType = kernel.getArgumentTypes()[i];
        std::string path;
        for (Attribute elem : param->path)
          path += (path.empty() ? "" : ".") +
                  cast<StringAttr>(elem).getValue().str();
        if (param->kind == "std_layout") {
          Type held = argType;
          if (auto byval = kernel.getArgAttrOfType<TypeAttr>(i, "llvm.byval"))
            held = byval.getValue();
          if (!getPointerSlotOffsets(held, layout).empty() && blocked.empty())
            blocked = "captured struct '" + path + "' holds a pointer";
          continue;
        }
        if (param->kind == "local_accessor") {
          if (param->accessorPart == "ptr")
            os << "    field " << path << " local accessor\n";
          continue;
        }
        bool isAccessor = param->kind == "accessor";
        if (!isAccessor && param->kind != "pointer")
          continue;
        if (isAccessor && param->accessorPart != "ptr")
          continue;
        if (isAccessor && !param->isTopLevel() && blocked.empty())
          blocked = "captured '" + param->field.str() +
                    "' holds a nested accessor";
        FieldIdentity field;
        field.path = path;
        auto ptrTy = dyn_cast<cir::PointerType>(argType);
        auto as = ptrTy ? dyn_cast_if_present<cir::TargetAddressSpaceAttr>(
                              ptrTy.getAddrSpace())
                        : nullptr;
        os << "    field " << path << " ";
        if (as && as.getValue() == 3) {
          os << "local accessor\n";
          continue;
        }
        std::optional<int64_t> offset;
        if (launch.closure && launch.closureType)
          offset = cir::getSYCLClosureFieldOffset(module, launch.closureType,
                                                  param->path, layout);
        if (!offset) {
          os << "not in closure\n";
          field.kind = FieldIdentity::Pointer;
          field.why = "not in closure";
          fields.push_back(field);
          continue;
        }
        Cell at(launch.closure->getNode(), launch.closure->getOffset() + *offset);
        if (isAccessor) {
          field.kind = FieldIdentity::Storage;
          std::optional<Cell> storage =
              graph.lookupGhostLink(at, cir::getSYCLStorageField());
          os << "accessor storage=";
          Graph::printCell(os, storage ? *storage : Cell());
          if (storage) {
            field.node = storage->getNode();
            os << " [" << toString(field.node->getFlags()) << "] origins="
               << field.node->getOrigins().size();
            std::optional<Cell> host =
                graph.lookupGhostLink(*storage, cir::getSYCLHostField());
            os << " host=";
            if (host) {
              field.host = host->getNode();
              Graph::printCell(os, *host);
              os << " [" << toString(field.host->getFlags()) << "]";
            } else {
              os << "none";
            }
            field.complete = isCompleteNode(field.node) &&
                             !field.node->getOrigins().empty() &&
                             (!field.host || isCompleteNode(field.host));
            if (!field.complete)
              field.why = field.node->getOrigins().empty()
                              ? std::string("storage made by no model")
                              : incompleteWhy(isCompleteNode(field.node)
                                                  ? field.host
                                                  : field.node);
          } else {
            field.why = "no storage";
          }
        } else {
          field.kind = FieldIdentity::Pointer;
          std::optional<Cell> target = graph.lookupLink(at);
          os << "pointer target=";
          Graph::printCell(os, target ? *target : Cell());
          if (target) {
            field.node = target->getNode();
            os << " [" << toString(field.node->getFlags()) << "]";
            field.complete = isCompleteNode(field.node);
            if (!field.complete)
              field.why = incompleteWhy(field.node);
          } else {
            field.why = "nothing stored";
          }
        }
        os << (field.complete ? " complete" : " unknown (" + field.why + ")")
           << "\n";
        fields.push_back(field);
      }
      if (!blocked.empty())
        os << "    blocked: " << blocked << "\n";
      SmallVector<std::string> noAlias;
      for (size_t a = 0; a < fields.size(); ++a) {
        bool distinct = blocked.empty() && fields[a].complete;
        for (size_t b = 0; b < fields.size(); ++b) {
          if (a == b)
            continue;
          Verdict v = compareIdentities(fields[a], fields[b]);
          if (v == Verdict::Same && a < b)
            os << "    same " << fields[a].path << " " << fields[b].path
               << "\n";
          distinct &= v == Verdict::Distinct;
        }
        if (distinct)
          noAlias.push_back(fields[a].path);
      }
      os << "    no-alias";
      for (const std::string &f : noAlias)
        os << " " << f;
      os << "\n";
      auto &all = perKernel[name];
      if (!all) {
        all = noAlias;
      } else {
        llvm::SmallVector<std::string> kept;
        for (const std::string &f : *all)
          if (llvm::is_contained(noAlias, f))
            kept.push_back(f);
        all = kept;
      }
    }
  }
  for (auto &[name, fields] : perKernel) {
    os << "kernel " << name.getValue() << ": no-alias";
    for (const std::string &f : *fields)
      os << " " << f;
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
