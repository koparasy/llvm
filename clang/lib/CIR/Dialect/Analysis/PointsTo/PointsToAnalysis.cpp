//===- PointsToAnalysis.cpp - Interprocedural CIR points-to ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToAnalysis.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/GlobalNodes.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToModels.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToVerifier.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Interfaces/CIRLoopOpInterface.h"
#include "clang/CIR/Interfaces/CIROpInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "cir-points-to"

using namespace mlir;
using namespace cir;
using namespace cir::pointsto;

StringRef cir::pointsto::getCallResolutionName(CallResolution resolution) {
  switch (resolution) {
  case CallResolution::Direct:
    return "direct";
  case CallResolution::Model:
    return "model";
  case CallResolution::Unknown:
    return "unknown";
  case CallResolution::Indirect:
    return "indirect";
  case CallResolution::IndirectUnresolved:
    return "indirect-unresolved";
  }
  llvm_unreachable("unknown call resolution");
}

PointsToAnalysis::PointsToAnalysis(ModuleOp module,
                                   const PointsToModelRegistry *models,
                                   PointsToAnalysisOptions options)
    : module(module), models(models), options(options) {
  run();
}

PointsToAnalysis::~PointsToAnalysis() = default;

Cell PointsToAnalysis::cellOf(Value value) const {
  return scalarMap.lookup(value);
}

const AggregateSlots *PointsToAnalysis::slotsOf(Value value) const {
  auto it = aggregateMap.find(value);
  return it == aggregateMap.end() ? nullptr : &it->second;
}

/// The set of the single cell \p cell, complete unless its node is
/// Incomplete, External or UnknownObj.
static ObjectSet makeObjectSet(Cell cell) {
  ObjectSet set;
  Cell resolved = cell.resolve();
  set.cells.push_back(resolved);
  const Node *node = resolved.getNode();
  if (hasAny(node->getFlags(), NodeFlags::Incomplete | NodeFlags::External |
                                   NodeFlags::UnknownObj)) {
    set.complete = false;
    set.why = node->getReason(NodeFlags::Incomplete);
    if (!set.why)
      set.why = node->getReason(NodeFlags::External);
  }
  return set;
}

ObjectSet PointsToAnalysis::pointsTo(Value value) const {
  if (Cell cell = scalarMap.lookup(value))
    return makeObjectSet(cell);
  ObjectSet set;
  if (isPointerValueType(value.getType()) && isNullPointerConstant(value))
    return set;
  set.complete = false;
  set.why = notAnalysedWhy;
  return set;
}

ObjectSet PointsToAnalysis::objectsAt(Cell cell) const {
  ObjectSet set;
  if (!cell) {
    set.complete = false;
    set.why = notAnalysedWhy;
    return set;
  }
  if (std::optional<Cell> target = graph->lookupLink(cell))
    return makeObjectSet(*target);
  // Nothing seen stored there: that is all unless code we don't see may
  // have stored something (by propagation, a link target of an External or
  // Incomplete node carries the flag too).
  ObjectSet holder = makeObjectSet(cell);
  set.complete = holder.complete;
  set.why = holder.why;
  return set;
}

std::string cir::pointsto::explain(const ObjectSet &set) {
  if (!set.complete)
    return "pointer may refer to unknown memory: " + explain(set.why);
  if (set.cells.empty())
    return "pointer refers to no object";
  std::string out = "pointer refers to known memory";
  for (Cell cell : set.cells)
    if (const Reason *why = cell.getNode()->getReason(NodeFlags::Summary))
      return out + "; " + explain(why);
  return out;
}

SmallVector<std::pair<FlatSymbolRefAttr, Node *>, 0>
PointsToAnalysis::getSymbolNodes() const {
  SmallVector<std::pair<FlatSymbolRefAttr, Node *>, 0> result =
      globals->getSymbolNodes();
  for (auto &[sym, node] : result)
    node = Cell(node).getNode();
  return result;
}

bool PointsToAnalysis::isRecursive(cir::FuncOp func) const {
  return recursive.contains(func.getOperation());
}

const TUFunction *PointsToAnalysis::getFunction(cir::FuncOp func) const {
  return functionMap.lookup(func.getOperation());
}

Node *PointsToAnalysis::getSymbolNode(FlatSymbolRefAttr sym) const {
  for (auto [s, node] : globals->getSymbolNodes())
    if (s == sym)
      return Cell(node).getNode();
  return nullptr;
}

/// Whether \p func is analysed: it has a body and sits directly in \p module.
static bool isAnalysed(cir::FuncOp func, ModuleOp module) {
  return func && !func.isDeclaration() &&
         func->getParentOp() == module.getOperation();
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

void PointsToAnalysis::run() {
  llvm::TimeRecord start = llvm::TimeRecord::getCurrentTime(true);
  dataLayout = std::make_unique<mlir::DataLayout>(module);
  auto voidPtrTy =
      cir::PointerType::get(cir::VoidType::get(module.getContext()));
  unsigned pointerSize = dataLayout->getTypeSize(voidPtrTy).getFixedValue();
  graph = std::make_unique<Graph>(pointerSize, &reasons);
  notAnalysedWhy =
      reasons.make(Reason::Kind::UnhandledOp, nullptr, "value not analysed");
  globals = std::make_unique<GlobalNodes>(*graph, *dataLayout, symbols, module);

  computeVisibleCallers();

  // Call-graph SCCs in post-order: callees before their callers. The
  // context-insensitive result does not depend on the order; the
  // context-sensitive phase that replaces this one will.
  SmallVector<cir::FuncOp, 0> order;
  llvm::DenseSet<Operation *> seen;
  {
    CallGraph callGraph(module);
    for (auto it = llvm::scc_begin(static_cast<const CallGraph *>(&callGraph));
         !it.isAtEnd(); ++it)
      for (CallGraphNode *node : *it) {
        if (node->isExternal())
          continue;
        auto func =
            dyn_cast<cir::FuncOp>(node->getCallableRegion()->getParentOp());
        if (isAnalysed(func, module) && seen.insert(func).second)
          order.push_back(func);
      }
  }
  for (auto func : module.getOps<cir::FuncOp>())
    if (isAnalysed(func, module) && seen.insert(func).second)
      order.push_back(func);

  llvm::TimeRecord localStart = llvm::TimeRecord::getCurrentTime(true);
  for (cir::FuncOp func : order) {
    bool complete =
        options.assumeClosedTU || visibleCallers.contains(func.getOperation());
    buildLocal(func, complete);
  }
  // A global another TU can name may hand the functions in its initializer
  // to code we don't see even if no function here names it: give it a node,
  // so that markEscapedFunctions sees those functions.
  for (auto global : module.getOps<cir::GlobalOp>()) {
    std::optional<Attribute> init = global.getInitialValue();
    if (!init || cir::isLocalLinkage(global.getLinkage()))
      continue;
    bool namesSymbol = false;
    init->walk([&](FlatSymbolRefAttr) { namesSymbol = true; });
    if (namesSymbol)
      globals->getNode(FlatSymbolRefAttr::get(global.getSymNameAttr()));
  }
  llvm::TimeRecord localTime = llvm::TimeRecord::getCurrentTime(false);
  localTime -= localStart;
  stats.localSeconds = localTime.getWallTime();
  stats.numFunctions = functions.size();

  // Context-insensitive binding: every caller shares the callee's formals
  // and return cell. Unification only merges, so one pass is enough.
  stats.numBindings = bindCalls(/*dryRun=*/false);

  // The bindings added links on both sides of the copies: relate the new
  // offsets.
  stats.transfersSettled = closeMemTransfers(*graph, memTransfers);

  // Status propagation, global clobbering and indirect-call resolution, to
  // a fixpoint of all of them (see the file comment):
  //  - an unknown-call rule adds External flags and clobbers globals;
  //  - propagation can show more globals to be escaped, which the clobbering
  //    calls then reach too;
  //  - binding an indirect target merges nodes, which can give another
  //    callee node more targets, and adds links the copy closure must relate
  //    (N8);
  //  - any of them can make a callee node External or Incomplete, which turns
  //    a call resolved earlier into an unresolved one;
  //  - a function whose node turned External, or that is a target of an
  //    unresolved call, gets Incomplete formals (markEscapedFunctions, N13),
  //    which propagation then spreads.
  // The loop stops only after a round in which no rule changed anything, so
  // the result satisfies every rule. It terminates: every rule only adds
  // flags or merges nodes, both monotone, and the graph has finitely many
  // nodes and flags; the nodes a round may create (the unknown results of an
  // unknown-call rule, the model objects of a declaration target) are made
  // at most once per (call, target) pair, and a call becomes unresolved at
  // most once. Binding is bounded too: each round that binds adds at least
  // one (call, target) pair, of finitely many, and maxIndirectRounds caps
  // the rounds that may do so.
  bool changed = true;
  while (changed) {
    propagateStatus();
    changed = clobberGlobals();
    propagateStatus();
    bool bound = false;
    changed |= resolveIndirectCalls(/*finalize=*/false, bound);
    changed |= markEscapedFunctions();
    if (!changed)
      changed = resolveIndirectCalls(/*finalize=*/true, bound);
    if (bound)
      stats.transfersSettled &= closeMemTransfers(*graph, memTransfers);
  }
  for (std::unique_ptr<TUFunction> &fn : functions)
    for (TUCallSite &call : fn->callSites) {
      if (call.resolution == CallResolution::Indirect)
        ++stats.numIndirectResolved;
      else if (call.resolution == CallResolution::IndirectUnresolved)
        ++stats.numIndirectUnresolved;
    }

  // The graph is final: no rule merges nodes any more.
  computeRecursion();
  markSummaries();

#ifndef NDEBUG
  // Unification is monotone: once every call is bound, binding them again
  // must merge nothing.
  size_t unbound = bindCalls(/*dryRun=*/true);
  assert(unbound == 0 && "a second pass over the calls merged more nodes");
  (void)unbound;
#endif

  if (isPointsToVerifyEnabled()) {
    std::string errors;
    llvm::raw_string_ostream os(errors);
    if (!graph->verify(&os)) {
      module.emitError() << "points-to TU graph failed verification:\n"
                         << errors;
      verifyErrors = true;
    }
    if (failed(verifyPointsToAnalysis(*this)))
      verifyErrors = true;
  }

  llvm::TimeRecord total = llvm::TimeRecord::getCurrentTime(false);
  total -= start;
  stats.totalSeconds = total.getWallTime();
}

void PointsToAnalysis::computeVisibleCallers() {
  // SOUNDNESS: a function may have callers we don't see unless it has local
  // linkage (no other TU names it) and every use of its symbol is a direct
  // call from a function walked here (no address is taken, no alias names
  // it, no global initializer or ctor region calls it).
  llvm::DenseSet<StringAttr> otherUses;
  // The region, not the module op: on a symbol table op getSymbolUses only
  // looks at the op's own attributes.
  std::optional<SymbolTable::UseRange> uses =
      SymbolTable::getSymbolUses(&module.getBodyRegion());
  if (!uses)
    return;
  for (const SymbolTable::SymbolUse &use : *uses) {
    StringAttr name = use.getSymbolRef().getRootReference();
    Operation *user = use.getUser();
    bool direct = false;
    if (auto call = dyn_cast<CIRCallOpInterface>(user)) {
      auto callee = call->getAttrOfType<FlatSymbolRefAttr>("callee");
      direct = !call.isIndirect() && callee && callee.getAttr() == name;
    } else if (isa<cir::DeleteArrayOp>(user)) {
      // The element destructor and operator delete: synthetic call sites.
      direct = true;
    }
    if (direct)
      direct = isAnalysed(user->getParentOfType<cir::FuncOp>(), module);
    if (!direct)
      otherUses.insert(name);
  }
  for (auto func : module.getOps<cir::FuncOp>()) {
    if (!isAnalysed(func, module) || !cir::isLocalLinkage(func.getLinkage()))
      continue;
    // Run at program start or exit, by code we don't see.
    if (func->hasAttr("global_ctor_priority") ||
        func->hasAttr("global_dtor_priority"))
      continue;
    if (!otherUses.contains(func.getSymNameAttr()))
      visibleCallers.insert(func.getOperation());
  }
}

void PointsToAnalysis::buildLocal(cir::FuncOp func, bool formalsComplete) {
  LocalBuilder builder(func, &symbols, &reasons, models);
  builder.setFormalsComplete(formalsComplete);
  builder.setDeferIndirectCalls(true);
  std::unique_ptr<FunctionGraph> fg = builder.build();
  if (builder.hadVerifyErrors())
    verifyErrors = true;
  for (const auto &entry : builder.getUnhandledOps())
    unhandledOps[entry.getKey()] += entry.getValue();

  // Clone every node: the scalar map, the call sites and the copies may name
  // any of them.
  const Graph &local = fg->getGraph();
  stats.numLocalNodes += local.getNumLiveNodes();
  SmallVector<Cell, 0> roots;
  for (Node *node : local.nodes())
    roots.push_back(Cell(node));
  CloneMap clones = local.cloneInto(*graph, roots);
  auto map = [&](Cell cell) { return cell ? clones.map(cell) : Cell(); };
  auto mapOpt = [&](std::optional<Cell> cell) -> std::optional<Cell> {
    if (!cell)
      return std::nullopt;
    return map(*cell);
  };
  auto mapSlots = [&](const AggregateSlots &slots) {
    AggregateSlots out;
    for (auto [off, cell] : slots)
      out.emplace_back(off, map(cell));
    return out;
  };

  auto fn = std::make_unique<TUFunction>();
  fn->func = func;
  fn->formalsComplete = formalsComplete;
  if (formalsComplete)
    ++stats.numCompleteFunctions;
  for (auto [value, cell] : fg->scalarMap)
    scalarMap[value] = map(cell);
  for (const auto &[value, slots] : fg->aggregateMap)
    aggregateMap[value] = mapSlots(slots);
  for (auto [i, formal] : llvm::enumerate(fg->formals)) {
    fn->formals.push_back(mapOpt(formal));
    const AggregateSlots *slots = fg->lookupSlots(func.getArgument(i));
    fn->formalSlots.push_back(slots ? std::optional(mapSlots(*slots))
                                    : std::nullopt);
  }
  fn->returnCell = mapOpt(fg->returnCell);
  if (fg->returnSlots)
    fn->returnSlots = mapSlots(*fg->returnSlots);
  if (!formalsComplete) {
    // What the function returns reaches its unknown callers.
    const Reason *why = graph->makeReason(Reason::Kind::UnknownCaller, func,
                                          "returned to an unknown caller");
    if (fn->returnCell)
      graph->addFlags(*fn->returnCell, NodeFlags::Incomplete, why);
    if (fn->returnSlots)
      for (auto [off, cell] : *fn->returnSlots)
        if (cell)
          graph->addFlags(cell, NodeFlags::Incomplete, why);
  }

  for (const MemTransferRecord &record : fg->memTransfers)
    memTransfers.push_back(
        {record.op, map(record.dst), map(record.src), record.length});

  // One node per global symbol: the TU's, with its initializer read once.
  // The local copies are unified into it at the cell the symbol's storage
  // starts at in the local graph.
  const Reason *symbolWhy = nullptr;
  for (auto [sym, cell] : fg->symbolCells) {
    if (!symbolWhy)
      symbolWhy = graph->makeReason(Reason::Kind::MergedAt, func,
                                    "one node per global symbol");
    graph->unify(map(cell), Cell(globals->getNode(sym)), symbolWhy);
  }

  for (const CallSiteRecord &local : fg->callSites) {
    TUCallSite call;
    call.site = local;
    CallSiteRecord &site = call.site;
    site.calleePtr = mapOpt(local.calleePtr);
    for (std::optional<Cell> &arg : site.args)
      arg = mapOpt(arg);
    site.result = mapOpt(local.result);
    for (auto &[index, slots] : site.aggregateArgs)
      slots = mapSlots(slots);
    if (local.resultSlots)
      site.resultSlots = mapSlots(*local.resultSlots);

    switch (local.kind) {
    case CallKind::Defined: {
      auto callee = symbols.lookupNearestSymbolFrom<cir::FuncOp>(local.call,
                                                                 local.callee);
      if (isAnalysed(callee, module)) {
        call.resolution = CallResolution::Direct;
        call.callee = callee;
        break;
      }
      // A body this analysis does not walk (another module).
      call.resolution = CallResolution::Unknown;
      applyUnknownCall(
          call, graph->makeReason(
                    Reason::Kind::EscapedTo, local.call,
                    ("@" + local.callee.getValue() + " not analysed").str()));
      break;
    }
    case CallKind::Modelled:
      call.resolution = CallResolution::Model;
      break;
    case CallKind::UnknownDecl:
      call.resolution = CallResolution::Unknown;
      break;
    case CallKind::IndirectDeferred:
      // Resolved on the TU graph, in run().
      call.resolution = CallResolution::Indirect;
      break;
    case CallKind::IndirectResolved:
    case CallKind::IndirectUnknown:
      llvm_unreachable("the local phase defers indirect calls to this one");
    }
    if (call.resolution == CallResolution::Direct)
      ++stats.numDirectCalls;
    fn->callSites.push_back(std::move(call));
  }

  functionMap[func.getOperation()] = fn.get();
  functions.push_back(std::move(fn));
}

size_t PointsToAnalysis::bindCalls(bool dryRun) {
  size_t count = 0;
  for (std::unique_ptr<TUFunction> &fn : functions) {
    for (TUCallSite &call : fn->callSites) {
      if (call.resolution == CallResolution::Direct) {
        const TUFunction *callee =
            functionMap.lookup(call.callee.getOperation());
        assert(callee && "a direct call to a function that was not analysed");
        count += bindCall(call, *callee, dryRun);
        continue;
      }
      // An indirect call stays bound to its targets even once it is
      // unresolved.
      for (FlatSymbolRefAttr target : call.targets) {
        auto func =
            symbols.lookupNearestSymbolFrom<cir::FuncOp>(module, target);
        if (const TUFunction *callee =
                func ? functionMap.lookup(func.getOperation()) : nullptr)
          count += bindCall(call, *callee, dryRun);
      }
    }
  }
  return count;
}

size_t PointsToAnalysis::bindCall(TUCallSite &call, const TUFunction &callee,
                                  bool dryRun) {
  size_t count = 0;
  CallSiteRecord &site = call.site;
  const Reason *why = nullptr;
  auto bind = [&](Cell actual, Cell formal) {
    if (!actual || !formal || actual.resolve() == formal.resolve())
      return;
    ++count;
    if (dryRun)
      return;
    if (!why)
      why = graph->makeReason(
          Reason::Kind::MergedAt, site.call,
          ("@" + cir::FuncOp(callee.func).getSymName()).str());
    graph->unify(actual, formal, why);
  };
  auto bindSlots = [&](const AggregateSlots &actual,
                       const AggregateSlots &formal) {
    for (auto [off, cell] : actual)
      for (auto [formalOff, formalCell] : formal)
        if (off == formalOff)
          bind(cell, formalCell);
  };

  // Actuals to formals. An argument past the formals is variadic: the
  // callee reads it through a va_list, which the local phase treats as
  // unknown, so what it points to escapes.
  size_t numFormals = callee.formals.size();
  for (auto [i, arg] : llvm::enumerate(site.args)) {
    if (i < numFormals) {
      if (arg && callee.formals[i])
        bind(*arg, *callee.formals[i]);
      continue;
    }
    if (arg && !dryRun)
      graph->addFlags(*arg,
                      NodeFlags::External | NodeFlags::Mod | NodeFlags::Ref,
                      graph->makeReason(Reason::Kind::EscapedTo, site.call,
                                        "variadic argument"));
  }
  for (const auto &[index, slots] : site.aggregateArgs) {
    if (index < numFormals) {
      if (callee.formalSlots[index])
        bindSlots(slots, *callee.formalSlots[index]);
      continue;
    }
    if (!dryRun)
      for (auto [off, cell] : slots)
        if (cell)
          graph->addFlags(cell,
                          NodeFlags::External | NodeFlags::Mod | NodeFlags::Ref,
                          graph->makeReason(Reason::Kind::EscapedTo, site.call,
                                            "variadic argument"));
  }

  // The result to the return cell. A struct returned through an sret
  // pointer is an argument like any other.
  if (site.result && callee.returnCell)
    bind(*site.result, *callee.returnCell);
  if (site.resultSlots && callee.returnSlots)
    bindSlots(*site.resultSlots, *callee.returnSlots);
  return count;
}

namespace {
/// What one round of resolveIndirectCalls decides for a call, from the graph
/// as it was at the start of the round.
struct IndirectDecision {
  TUCallSite *call;
  /// Set: the call is unresolved, for this reason.
  llvm::StringRef unresolved;
  /// The targets the call did not have yet, sorted by name.
  SmallVector<FlatSymbolRefAttr, 2> newTargets;
};
} // namespace

bool PointsToAnalysis::resolveIndirectCalls(bool finalize, bool &bound) {
  // Decide every call on the graph as it is at the start of the round, then
  // apply the decisions: a round then means one step of the chain "binding a
  // target gives another call a target", whatever the order of the calls.
  SmallVector<IndirectDecision, 0> decisions;
  for (std::unique_ptr<TUFunction> &fn : functions)
    for (TUCallSite &call : fn->callSites) {
      if (call.resolution != CallResolution::Indirect)
        continue;
      IndirectDecision decision{&call, {}, {}};
      if (!call.site.calleePtr) {
        decision.unresolved = "callee is a null pointer";
        decisions.push_back(std::move(decision));
        continue;
      }
      // SOUNDNESS: the targets are known only if the callee node stands for
      // nothing but the functions named by its origins: no unknown object,
      // nothing that code we don't see could have stored (External), and
      // nothing an unknown caller could have passed in (Incomplete).
      const Node *callee = call.site.calleePtr->getNode();
      NodeFlags flags = callee->getFlags();
      if (hasAny(flags, NodeFlags::UnknownObj))
        decision.unresolved = "callee may be an unknown object";
      else if (hasAny(flags, NodeFlags::External))
        decision.unresolved = "callee node is External";
      else if (hasAny(flags, NodeFlags::Incomplete))
        decision.unresolved = "callee node is Incomplete";
      else if (!llvm::all_of(callee->getOrigins(), [](const Origin &origin) {
                 return origin.kind == Origin::Kind::FuncSym &&
                        isa_and_present<FlatSymbolRefAttr>(origin.sym);
               }))
        decision.unresolved = "callee may point to something not a function";
      else if (callee->getOrigins().empty()) {
        // Nothing seen yet: a later round may store a function there. When
        // nothing else changes any more, it never will.
        if (!finalize)
          continue;
        decision.unresolved = "no target seen";
      }
      if (decision.unresolved.empty()) {
        for (const Origin &origin : callee->getOrigins()) {
          auto sym = cast<FlatSymbolRefAttr>(origin.sym);
          if (!llvm::is_contained(call.targets, sym) &&
              !llvm::is_contained(decision.newTargets, sym))
            decision.newTargets.push_back(sym);
        }
        if (decision.newTargets.empty())
          continue;
        llvm::sort(decision.newTargets,
                   [](FlatSymbolRefAttr a, FlatSymbolRefAttr b) {
                     return a.getValue() < b.getValue();
                   });
      }
      decisions.push_back(std::move(decision));
    }
  if (decisions.empty())
    return false;

  bool overBudget = stats.numIndirectRounds >= options.maxIndirectRounds;
  bool bindsThisRound = false;
  for (IndirectDecision &decision : decisions) {
    TUCallSite &call = *decision.call;
    if (decision.unresolved.empty() && overBudget) {
      ++stats.numIndirectBudget;
      decision.unresolved = "indirect-call rounds exhausted";
    }
    if (!decision.unresolved.empty()) {
      Reason::Kind kind = overBudget && !decision.newTargets.empty()
                              ? Reason::Kind::Budget
                              : Reason::Kind::IndirectUnresolved;
      call.resolution = CallResolution::IndirectUnresolved;
      call.unresolvedWhy =
          graph->makeReason(kind, call.site.call, decision.unresolved);
      applyUnknownCall(call, call.unresolvedWhy);
      continue;
    }
    for (FlatSymbolRefAttr target : decision.newTargets) {
      call.targets.push_back(target);
      auto func = symbols.lookupNearestSymbolFrom<cir::FuncOp>(module, target);
      if (const TUFunction *callee =
              func ? functionMap.lookup(func.getOperation()) : nullptr) {
        bindCall(call, *callee, /*dryRun=*/false);
        ++stats.numIndirectTargetsBound;
        bindsThisRound = true;
        continue;
      }
      applyDeclarationTarget(call, target, func);
    }
  }
  if (bindsThisRound) {
    ++stats.numIndirectRounds;
    bound = true;
  }
  return true;
}

void PointsToAnalysis::applyDeclarationTarget(TUCallSite &call,
                                              FlatSymbolRefAttr target,
                                              cir::FuncOp calleeOrNull) {
  // The same effect a direct call to the declaration got in the local phase,
  // on the TU graph.
  auto op = cast<CIRCallOpInterface>(call.site.call);
  if (models)
    if (const PointsToModel *model =
            models->lookup(op, calleeOrNull, target.getValue())) {
      // The model reads the callee from the site: give it the target. The
      // cells are shared, so the copy's effects are the site's.
      CallSiteRecord site = call.site;
      site.callee = target;
      SmallVector<DeferredEffect, 0> deferred;
      ModelContext ctx(*graph, *dataLayout, deferred, memTransfers);
      model->apply(site, ctx);
      call.site.clobbersGlobals |= site.clobbersGlobals;
      call.site.modelled = model->name();
      for (const DeferredEffect &effect : deferred) {
        switch (effect.kind) {
        case DeferredEffect::Kind::Reachable:
          graph->markReachable(effect.cell, effect.flags, effect.why);
          break;
        }
      }
      return;
    }
  applyUnknownCall(call,
                   graph->makeReason(Reason::Kind::EscapedTo, call.site.call,
                                     ("@" + target.getValue()).str()));
}

void PointsToAnalysis::applyUnknownCall(TUCallSite &call, const Reason *why) {
  // SOUNDNESS: the same rule as the local phase's (LocalBuilder): what the
  // arguments reach escapes (propagateStatus spreads it along the links),
  // the result is unknown, and the globals are clobbered.
  CallSiteRecord &site = call.site;
  ModelContext::forEachPointerArg(site, [&](Cell cell) {
    graph->addFlags(cell, NodeFlags::External | NodeFlags::Mod | NodeFlags::Ref,
                    why);
  });
  auto unknown = [&](Cell cell) {
    Cell fresh(graph->makeNode(NodeFlags::UnknownObj));
    graph->addFlags(fresh, NodeFlags::External | NodeFlags::Incomplete, why);
    graph->unify(cell, fresh, why);
  };
  if (site.result)
    unknown(*site.result);
  if (site.resultSlots)
    for (auto [off, cell] : *site.resultSlots)
      if (cell)
        unknown(cell);
  site.clobbersGlobals = true;
}

bool PointsToAnalysis::markEscapedFunctions() {
  // SOUNDNESS (N13): complete formals (visible callers, or assumeClosedTU)
  // claim that every caller is bound to the function. Code we don't see can
  // call it all the same once its address reaches that code: its Function
  // node is External; or once a call through a pointer to it is unresolved:
  // the unknown-call rule binds no target. Such a caller may pass anything,
  // so the formals and the return cell become Incomplete. Only flags are
  // added, so the fixpoint in run() stays monotone.
  //
  // The targets of an unresolved call are not all known: its callee node may
  // hold a pointer unknown code made (External, Incomplete, UnknownObj, or a
  // non-function origin). Unknown code can only have the address of a
  // function of this TU if it reached that code: through an External node
  // (covered by the first rule) or through a global another TU can name,
  // such as the vtable of a class whose key function is here, which an
  // object built by unknown code points to. So such a call may call every
  // function reachable from a nameable global. (Unknown code naming one of
  // our functions directly is a caller we don't see, which assumeClosedTU
  // rules out.)
  llvm::DenseMap<Operation *, const Reason *> causes;
  auto consider = [&](FlatSymbolRefAttr sym, const Reason *cause) {
    auto func = symbols.lookupNearestSymbolFrom<cir::FuncOp>(module, sym);
    if (!func)
      return;
    TUFunction *fn = functionMap.lookup(func.getOperation());
    if (fn && fn->formalsComplete)
      causes.try_emplace(func.getOperation(), cause);
  };
  // The first cause found is kept: External nodes, then unresolved calls.
  for (auto [sym, raw] : globals->getSymbolNodes()) {
    const Node *node = Cell(raw).getNode();
    if (node->hasFlags(NodeFlags::External))
      consider(sym, node->getReason(NodeFlags::External));
  }
  const Reason *unknownTargetWhy = nullptr;
  for (std::unique_ptr<TUFunction> &fn : functions)
    for (TUCallSite &call : fn->callSites) {
      if (call.resolution != CallResolution::IndirectUnresolved)
        continue;
      for (FlatSymbolRefAttr target : call.targets)
        consider(target, call.unresolvedWhy);
      if (!call.site.calleePtr)
        continue;
      const Node *callee = call.site.calleePtr->getNode();
      bool unknownTarget = hasAny(callee->getFlags(),
                                  NodeFlags::External | NodeFlags::Incomplete |
                                      NodeFlags::UnknownObj);
      for (const Origin &origin : callee->getOrigins()) {
        auto sym = dyn_cast_if_present<FlatSymbolRefAttr>(origin.sym);
        if (origin.kind == Origin::Kind::FuncSym && sym)
          consider(sym, call.unresolvedWhy);
        else
          unknownTarget = true;
      }
      if (unknownTarget && !unknownTargetWhy)
        unknownTargetWhy = call.unresolvedWhy;
    }
  if (unknownTargetWhy) {
    // Every node reachable from a global another TU can name.
    llvm::DenseSet<const Node *> visited;
    SmallVector<const Node *, 0> worklist;
    for (auto [sym, raw] : globals->getSymbolNodes()) {
      Operation *target = symbols.lookupNearestSymbolFrom(module, sym);
      auto global = dyn_cast_if_present<cir::GlobalOp>(target);
      bool nameable = global ? global.isDeclaration() ||
                                   !cir::isLocalLinkage(global.getLinkage())
                             : !isa_and_present<cir::FuncOp>(target);
      const Node *node = Cell(raw).getNode();
      if (nameable && visited.insert(node).second)
        worklist.push_back(node);
    }
    while (!worklist.empty()) {
      const Node *node = worklist.pop_back_val();
      for (const Origin &origin : node->getOrigins())
        if (origin.kind == Origin::Kind::FuncSym)
          if (auto sym = dyn_cast_if_present<FlatSymbolRefAttr>(origin.sym))
            consider(sym, unknownTargetWhy);
      for (auto [off, cell] : node->getLinks()) {
        const Node *next = cell.getNode();
        if (visited.insert(next).second)
          worklist.push_back(next);
      }
    }
  }
  if (causes.empty())
    return false;

  // In processing order, for deterministic reasons.
  for (std::unique_ptr<TUFunction> &fn : functions) {
    auto it = causes.find(fn->func.getOperation());
    if (it == causes.end())
      continue;
    fn->formalsComplete = false;
    ++stats.numEscapedFunctions;
    const Reason *why =
        graph->makeReason(Reason::Kind::UnknownCaller, fn->func,
                          "address escapes to unknown code", it->second);
    auto mark = [&](Cell cell) {
      if (cell)
        graph->addFlags(cell, NodeFlags::Incomplete, why);
    };
    for (const std::optional<Cell> &formal : fn->formals)
      if (formal)
        mark(*formal);
    for (const std::optional<AggregateSlots> &slots : fn->formalSlots)
      if (slots)
        for (auto [off, cell] : *slots)
          mark(cell);
    if (fn->returnCell)
      mark(*fn->returnCell);
    if (fn->returnSlots)
      for (auto [off, cell] : *fn->returnSlots)
        mark(cell);
  }
  return true;
}

void PointsToAnalysis::propagateStatus() {
  // N1: a node reachable from an External (Incomplete) node is External
  // (Incomplete) too. One markReachable per flagged node, in creation order
  // (deterministic reasons), each stopping at nodes already flagged, whose
  // own call covers what they reach.
  for (NodeFlags flag : {NodeFlags::External, NodeFlags::Incomplete}) {
    SmallVector<Node *, 0> sources;
    for (Node *node : graph->nodes())
      if (node->hasFlags(flag))
        sources.push_back(node);
    for (Node *node : sources)
      graph->markReachable(Cell(node), flag, node->getReason(flag),
                           /*stopAtFlagged=*/true);
  }
}

bool PointsToAnalysis::clobberGlobals() {
  // SOUNDNESS: an unknown callee may read or write every global that another
  // TU can name, and every global whose address escaped. An escaped address
  // is in the graph: the global's node is then External or Incomplete (after
  // propagateStatus), which covers a pointer to it passed to, stored where,
  // or returned to code we don't see. A constant cannot be written (UB), and
  // a function is not data.
  // Runs once per round of the fixpoint in run(); returns whether it set a
  // flag that was not there, so that the loop knows whether to go again.
  stats.numClobberingCalls = 0;
  SmallVector<Node *, 0> clobbered;
  for (auto [sym, raw] : globals->getSymbolNodes()) {
    auto global = symbols.lookupNearestSymbolFrom<cir::GlobalOp>(module, sym);
    if (!global || global.getConstant())
      continue;
    Node *node = Cell(raw).getNode();
    if (!cir::isLocalLinkage(global.getLinkage()) ||
        hasAny(node->getFlags(), NodeFlags::External | NodeFlags::Incomplete))
      clobbered.push_back(raw);
  }
  if (clobbered.empty())
    return false;
  constexpr NodeFlags clobber =
      NodeFlags::External | NodeFlags::Mod | NodeFlags::Ref;
  bool changed = false;
  for (std::unique_ptr<TUFunction> &fn : functions)
    for (TUCallSite &call : fn->callSites) {
      if (!call.site.clobbersGlobals)
        continue;
      ++stats.numClobberingCalls;
      std::string detail =
          call.site.callee
              ? ("@" + call.site.callee.getValue() + " may access the global")
                    .str()
              : std::string("an indirect callee may access the global");
      const Reason *why =
          graph->makeReason(Reason::Kind::EscapedTo, call.site.call, detail);
      for (Node *node : clobbered) {
        Cell cell(node);
        changed |= !cell.getNode()->hasFlags(clobber);
        graph->addFlags(cell, clobber, why);
      }
    }
  return changed;
}

//===----------------------------------------------------------------------===//
// Summary objects
//===----------------------------------------------------------------------===//

namespace {
/// A function of the call graph computeRecursion walks; the root is a
/// pseudo-node that calls every function, so that one scc_iterator walk
/// visits them all.
struct CallNode {
  SmallVector<CallNode *, 4> callees;
  unsigned index = 0;
};
} // namespace

namespace llvm {
template <> struct GraphTraits<CallNode *> {
  using NodeRef = CallNode *;
  using ChildIteratorType = SmallVectorImpl<CallNode *>::iterator;
  static NodeRef getEntryNode(CallNode *node) { return node; }
  static ChildIteratorType child_begin(NodeRef node) {
    return node->callees.begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return node->callees.end();
  }
};
} // namespace llvm

/// Call \p fn with the index (into \p functions) of every callee with a body
/// that \p call is bound to: the direct callee, or the bound indirect
/// targets.
template <typename Fn>
static void forEachBoundCallee(const TUCallSite &call,
                               const DenseMap<Operation *, unsigned> &indexOf,
                               SymbolTableCollection &symbols, ModuleOp module,
                               Fn fn) {
  if (call.resolution == CallResolution::Direct) {
    auto it = indexOf.find(cir::FuncOp(call.callee).getOperation());
    if (it != indexOf.end())
      fn(it->second);
    return;
  }
  for (FlatSymbolRefAttr target : call.targets) {
    auto func = symbols.lookupNearestSymbolFrom<cir::FuncOp>(module, target);
    auto it = func ? indexOf.find(func.getOperation()) : indexOf.end();
    if (it != indexOf.end())
      fn(it->second);
  }
}

void PointsToAnalysis::computeRecursion() {
  // The call graph of the final bindings: mlir::CallGraph sees only the
  // direct calls, and a bound indirect target can close a cycle too.
  // Recursion through code we don't see (an escaped function pointer that
  // a library calls back) is not a cycle here; see markSummaries.
  DenseMap<Operation *, unsigned> indexOf;
  for (auto [i, fn] : llvm::enumerate(functions))
    indexOf[fn->func.getOperation()] = i;
  SmallVector<CallNode, 0> nodes(functions.size() + 1);
  CallNode &root = nodes.back();
  for (auto [i, fn] : llvm::enumerate(functions)) {
    nodes[i].index = i;
    root.callees.push_back(&nodes[i]);
    for (const TUCallSite &call : fn->callSites)
      forEachBoundCallee(call, indexOf, symbols, module, [&](unsigned callee) {
        nodes[i].callees.push_back(&nodes[callee]);
      });
  }
  for (auto it = llvm::scc_begin(&root); !it.isAtEnd(); ++it) {
    if (!it.hasCycle())
      continue;
    for (CallNode *node : *it) {
      if (node == &root)
        continue;
      Operation *func = functions[node->index]->func.getOperation();
      recursive[func] =
          reasons.make(Reason::Kind::SummaryObject, func, "recursion");
    }
  }
  stats.numRecursiveFunctions = recursive.size();
}

namespace {
/// Whether an op may run more than once per activation of its function.
class LoopInfo {
public:
  /// SOUNDNESS: true if an enclosing region may repeat (a LoopOpInterface,
  /// a repetitive region of a RegionBranchOpInterface, or any region of an
  /// op that implements neither, such as cir.array.ctor) or an enclosing
  /// block is on a CFG cycle (a flattened loop).
  bool isInLoop(Operation *op) {
    for (Operation *cur = op; cur; cur = cur->getParentOp()) {
      Block *block = cur->getBlock();
      if (!block)
        return false;
      if (isOnCycle(block))
        return true;
      Region *region = block->getParent();
      Operation *parent = region->getParentOp();
      if (!parent || isa<ModuleOp, cir::FuncOp>(parent))
        return false;
      if (isa<cir::LoopOpInterface>(parent))
        return true;
      auto branch = dyn_cast<RegionBranchOpInterface>(parent);
      if (!branch || branch.isRepetitiveRegion(region->getRegionNumber()))
        return true;
    }
    return false;
  }

private:
  bool isOnCycle(Block *block) {
    Region *region = block->getParent();
    // The entry block has no predecessors, so a one-block region has no
    // cycle.
    if (region->hasOneBlock())
      return false;
    auto [it, inserted] = cyclicBlocks.try_emplace(region);
    if (inserted)
      for (auto scc = llvm::scc_begin(region); !scc.isAtEnd(); ++scc)
        if (scc.hasCycle())
          it->second.insert(scc->begin(), scc->end());
    return it->second.contains(block);
  }

  DenseMap<Region *, llvm::SmallPtrSet<Block *, 8>> cyclicBlocks;
};
} // namespace

void PointsToAnalysis::markSummaries() {
  // A node is a summary if it may stand for more than one simultaneously
  // live concrete object:
  //  - it has more than one origin, or an origin and the UnknownObj kind
  //    ("merged origins");
  //  - a heap origin whose allocation may run more than once per activation
  //    of its function: inside a loop or on a CFG cycle ("loop");
  //  - a stack or heap origin of a function in a recursive SCC: two
  //    activations are live at once ("recursion");
  //  - a heap origin of a function that may run more than once
  //    ("several callers"): it has an unknown caller (its formals are not
  //    complete), its address is taken, it has more than one call site, a
  //    call site in a loop, or a caller that may itself run more than once.
  //    The heap objects of the activations all outlive them. The
  //    context-sensitive phase refines this per call path.
  // Global, function and ghost origins stand for one object each.
  //
  // A stack object of a non-recursive function stays a singleton even if
  // the function is called twice from straight-line code: the two
  // activations are not live at the same time, and a pointer to the object
  // that outlives its activation is dangling, so using it is UB. Each
  // object is alive in a single activation, which is where a query about
  // it is asked. The same holds for an alloca in a loop body: each
  // iteration's object is dead before the next one is made.
  //
  // Recursion through code we don't see is covered conservatively: a stack
  // object is also a summary ("may be re-entered") when its function can be
  // entered from code we don't see (an unknown caller or a taken address)
  // and itself calls code we don't see (an unknown or unresolved call),
  // since that code may call back into it while the activation is live.
  DenseMap<Operation *, unsigned> indexOf;
  for (auto [i, fn] : llvm::enumerate(functions))
    indexOf[fn->func.getOperation()] = i;
  size_t n = functions.size();
  LoopInfo loops;

  // Why each function may run more than once while its heap objects live.
  SmallVector<const Reason *, 0> runsMany(n, nullptr);
  SmallVector<unsigned, 0> numCallSites(n, 0);
  SmallVector<Operation *, 0> secondCallSite(n, nullptr);
  SmallVector<SmallVector<std::pair<unsigned, Operation *>, 4>, 0> callees(n);
  for (auto [i, fn] : llvm::enumerate(functions))
    for (const TUCallSite &call : fn->callSites)
      forEachBoundCallee(call, indexOf, symbols, module, [&](unsigned callee) {
        callees[i].emplace_back(callee, call.site.call);
        if (++numCallSites[callee] == 2)
          secondCallSite[callee] = call.site.call;
        if (!runsMany[callee] && loops.isInLoop(call.site.call))
          runsMany[callee] = reasons.make(Reason::Kind::SummaryObject,
                                          call.site.call, "called in a loop");
      });
  llvm::DenseSet<StringRef> addressTaken;
  for (auto [sym, node] : globals->getSymbolNodes())
    addressTaken.insert(sym.getValue());
  for (auto [i, fn] : llvm::enumerate(functions)) {
    if (runsMany[i])
      continue;
    Operation *func = fn->func.getOperation();
    if (const Reason *why = recursive.lookup(func))
      runsMany[i] = why;
    else if (!fn->formalsComplete)
      runsMany[i] =
          reasons.make(Reason::Kind::SummaryObject, func, "unknown caller");
    else if (addressTaken.contains(fn->func.getSymName()))
      runsMany[i] =
          reasons.make(Reason::Kind::SummaryObject, func, "address taken");
    else if (secondCallSite[i])
      runsMany[i] = reasons.make(Reason::Kind::SummaryObject, secondCallSite[i],
                                 "several call sites");
  }
  // A callee of a function that runs more than once does too.
  SmallVector<unsigned, 0> worklist;
  for (unsigned i = 0; i != n; ++i)
    if (runsMany[i])
      worklist.push_back(i);
  while (!worklist.empty()) {
    unsigned caller = worklist.pop_back_val();
    for (auto [callee, op] : callees[caller]) {
      if (runsMany[callee])
        continue;
      runsMany[callee] =
          reasons.make(Reason::Kind::SummaryObject, op,
                       "caller may run several times", runsMany[caller]);
      worklist.push_back(callee);
    }
  }

  // Why each function may have two live activations through code we don't
  // see: it can be entered from there and calls out to there.
  SmallVector<const Reason *, 0> reentered(n, nullptr);
  for (auto [i, fn] : llvm::enumerate(functions)) {
    if (fn->formalsComplete && !addressTaken.contains(fn->func.getSymName()))
      continue;
    for (const TUCallSite &call : fn->callSites)
      if (call.resolution == CallResolution::Unknown ||
          call.resolution == CallResolution::IndirectUnresolved) {
        reentered[i] = reasons.make(Reason::Kind::SummaryObject, call.site.call,
                                    "may be re-entered through unknown code");
        break;
      }
  }

  // The reason \p node is a summary, or null.
  auto summaryReason = [&](const Node *node) -> const Reason * {
    ArrayRef<Origin> origins = node->getOrigins();
    if (origins.empty())
      return nullptr;
    if (origins.size() > 1 || hasAny(node->getFlags(), NodeFlags::UnknownObj))
      return reasons.make(Reason::Kind::SummaryObject,
                          origins[origins.size() > 1 ? 1 : 0].site,
                          "merged origins");
    const Origin &origin = origins.front();
    bool heap = origin.kind == Origin::Kind::HeapCall;
    if (!origin.site || (!heap && origin.kind != Origin::Kind::Alloca))
      return nullptr;
    if (heap && loops.isInLoop(origin.site))
      return reasons.make(Reason::Kind::SummaryObject, origin.site, "loop");
    auto func = origin.site->getParentOfType<cir::FuncOp>();
    auto it = func ? indexOf.find(func.getOperation()) : indexOf.end();
    if (it == indexOf.end())
      return nullptr;
    if (const Reason *rec = recursive.lookup(func.getOperation()))
      return reasons.make(Reason::Kind::SummaryObject, origin.site, "recursion",
                          rec);
    if (heap && runsMany[it->second])
      return reasons.make(Reason::Kind::SummaryObject, origin.site,
                          "several callers", runsMany[it->second]);
    if (!heap && reentered[it->second])
      return reasons.make(Reason::Kind::SummaryObject, origin.site,
                          "may be re-entered", reentered[it->second]);
    return nullptr;
  };

  stats.numSummaryNodes = 0;
  for (Node *node : graph->nodes()) {
    const Reason *why = summaryReason(node);
    if (!why)
      continue;
    graph->addFlags(Cell(node), NodeFlags::Summary, why);
    ++stats.numSummaryNodes;
  }
}
