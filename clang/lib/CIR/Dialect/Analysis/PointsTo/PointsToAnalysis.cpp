//===- PointsToAnalysis.cpp - Interprocedural CIR points-to ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToAnalysis.h"
#include "mlir/Analysis/CallGraph.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/GlobalNodes.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToModels.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToVerifier.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
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
  //    a call resolved earlier into an unresolved one.
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
