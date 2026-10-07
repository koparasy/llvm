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

  // Status propagation, global clobbering, and the indirect calls they show
  // to be unresolved, to a fixpoint of all three: an unknown-call rule adds
  // External flags and clobbers globals; propagation can show more globals
  // to be escaped, which the clobbering calls then reach too; and either can
  // make another callee pointer unresolved. Every step only adds flags or
  // merges nodes, both finite, so the loop terminates.
  bool changed = true;
  while (changed) {
    propagateStatus();
    changed = clobberGlobals();
    propagateStatus();
    for (std::unique_ptr<TUFunction> &fn : functions)
      for (TUCallSite &call : fn->callSites) {
        if (call.site.kind != CallKind::IndirectResolved ||
            call.resolution == CallResolution::IndirectUnresolved)
          continue;
        // SOUNDNESS: the local phase resolved the call from a callee pointer
        // that held only function symbols then. In the TU graph it may have
        // gained other targets, or be reachable from unknown code.
        const Node *callee = call.site.calleePtr->getNode();
        bool resolved =
            !hasAny(callee->getFlags(), NodeFlags::UnknownObj |
                                            NodeFlags::Incomplete |
                                            NodeFlags::External) &&
            !callee->getOrigins().empty() &&
            llvm::all_of(callee->getOrigins(), [](const Origin &origin) {
              return origin.kind == Origin::Kind::FuncSym;
            });
        if (resolved)
          continue;
        call.resolution = CallResolution::IndirectUnresolved;
        ++stats.numIndirectUnresolved;
        applyUnknownCall(call,
                         graph->makeReason(Reason::Kind::IndirectUnresolved,
                                           call.site.call, "callee not known"));
        changed = true;
      }
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
    case CallKind::IndirectUnknown:
      call.resolution = CallResolution::IndirectUnresolved;
      break;
    case CallKind::IndirectResolved: {
      // The targets that are declarations already gave their effect. The
      // defined ones are not bound yet (indirect-call resolution is a later
      // step): the unknown-call rule stands in for them.
      bool hasDefinedTarget = false;
      for (const Origin &origin : local.calleePtr->getNode()->getOrigins()) {
        auto sym = dyn_cast_if_present<FlatSymbolRefAttr>(origin.sym);
        if (sym &&
            isAnalysed(symbols.lookupNearestSymbolFrom<cir::FuncOp>(func, sym),
                       module))
          hasDefinedTarget = true;
      }
      if (hasDefinedTarget) {
        call.resolution = CallResolution::IndirectUnresolved;
        ++stats.numIndirectUnresolved;
        applyUnknownCall(
            call, graph->makeReason(Reason::Kind::IndirectUnresolved,
                                    local.call, "defined target not bound"));
      } else {
        call.resolution = local.modelled.empty() ? CallResolution::Unknown
                                                 : CallResolution::Model;
      }
      break;
    }
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
      if (call.resolution != CallResolution::Direct)
        continue;
      const TUFunction *callee = functionMap.lookup(call.callee.getOperation());
      assert(callee && "a direct call to a function that was not analysed");
      CallSiteRecord &site = call.site;
      const Reason *why = nullptr;
      auto bind = [&](Cell actual, Cell formal) {
        if (!actual || !formal || actual.resolve() == formal.resolve())
          return;
        ++count;
        if (dryRun)
          return;
        if (!why)
          why = graph->makeReason(Reason::Kind::MergedAt, site.call,
                                  ("@" + call.callee.getSymName()).str());
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
      size_t numFormals = callee->formals.size();
      for (auto [i, arg] : llvm::enumerate(site.args)) {
        if (i < numFormals) {
          if (arg && callee->formals[i])
            bind(*arg, *callee->formals[i]);
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
          if (callee->formalSlots[index])
            bindSlots(slots, *callee->formalSlots[index]);
          continue;
        }
        if (!dryRun)
          for (auto [off, cell] : slots)
            if (cell)
              graph->addFlags(
                  cell, NodeFlags::External | NodeFlags::Mod | NodeFlags::Ref,
                  graph->makeReason(Reason::Kind::EscapedTo, site.call,
                                    "variadic argument"));
      }

      // The result to the return cell. A struct returned through an sret
      // pointer is an argument like any other.
      if (site.result && callee->returnCell)
        bind(*site.result, *callee->returnCell);
      if (site.resultSlots && callee->returnSlots)
        bindSlots(*site.resultSlots, *callee->returnSlots);
    }
  }
  return count;
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
