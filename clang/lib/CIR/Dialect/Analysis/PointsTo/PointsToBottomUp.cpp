//===- PointsToBottomUp.cpp - Bottom-up cloning for CIR points-to ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The bottom-up (BU) mode of PointsToAnalysis (PointsToMode::BU): each call
// site gets its own copy of its callee's graph (DSA's bottom-up phase with
// heap cloning; Lattner et al., PLDI 2007), so that one allocation site gives
// one node per acyclic calling context.
//
// Graphs. Every function's graph lives in the one TU Graph, made of
//  - the function's own nodes (its local graph);
//  - a copy of the graph of each callee, made at each call site to it;
//  - the shared nodes: those reachable from a global or function symbol
//    (Graph::markShared). There is one node per global, and a copy never
//    copies what a global reaches: it refers to the shared nodes instead
//    (CloneOptions::isShared). So the globals graph is one, and whatever a
//    function stores into a global every other function sees. That is the
//    context-insensitive treatment of what globals reach, and it is sound
//    for the same reason as the CI analysis: two pointers to one object
//    always have the same node.
// The functions of one recursive SCC share one graph (their calls to each
// other are bound as in the CI mode), as do graphs bound to each other
// context-insensitively later (a cycle, or over budget: mergePartitions).
//
// Order. SCCs in post-order (callees first). Binding a call needs its
// callee's graph finished; an SCC reached first through an indirect call is
// processed then (processSCC recurses). A callee still being processed (a
// cycle through an indirect call) can be neither copied nor bound: the copy,
// or the callers' copies of a graph bound to it, would miss what its graph
// gets later. Such a call gets the unknown-call rule and its callee gets
// Incomplete formals (the fallback; sound like any unknown call).
//
// A copy (cloneCallee) at call c of a function F's graph to its callee G:
//  - copies the nodes G's graph reaches from G's formals, return cell, the
//    cells of its indirect call sites and of its launch records, except the
//    shared ones; each copied origin gets the context c followed by its old
//    one (Origin::ctx, a CallPathTable id);
//  - unifies the copy's formals with the actuals and its return with the
//    result (bindMapped);
//  - gives F's graph a copy of each indirect call site of G's graph that is
//    resolved or pending (TUCallSite::prefix): in F's graph the callee
//    pointer may point to more (a virtual call through a formal), so F
//    resolves it again and copies the new targets' graphs into its own;
//  - copies G's launch records (LaunchRecord::ctx) and copy records
//    (MemTransferRecord, rerun by closePartition, N8).
// What is copied is G's graph as it is when F is built: G's own effects are
// all in it then, since the only things the later fixpoint adds to G's graph
// are its own unknown callers (Incomplete formals, which F's copy must not
// get: F is a known caller), the effects of the indirect calls still pending
// in G's graph (F has its own copy of each of those sites and resolves it in
// its own graph), the device-write rule (applied to every launch record,
// copies included), clobbered globals (shared nodes), and the projection
// below.
//
// Indirect calls. In each SCC, after its direct calls are bound, its indirect
// call sites (own and inherited) are resolved by T09's rules in rounds on its
// graph, copying the graphs of new targets; a call whose callee node is
// still empty stays pending, to be resolved by the callers' copies or, in
// the fixpoint, get the unknown-call rule ("no target seen"). A target whose
// graph is already copied along the call's path (TUCallSite::pathFuncs) is a
// cycle: the call is bound to that graph context-insensitively. The rounds
// are capped by maxIndirectRounds per SCC; past it a call that gains a target
// gets the unknown-call rule (Reason Budget). The fixpoint of
// PointsToAnalysis::runFixpoint resolves all call sites again, in every
// graph, the same way.
//
// Budget. A copy that would make the caller's graph exceed maxCloneNodes, or
// all copies exceed maxTotalCloneNodes, is not made: the call is bound to
// the callee's own graph (context-insensitively, Reason Budget), which merges
// the two graphs. Every graph is therefore at most about maxCloneNodes
// nodes, plus what merges bring in.
//
// Status (SOUNDNESS). The fixpoint of runFixpoint runs over the whole TU
// graph, all functions' graphs at once: propagation, clobbered globals,
// escaped functions, the plug-in rule, indirect calls. A function's own
// graph then holds what the function itself does and what its unknown
// callers may do. What its known callers do is in their graphs, on the
// copies; projectClones brings it back, for every copy at every round:
//  - the status flags of each copy node (External, Incomplete, UnknownObj,
//    Mod, Ref, HasUSM, AsyncAccess) are added to the node it was copied
//    from: whatever the caller's context lets unknown code do to the object
//    holds for the callee's view of it;
//  - two nodes of the callee whose copies the caller's graph merged (one
//    object in that context, for example the same pointer passed for two
//    formals) are merged in the callee's graph too, at the same relative
//    offset (collapsing on a conflict, or as an array of the element size
//    of an array copy).
// By induction from the functions without analysed callers, each function's
// own graph is then sound for every context it runs in: distinct nodes are
// distinct objects in each activation, and the flags cover what every
// caller does. That is why queries may use a function's own graph (values of
// two functions are MayAlias, see PointsToQueries.cpp). The projection only
// adds flags and merges nodes, so the fixpoint stays monotone. It
// terminates: the graph is finite but for the copies the fixpoint makes when
// it binds an indirect target, and those are bounded: each (call site,
// target) pair is bound once, an inherited site is a copy of a site along an
// acyclic path of copies, and the budgets bound the nodes.
//
// Summary (T10 revisited). A heap origin in a calling context is a summary
// only if the path may run more than once per activation of the function
// whose graph holds it (a call on it in a loop or in a recursive function),
// or that function runs several times; see markSummaries.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToAnalysis.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "cir-points-to"

using namespace mlir;
using namespace cir;
using namespace cir::pointsto;

//===----------------------------------------------------------------------===//
// CallPathTable
//===----------------------------------------------------------------------===//

uint32_t CallPathTable::intern(Operation *call, uint32_t parent) {
  auto [it, inserted] =
      ids.try_emplace({call, parent}, static_cast<uint32_t>(entries.size()));
  if (inserted)
    entries.emplace_back(call, parent);
  return it->second;
}

std::optional<uint32_t> CallPathTable::lookup(Operation *call,
                                              uint32_t parent) const {
  auto it = ids.find({call, parent});
  if (it == ids.end())
    return std::nullopt;
  return it->second;
}

uint32_t CallPathTable::concat(uint32_t prefix, uint32_t path) {
  if (!prefix)
    return path;
  SmallVector<Operation *, 4> calls = getPath(prefix);
  for (Operation *call : llvm::reverse(calls))
    path = intern(call, path);
  return path;
}

SmallVector<Operation *, 4> CallPathTable::getPath(uint32_t id) const {
  SmallVector<Operation *, 4> calls;
  for (; id; id = entries[id].second)
    calls.push_back(entries[id].first);
  return calls;
}

//===----------------------------------------------------------------------===//
// Graphs of SCCs
//===----------------------------------------------------------------------===//

unsigned PointsToAnalysis::findPartition(unsigned scc) {
  unsigned root = scc;
  while (partitions[root].parent != root)
    root = partitions[root].parent;
  while (partitions[scc].parent != root) {
    unsigned next = partitions[scc].parent;
    partitions[scc].parent = root;
    scc = next;
  }
  return root;
}

void PointsToAnalysis::mergePartitions(unsigned into, unsigned from) {
  into = findPartition(into);
  from = findPartition(from);
  if (into == from)
    return;
  Partition &dst = partitions[into];
  Partition &src = partitions[from];
  src.parent = into;
  llvm::append_range(dst.members, src.members);
  llvm::append_range(dst.calls, src.calls);
  llvm::append_range(dst.launches, src.launches);
  llvm::append_range(dst.transfers, src.transfers);
  dst.nodes += src.nodes;
  src.members.clear();
  src.calls.clear();
  src.launches.clear();
  src.transfers.clear();
}

bool PointsToAnalysis::closePartition(unsigned scc) {
  Partition &partition = partitions[findPartition(scc)];
  SmallVector<MemTransferRecord, 0> records;
  records.reserve(partition.transfers.size());
  for (unsigned i : partition.transfers)
    records.push_back(memTransfers[i]);
  return closeMemTransfers(*graph, records);
}

void PointsToAnalysis::runBottomUp(ArrayRef<SmallVector<cir::FuncOp, 1>> sccs,
                                   ArrayRef<bool> cyclic) {
  partitions.resize(sccs.size());
  for (auto [i, scc] : llvm::enumerate(sccs)) {
    Partition &partition = partitions[i];
    partition.parent = i;
    partition.funcs = scc;
    partition.cyclic = cyclic[i];
    for (cir::FuncOp func : scc)
      sccOf[func.getOperation()] = i;
  }
  for (unsigned i = 0, e = sccs.size(); i != e; ++i)
    if (partitions[i].state == Partition::State::NotStarted)
      processSCC(i);
  for (const std::unique_ptr<TUFunction> &fn : functions)
    stats.numInheritedCalls += fn->inheritedCalls.size();
}

void PointsToAnalysis::processSCC(unsigned scc) {
  assert(partitions[scc].state == Partition::State::NotStarted &&
         findPartition(scc) == scc && "SCC processed twice");
  partitions[scc].state = Partition::State::InProgress;
  for (cir::FuncOp func : SmallVector<cir::FuncOp, 1>(partitions[scc].funcs)) {
    bool complete = (options.assumeClosedTU ||
                     visibleCallers.contains(func.getOperation())) &&
                    !unboundCallers.contains(func.getOperation());
    buildLocal(func, complete);
  }

  // The direct calls: a copy of the callee's graph, or the callee's own
  // graph inside the SCC. The members are those built above: a merge only
  // brings in members of other SCCs, whose calls are bound already.
  SmallVector<TUFunction *, 1> members(partitions[scc].members);
  for (TUFunction *fn : members)
    for (TUCallSite &call : fn->callSites) {
      if (call.resolution == CallResolution::Indirect) {
        partitions[findPartition(scc)].calls.emplace_back(&call, fn);
        continue;
      }
      if (call.resolution != CallResolution::Direct)
        continue;
      if (bindBottomUp(call, *fn, call.callee) == BindResult::Bound)
        continue;
      // SOUNDNESS: see the fallback in the file comment.
      call.fallback = true;
      call.notClonedWhy = graph->makeReason(
          Reason::Kind::IndirectUnresolved, call.site.call,
          ("@" + cir::FuncOp(call.callee).getSymName() +
           " is still being analysed: recursion through an indirect call")
              .str());
      applyUnknownCall(call, call.notClonedWhy);
      fallbackCallees.insert(call.callee.getOperation());
    }
  closePartition(scc);

  // The indirect calls, in rounds on this graph.
  unsigned rounds = 0;
  while (true) {
    const auto &list = partitions[findPartition(scc)].calls;
    SmallVector<std::pair<TUCallSite *, TUFunction *>, 0> calls(list.begin(),
                                                                list.end());
    bool bound = false;
    if (!resolveIndirectRound(calls, /*finalize=*/false,
                              rounds >= options.maxIndirectRounds, bound))
      break;
    if (bound) {
      ++rounds;
      ++numIndirectRoundsBU;
      closePartition(scc);
    }
  }

  // Subobjects: a binding may use a member as its enclosing object (a child
  // at offset 0 past its extent). The fixpoint folds such children back
  // (absorbOverflowingSubobjects), in this graph and in every copy of it:
  // a copy has the same extent, so it overflows too.

  Partition &done = partitions[findPartition(scc)];
  done.state = Partition::State::Done;
  graph->flushShared();
  if (done.nodes > stats.maxGraphNodes) {
    stats.maxGraphNodes = done.nodes;
    stats.largestGraph = partitions[scc].funcs.front().getOperation();
  }
}

PointsToAnalysis::BindResult
PointsToAnalysis::bindBottomUp(TUCallSite &call, TUFunction &owner,
                               cir::FuncOp calleeFunc) {
  auto calleeName = [&] { return ("@" + calleeFunc.getSymName()).str(); };
  unsigned calleeSCC = sccOf.lookup(calleeFunc.getOperation());
  // Inside the SCC (or a graph merged into it): its own graph, as in CI.
  if (findPartition(calleeSCC) == findPartition(owner.scc)) {
    TUFunction &callee = *functionMap.lookup(calleeFunc.getOperation());
    const Reason *why =
        graph->makeReason(Reason::Kind::MergedAt, call.site.call, calleeName());
    bindMapped(call, callee, -1, [](Cell cell) { return cell; }, why);
    if (!call.notClonedWhy)
      call.notClonedWhy = why;
    ++stats.numSCCBindings;
    return BindResult::Bound;
  }
  if (partitions[findPartition(calleeSCC)].state ==
      Partition::State::NotStarted)
    processSCC(findPartition(calleeSCC));
  unsigned calleePart = findPartition(calleeSCC);
  unsigned ownerPart = findPartition(owner.scc);
  if (partitions[calleePart].state != Partition::State::Done)
    return BindResult::Fallback;
  TUFunction &callee = *functionMap.lookup(calleeFunc.getOperation());

  // A cycle: the callee's graph is copied along the path of this call
  // already (a recursion that the call graph of direct calls does not
  // show). Bind it to that graph, which is finished.
  bool cycle = llvm::any_of(call.pathFuncs, [&](Operation *func) {
    return findPartition(sccOf.lookup(func)) == calleePart;
  });
  // Over budget: bind it to the callee's own graph too.
  size_t estimate = partitions[calleePart].nodes;
  bool overBudget =
      partitions[ownerPart].nodes + estimate > options.maxCloneNodes ||
      stats.numClonedNodes + estimate > options.maxTotalCloneNodes;
  if (cycle || overBudget) {
    const Reason *why =
        cycle ? graph->makeReason(Reason::Kind::MergedAt, call.site.call,
                                  calleeName() + ": a cycle of copies")
              : graph->makeReason(
                    Reason::Kind::Budget, call.site.call,
                    calleeName() + ": the copy would exceed the clone budget");
    // SOUNDNESS: binding to the callee's own graph is the CI rule. The two
    // graphs become one, so that a later copy of either has the call sites
    // and records of both.
    mergePartitions(ownerPart, calleePart);
    bindMapped(call, callee, -1, [](Cell cell) { return cell; }, why);
    call.notClonedWhy = why;
    ++(cycle ? stats.numCycleBindings : stats.numBudgetBindings);
    return BindResult::Bound;
  }
  cloneCallee(call, owner, callee);
  return BindResult::Bound;
}

void PointsToAnalysis::bindMapped(TUCallSite &call, const TUFunction &callee,
                                  int clone, function_ref<Cell(Cell)> map,
                                  const Reason *why) {
  // The CI binding (bindCall) between the call's cells and the callee's
  // cells mapped into the call's graph.
  TUCallSite::Binding binding;
  binding.callee = callee.func;
  binding.clone = clone;
  CallSiteRecord &site = call.site;
  auto bind = [&](Cell actual, Cell formal) {
    if (!actual || !formal)
      return;
    Cell mapped = map(formal);
    if (!mapped)
      return;
    graph->unify(actual, mapped, why);
    binding.pairs.emplace_back(actual, mapped);
  };
  auto bindSlots = [&](const AggregateSlots &actual,
                       const AggregateSlots &formal) {
    for (auto [off, cell] : actual)
      for (auto [formalOff, formalCell] : formal)
        if (off == formalOff)
          bind(cell, formalCell);
  };
  const Reason *variadic = nullptr;
  auto escape = [&](Cell cell) {
    if (!variadic)
      variadic = graph->makeReason(Reason::Kind::EscapedTo, site.call,
                                   "variadic argument");
    graph->addFlags(cell, NodeFlags::External | NodeFlags::Mod | NodeFlags::Ref,
                    variadic);
  };
  size_t numFormals = callee.formals.size();
  for (auto [i, arg] : llvm::enumerate(site.args)) {
    if (i < numFormals) {
      if (arg && callee.formals[i])
        bind(*arg, *callee.formals[i]);
      continue;
    }
    if (arg)
      escape(*arg);
  }
  for (const auto &[index, slots] : site.aggregateArgs) {
    if (index < numFormals) {
      if (callee.formalSlots[index])
        bindSlots(slots, *callee.formalSlots[index]);
      continue;
    }
    for (auto [off, cell] : slots)
      if (cell)
        escape(cell);
  }
  if (site.result && callee.returnCell)
    bind(*site.result, *callee.returnCell);
  if (site.resultSlots && callee.returnSlots)
    bindSlots(*site.resultSlots, *callee.returnSlots);
  stats.numBindings += binding.pairs.size();
  call.bindings.push_back(std::move(binding));
}

void PointsToAnalysis::cloneCallee(TUCallSite &call, TUFunction &owner,
                                   TUFunction &callee) {
  graph->flushShared();
  unsigned ownerPart = findPartition(owner.scc);
  unsigned calleePart = findPartition(callee.scc);
  // Copies of the lists: the vectors below grow while they are read.
  SmallVector<std::pair<TUCallSite *, TUFunction *>, 0> inherit;
  for (auto [site, holder] : partitions[calleePart].calls)
    if (site->resolution == CallResolution::Indirect)
      inherit.emplace_back(site, holder);
  SmallVector<unsigned, 0> calleeLaunches(
      partitions[calleePart].launches.begin(),
      partitions[calleePart].launches.end());
  SmallVector<unsigned, 0> calleeTransfers(
      partitions[calleePart].transfers.begin(),
      partitions[calleePart].transfers.end());

  // The roots: what a caller can reach of the callee's graph, and what the
  // call sites and records the caller inherits name.
  SmallVector<Cell, 0> roots;
  auto addSite = [&](const CallSiteRecord &site) {
    if (site.calleePtr)
      roots.push_back(*site.calleePtr);
    if (site.vptrHolder)
      roots.push_back(*site.vptrHolder);
    for (const std::optional<Cell> &arg : site.args)
      if (arg)
        roots.push_back(*arg);
    for (const auto &[index, slots] : site.aggregateArgs)
      for (auto [off, cell] : slots)
        if (cell)
          roots.push_back(cell);
    if (site.result)
      roots.push_back(*site.result);
    if (site.resultSlots)
      for (auto [off, cell] : *site.resultSlots)
        if (cell)
          roots.push_back(cell);
  };
  for (const std::optional<Cell> &formal : callee.formals)
    if (formal)
      roots.push_back(*formal);
  for (const std::optional<AggregateSlots> &slots : callee.formalSlots)
    if (slots)
      for (auto [off, cell] : *slots)
        if (cell)
          roots.push_back(cell);
  if (callee.returnCell)
    roots.push_back(*callee.returnCell);
  if (callee.returnSlots)
    for (auto [off, cell] : *callee.returnSlots)
      if (cell)
        roots.push_back(cell);
  for (auto [site, holder] : inherit)
    addSite(site->site);
  for (unsigned i : calleeLaunches) {
    if (launches[i].closure)
      roots.push_back(*launches[i].closure);
    if (launches[i].geometry)
      roots.push_back(*launches[i].geometry);
  }

  // The context of each copied origin: this call (after the path that
  // brought the call into the owner's graph) followed by its old one.
  DenseMap<uint32_t, uint32_t> rebased;
  auto rebase = [&](uint32_t ctx) {
    auto [it, inserted] = rebased.try_emplace(ctx, 0);
    if (inserted)
      it->second =
          callPaths.concat(call.prefix, callPaths.intern(call.site.call, ctx));
    return it->second;
  };
  CloneOptions cloneOptions;
  auto isShared = [&](const Node *node) { return graph->isShared(node); };
  auto rewrite = [&](Origin &origin) { origin.ctx = rebase(origin.ctx); };
  cloneOptions.isShared = isShared;
  cloneOptions.rewriteOrigin = rewrite;
  auto record = std::make_unique<TUClone>(
      TUClone{&owner, &callee, call.site.call, rebase(0),
              graph->cloneInto(*graph, roots, NodeFlags::All, cloneOptions)});
  const CloneMap &map = record->map;
  size_t copied = 0;
  for (const auto &[src, dst] : map.entries())
    if (dst.getRawNode() != src)
      ++copied;
  int index = clones.size();
  clones.push_back(std::move(record));
  auto mapCell = [&](Cell cell) -> Cell {
    if (!cell)
      return Cell();
    if (graph->isShared(cell.getNode()))
      return cell.resolve();
    return map.map(cell);
  };
  auto mapOpt = [&](std::optional<Cell> &cell) {
    if (cell)
      cell = mapCell(*cell);
  };
  auto mapSlots = [&](AggregateSlots &slots) {
    for (auto &[off, cell] : slots)
      cell = mapCell(cell);
  };

  bindMapped(
      call, callee, index, mapCell,
      graph->makeReason(Reason::Kind::MergedAt, call.site.call,
                        ("@" + cir::FuncOp(callee.func).getSymName()).str()));

  // The indirect call sites, for this graph to resolve.
  Partition &dst = partitions[ownerPart];
  for (auto [site, holder] : inherit) {
    auto copy = std::make_unique<TUCallSite>(*site);
    CallSiteRecord &rec = copy->site;
    mapOpt(rec.calleePtr);
    mapOpt(rec.vptrHolder);
    for (std::optional<Cell> &arg : rec.args)
      mapOpt(arg);
    mapOpt(rec.result);
    for (auto &[i, slots] : rec.aggregateArgs)
      mapSlots(slots);
    if (rec.resultSlots)
      mapSlots(*rec.resultSlots);
    copy->prefix = callPaths.concat(
        call.prefix, callPaths.intern(call.site.call, site->prefix));
    copy->pathFuncs = call.pathFuncs;
    copy->pathFuncs.push_back(cir::FuncOp(callee.func).getOperation());
    llvm::append_range(copy->pathFuncs, site->pathFuncs);
    copy->bindings.clear();
    copy->numInheritedTargets = copy->targets.size();
    copy->notClonedWhy = nullptr;
    copy->fallback = false;
    dst.calls.emplace_back(copy.get(), &owner);
    owner.inheritedCalls.push_back(std::move(copy));
  }
  // The launch records and the copy records.
  for (unsigned i : calleeLaunches) {
    LaunchRecord launch = launches[i];
    mapOpt(launch.closure);
    mapOpt(launch.geometry);
    launch.ctx = rebase(launch.ctx);
    dst.launches.push_back(launches.size());
    launches.push_back(std::move(launch));
  }
  for (unsigned i : calleeTransfers) {
    MemTransferRecord transfer = memTransfers[i];
    Cell to = mapCell(transfer.dst);
    Cell from = mapCell(transfer.src);
    // A record between two shared nodes is the same record.
    if (!to || !from ||
        (to == transfer.dst.resolve() && from == transfer.src.resolve()))
      continue;
    transfer.dst = to;
    transfer.src = from;
    dst.transfers.push_back(memTransfers.size());
    memTransfers.push_back(transfer);
  }
  dst.nodes += copied;
  ++stats.numClones;
  stats.numClonedNodes += copied;
}

//===----------------------------------------------------------------------===//
// Projection
//===----------------------------------------------------------------------===//

bool PointsToAnalysis::projectClones() {
  constexpr NodeFlags projected = NodeFlags::External | NodeFlags::Incomplete |
                                  NodeFlags::UnknownObj | NodeFlags::Mod |
                                  NodeFlags::Ref | NodeFlags::HasUSM |
                                  NodeFlags::AsyncAccess;
  bool changed = false;
  // The latest copies first: a caller's copy is made after the copies in
  // the callee's graph, so what a root learns reaches the leaves in one
  // pass.
  for (const std::unique_ptr<TUClone> &clone : llvm::reverse(clones)) {
    // Deterministic order: by the id of the source node.
    SmallVector<std::pair<const Node *, Cell>, 0> entries(
        clone->map.entries().begin(), clone->map.entries().end());
    llvm::sort(entries, [](const auto &a, const auto &b) {
      return a.first->getId() < b.first->getId();
    });
    // The source cells (resolved now) whose copies share a node, with the
    // offset of the copy's cell minus the source cell's.
    llvm::MapVector<Node *, SmallVector<std::pair<Node *, int64_t>, 2>> groups;
    for (auto [src, dst] : entries) {
      Cell from = Cell(const_cast<Node *>(src)).resolve();
      Cell to = dst.resolve();
      Node *fromNode = from.getRawNode();
      Node *toNode = to.getRawNode();
      if (fromNode == toNode)
        continue; // A shared node, or merged into its copy since.
      // SOUNDNESS: what the caller's context lets code we don't see do to
      // the object, the callee's view of it must allow too; Mod and Ref
      // keep "no code writes (reads) it" true over every context.
      NodeFlags add = toNode->getFlags() & projected & ~fromNode->getFlags();
      for (unsigned bit = 0;
           add != NodeFlags::None &&
           (1u << bit) <= static_cast<uint32_t>(NodeFlags::All);
           ++bit) {
        auto flag = static_cast<NodeFlags>(1u << bit);
        if (!hasAny(add, flag))
          continue;
        graph->addFlags(Cell(fromNode), flag, toNode->getReason(flag));
        ++stats.numProjectedFlags;
        changed = true;
      }
      groups[toNode].emplace_back(fromNode,
                                  to.getRawOffset() - from.getRawOffset());
    }
    const Reason *why = nullptr;
    auto merge = [&](Cell a, Cell b) {
      if (a.resolve() == b.resolve())
        return;
      if (!why)
        why = graph->makeReason(
            Reason::Kind::MergedAt, clone->call,
            "one object in the context of this call to @" +
                cir::FuncOp(clone->callee->func).getSymName().str());
      graph->unify(a, b, why);
      ++stats.numPullBackMerges;
      changed = true;
    };
    for (auto &[toNode, sources] : groups) {
      if (sources.size() < 2)
        continue;
      // SOUNDNESS: in this context the sources are cells of one object, at
      // the offsets the copy's node gives them.
      const Node *image = Cell(toNode).getNode();
      auto [first, firstDelta] = sources.front();
      if (image->isCollapsed()) {
        for (auto [node, delta] : llvm::drop_begin(sources))
          merge(Cell(first), Cell(node));
        Node *merged = Cell(first).getNode();
        if (!merged->isCollapsed()) {
          if (!why)
            why = graph->makeReason(
                Reason::Kind::MergedAt, clone->call,
                "one object in the context of this call to @" +
                    cir::FuncOp(clone->callee->func).getSymName().str());
          graph->collapse(merged, why);
          changed = true;
        }
        continue;
      }
      if (std::optional<uint64_t> elem = image->getArrayElemSize()) {
        auto size = static_cast<int64_t>(*elem);
        for (auto [node, delta] : sources) {
          Node *live = Cell(node).getNode();
          if (!live->isCollapsed() && live->getArrayElemSize() != elem) {
            graph->markArray(live, *elem, why);
            changed = true;
          }
        }
        for (auto [node, delta] : llvm::drop_begin(sources))
          merge(Cell(first),
                Cell(node, (((firstDelta - delta) % size) + size) % size));
        continue;
      }
      for (auto [node, delta] : llvm::drop_begin(sources))
        merge(Cell(first), Cell(node, firstDelta - delta));
    }
  }
  return changed;
}

//===----------------------------------------------------------------------===//
// Launches and contexts
//===----------------------------------------------------------------------===//

const TUFunction *
PointsToAnalysis::getLaunchOwner(const LaunchRecord &launch) const {
  Operation *call =
      launch.ctx ? callPaths.getPath(launch.ctx).front() : launch.call;
  auto func = call->getParentOfType<cir::FuncOp>();
  return func ? functionMap.lookup(func.getOperation()) : nullptr;
}

void PointsToAnalysis::computeCoveringLaunches() {
  // A function's records are covered by their copies when every execution
  // of the function is in a context that copied its graph: its formals are
  // complete (no caller we don't see, its address does not escape), it is
  // not recursive, it has an analysed caller, and every caller copied it:
  // no call is bound to its own graph, gets the fallback, or is an
  // unresolved indirect call that names it.
  llvm::DenseMap<Operation *, unsigned> numCopies;
  llvm::DenseSet<Operation *> notCovered(fallbackCallees.begin(),
                                         fallbackCallees.end());
  for (auto [call, owner] : allCallSites()) {
    for (const TUCallSite::Binding &binding : call->bindings) {
      Operation *callee = cir::FuncOp(binding.callee).getOperation();
      if (binding.clone < 0)
        notCovered.insert(callee);
      else
        ++numCopies[callee];
    }
    if (call->resolution == CallResolution::IndirectUnresolved)
      for (FlatSymbolRefAttr target : call->targets)
        if (auto func =
                symbols.lookupNearestSymbolFrom<cir::FuncOp>(module, target))
          notCovered.insert(func.getOperation());
  }
  auto covered = [&](const TUFunction *fn) {
    Operation *func = cir::FuncOp(fn->func).getOperation();
    return fn->formalsComplete && !recursive.contains(func) &&
           numCopies.lookup(func) && !notCovered.contains(func);
  };
  coveredClones.clear();
  for (const std::unique_ptr<TUClone> &clone : clones)
    if (covered(clone->callee))
      coveredClones[clone->callee].push_back(clone.get());
  coveringLaunches.clear();
  for (const LaunchRecord &launch : launches) {
    const TUFunction *owner = getLaunchOwner(launch);
    if (!owner || !covered(owner))
      coveringLaunches.push_back(launch);
  }
}

bool PointsToAnalysis::isCovered(cir::FuncOp func) const {
  const TUFunction *fn = functionMap.lookup(func.getOperation());
  return fn && coveredClones.contains(fn);
}

ObjectSet PointsToAnalysis::pointsTo(Value value,
                                     ArrayRef<Operation *> callPath) const {
  ObjectSet own = pointsTo(value);
  if (options.mode != PointsToMode::BU || callPath.empty() || own.cells.empty())
    return own;
  auto func = value.getParentRegion()->getParentOfType<cir::FuncOp>();
  const TUFunction *fn =
      func ? functionMap.lookup(func.getOperation()) : nullptr;
  if (!fn)
    return own;

  // The copies by (context, function copied).
  DenseMap<std::pair<uint32_t, const TUFunction *>, const TUClone *> byContext;
  for (const std::unique_ptr<TUClone> &clone : clones)
    byContext.try_emplace({clone->ctx, clone->callee}, clone.get());

  // Map through the copies, the longest suffix of the path first: a copy
  // made in the graph of the function holding callPath[start] at the
  // context callPath[start..end). A source node merged since the copy was
  // made (projectClones) stands for each of the nodes merged into it: the
  // copies of all of them are in the result.
  SmallVector<Cell, 2> cells{own.cells.front()};
  size_t end = callPath.size();
  while (end > 0) {
    const TUClone *clone = nullptr;
    size_t start = 0;
    for (; start < end && !clone; ++start) {
      std::optional<uint32_t> id = 0;
      for (size_t j = end; j-- > start && id;)
        id = callPaths.lookup(callPath[j], *id);
      if (!id)
        continue;
      auto it = byContext.find({*id, fn});
      if (it != byContext.end())
        clone = it->second;
    }
    if (!clone)
      return own;
    --start;
    SmallVector<Cell, 2> next;
    for (Cell cell : cells) {
      Cell resolved = cell.resolve();
      if (graph->isShared(resolved.getNode())) {
        if (!llvm::is_contained(next, resolved))
          next.push_back(resolved);
        continue;
      }
      for (const auto &[src, dst] : clone->map.entries()) {
        Cell from = Cell(const_cast<Node *>(src)).resolve();
        if (from.getNode() != resolved.getNode())
          continue;
        Cell image =
            Cell(dst.getRawNode(),
                 dst.getRawOffset() + resolved.getOffset() - from.getOffset())
                .resolve();
        if (!llvm::is_contained(next, image))
          next.push_back(image);
      }
    }
    if (next.empty())
      return own;
    cells = std::move(next);
    fn = clone->caller;
    end = start;
  }
  ObjectSet set;
  for (Cell cell : cells) {
    set.cells.push_back(cell);
    const Node *node = cell.getNode();
    if (set.complete &&
        hasAny(node->getFlags(), NodeFlags::Incomplete | NodeFlags::External |
                                     NodeFlags::UnknownObj)) {
      set.complete = false;
      set.why = node->getReason(NodeFlags::Incomplete);
      if (!set.why)
        set.why = node->getReason(NodeFlags::External);
    }
  }
  return set;
}
