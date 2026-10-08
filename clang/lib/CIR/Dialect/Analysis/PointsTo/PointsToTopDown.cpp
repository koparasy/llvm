//===- PointsToTopDown.cpp - Top-down queries for CIR points-to -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Top-down queries of the bottom-up mode (PointsToMode::BU with
// PointsToAnalysisOptions::topDownQueries): the caller facts that DSA's
// top-down phase would unify into a callee's graph, used at query time
// instead, by enumerating the callee's calling contexts.
//
// Why. In BU mode a function F's own graph is sound for all of its callers:
// projectClones (PointsToBottomUp.cpp) adds to it the flags every copy got
// in its caller's graph and merges the nodes any caller's context merged. So
// it is the join of all of F's contexts, and a query about two values of F
// answered there is MayAlias as soon as one context makes the pair MayAlias,
// or one context makes the first object escape and another the second. The
// copies of F's graph in its callers' graphs know each context apart.
//
// What. For a covered function F (isCovered: its formals are complete after
// the fixpoint, it is not recursive, and every call to it, of at least one,
// is a copy of its graph; the same condition computeCoveringLaunches uses
// for the launch records), a query about values of F
//  - maps their cells through each copy of F's graph (TUClone) into the
//    caller's graph (mapThroughClone);
//  - recursively, while the caller is covered too, through each copy of the
//    caller's graph into its callers' graphs;
//  - and answers in each graph it stops at (a leaf: a function that is not
//    covered, or a cell without a faithful image, see below), with the rules
//    of PointsToQueries.cpp on the mapped cells.
// alias() is the meet over the leaves: NoAlias only if every leaf says
// NoAlias, MustAlias only if every one says MustAlias, PartialAlias if every
// one says MustAlias or PartialAlias (AliasResult::merge), MayAlias
// otherwise; a leaf that maps a value to several cells answers the join over
// the pairs. pointsToInContexts is the union of the leaves' cells, complete
// only if every one is. getModRef of a call intersects the own graph's
// answer with the join over the leaves, where the call's reach is closed in
// each leaf's graph. The query stops as soon as the meet can't be better
// than the own graph's answer (a leaf answers MayAlias, or ModRef), and past
// PointsToAnalysisOptions::maxQueryContexts leaves it answers from F's own
// graph instead (Budget).
//
// Images. The copy map of a TUClone relates F's nodes as they were when the
// copy was made to the copies. A cell (N, o) of F's graph now (resolved) has
// an image for each copied node that resolves to N: if that node now starts
// at offset d of N, its copy C gives (C, o - d). A node merged into N since
// the copy (projectClones' pull-back, or a merge in F's own graph that the
// caller's copy of the same rule makes too) gives N several images: the
// value may be in any of them, so the context keeps all of them.
// A node that became collapsed, or an array, after the copy (another
// context's pull-back made it so) keeps no exact offsets; its image keeps
// the old layout, so (C, o - d) need not be where the value is in C. Such an
// image is not faithful: the context is answered in the graph the walk is
// in instead (F's own graph at the first step), unless the image node is
// collapsed too, or an array of the same element size.
// A cell with no image at all is a node the copy didn't include: one made
// after the copy, or not reachable from F's roots (formals, return, the
// cells of the indirect calls and launch records the caller inherits) when
// it was made. The context is answered one level down then too.
// A reach group (getModRef's call reach) stands for every node reachable
// from its cells: it is closed over the links of the graph it is in before
// each copy, and the nodes without an image are dropped: a node no copy
// includes holds only what the function's own code put there, which the
// closure walked through to the nodes the copy has.
//
// SOUNDNESS. Every execution of a covered function F runs in one of the
// contexts that copied F's graph: its callers we don't see would have made
// its formals Incomplete, a call bound to F's own graph (in its SCC, a
// cycle, over budget) or a fallback would make it not covered, and so would
// an unresolved indirect call naming it. Each copy is sound for the
// executions of F in its context, as a part of its caller's graph G, which
// is sound for all of G's contexts (T17's induction: projectClones). If G is
// covered as well, every execution of G (and so of F within it) is in one of
// G's copies, and so on up the call graph, which has no cycle through a
// covered function. So the leaves of the walk cover every execution of F,
// each leaf answers soundly for its executions (its graph is sound, and the
// images are where F's values are in it), and the meet over a set of
// contexts that covers every execution is sound. The own graph is sound for
// every context, so answering a context, or the whole query, from a graph
// lower in the walk (no faithful image, the budget) is sound too.
//
// A query about a function that is not covered answers from its own graph.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToAnalysis.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace cir;
using namespace cir::pointsto;

/// The deepest chain of copies a walk follows: a guard only, the call graph
/// has no cycle through a covered function.
static constexpr unsigned kMaxContextDepth = 256;

const PointsToAnalysis::CloneIndex &
PointsToAnalysis::getCloneIndex(const TUClone &clone) const {
  std::unique_ptr<CloneIndex> &slot = cloneIndex[&clone];
  if (slot)
    return *slot;
  slot = std::make_unique<CloneIndex>();
  for (const auto &[src, dst] : clone.map.entries()) {
    Cell from = Cell(const_cast<Node *>(src)).resolve();
    slot->images[from.getNode()].emplace_back(from.getOffset(), dst);
  }
  return *slot;
}

bool PointsToAnalysis::mapThroughClone(const TUClone &clone,
                                       ArrayRef<Cell> cells,
                                       SmallVectorImpl<Cell> &out,
                                       bool exact) const {
  const CloneIndex &index = getCloneIndex(clone);
  auto add = [&](Cell cell) {
    if (!llvm::is_contained(out, cell))
      out.push_back(cell);
  };
  for (Cell cell : cells) {
    Cell resolved = cell.resolve();
    const Node *node = resolved.getNode();
    bool mapped = false;
    // A shared node is the same node in every graph.
    if (graph->isShared(node)) {
      add(resolved);
      mapped = true;
    }
    auto it = index.images.find(node);
    if (it != index.images.end())
      for (auto [fromOffset, dst] : it->second) {
        Cell copy = dst.resolve();
        const Node *image = copy.getNode();
        if (exact) {
          // SOUNDNESS: see "Images" in the file comment. A node that lost its
          // offsets after the copy maps faithfully only onto an image that
          // lost them the same way.
          if (node->isCollapsed() && !image->isCollapsed())
            return false;
          if (std::optional<uint64_t> elem = node->getArrayElemSize())
            if (!image->isCollapsed() && image->getArrayElemSize() != elem)
              return false;
        }
        add(Cell(dst.getRawNode(),
                 dst.getRawOffset() + resolved.getOffset() - fromOffset)
                .resolve());
        mapped = true;
      }
    if (!mapped && exact)
      return false;
  }
  return true;
}

PointsToAnalysis::ContextWalkResult PointsToAnalysis::forEachContext(
    const TUFunction &fn, ArrayRef<RootGroup> roots,
    function_ref<bool(ArrayRef<RootGroup>)> leaf, unsigned &numContexts) const {
  ContextWalkResult result = ContextWalkResult::Done;
  // Answer the leaf with the reach groups closed in the graph they are in.
  auto visitLeaf = [&](ArrayRef<RootGroup> groups) {
    if (++numContexts > options.maxQueryContexts) {
      result = ContextWalkResult::OverBudget;
      return;
    }
    SmallVector<RootGroup, 4> closed(groups.begin(), groups.end());
    for (RootGroup &group : closed) {
      if (!group.reach)
        continue;
      llvm::DenseSet<const Node *> reach;
      closeReach(group.cells, reach);
      group.cells.clear();
      for (const Node *node : reach)
        group.cells.push_back(Cell(const_cast<Node *>(node)));
    }
    if (!leaf(closed))
      result = ContextWalkResult::Stopped;
  };
  std::function<void(const TUFunction &, ArrayRef<RootGroup>, unsigned)> walk =
      [&](const TUFunction &at, ArrayRef<RootGroup> groups, unsigned depth) {
        auto copies = coveredClones.find(&at);
        if (copies == coveredClones.end() || depth >= kMaxContextDepth) {
          visitLeaf(groups);
          return;
        }
        // The reach groups, closed in this graph once for all the copies.
        SmallVector<RootGroup, 4> here(groups.begin(), groups.end());
        for (RootGroup &group : here) {
          if (!group.reach)
            continue;
          llvm::DenseSet<const Node *> reach;
          closeReach(group.cells, reach);
          group.cells.clear();
          for (const Node *node : reach)
            group.cells.push_back(Cell(const_cast<Node *>(node)));
        }
        for (const TUClone *clone : copies->second) {
          if (result != ContextWalkResult::Done)
            return;
          SmallVector<RootGroup, 4> mapped(here.size());
          bool faithful = true;
          for (auto [group, image] : llvm::zip_equal(here, mapped)) {
            image.reach = group.reach;
            if (!mapThroughClone(*clone, group.cells, image.cells,
                                 /*exact=*/!group.reach)) {
              faithful = false;
              break;
            }
          }
          // SOUNDNESS: this graph is sound for this context too.
          if (!faithful) {
            visitLeaf(groups);
            continue;
          }
          walk(*clone->caller, mapped, depth + 1);
        }
      };
  walk(fn, roots, 0);
  return result;
}

const TUFunction *PointsToAnalysis::getTopDownFunction(Value value) const {
  if (options.mode != PointsToMode::BU || !options.topDownQueries)
    return nullptr;
  Region *region = value.getParentRegion();
  auto func = region ? region->getParentOfType<cir::FuncOp>() : cir::FuncOp();
  const TUFunction *fn =
      func ? functionMap.lookup(func.getOperation()) : nullptr;
  if (!fn)
    return nullptr;
  ++queryStats.numQueries;
  if (!coveredClones.contains(fn)) {
    ++queryStats.numUncovered;
    return nullptr;
  }
  return fn;
}

void PointsToAnalysis::countContexts(ContextWalkResult result,
                                     unsigned numContexts) const {
  switch (result) {
  case ContextWalkResult::Done:
    ++queryStats.numEnumerated;
    break;
  case ContextWalkResult::Stopped:
    ++queryStats.numStopped;
    break;
  case ContextWalkResult::OverBudget:
    ++queryStats.numBudgetHits;
    return;
  }
  queryStats.totalContexts += numContexts;
  queryStats.maxContexts =
      std::max<size_t>(queryStats.maxContexts, numContexts);
}

std::optional<AliasAnswer>
PointsToAnalysis::aliasInContexts(const TUFunction &fn, Cell a, Cell b,
                                  uint64_t sizeA, uint64_t sizeB) const {
  std::optional<AliasResult> meet;
  auto leaf = [&](ArrayRef<RootGroup> groups) {
    for (Cell cellA : groups[0].cells)
      for (Cell cellB : groups[1].cells) {
        AliasResult result = aliasCells(cellA, cellB, sizeA, sizeB).result;
        meet = meet ? meet->merge(result) : result;
        if (meet->isMay())
          return false;
      }
    return true;
  };
  RootGroup groups[2];
  groups[0].cells.push_back(a);
  groups[1].cells.push_back(b);
  unsigned numContexts = 0;
  ContextWalkResult result = forEachContext(fn, groups, leaf, numContexts);
  countContexts(result, numContexts);
  if (result != ContextWalkResult::Done || !meet || meet->isMay())
    return std::nullopt;
  ++queryStats.numRefined;
  return AliasAnswer{*meet, AliasRule::CallingContexts, numContexts};
}

ObjectSet PointsToAnalysis::pointsToInContexts(Value value,
                                               unsigned *numContexts) const {
  if (numContexts)
    *numContexts = 0;
  ObjectSet own = pointsTo(value);
  if (own.cells.empty())
    return own;
  const TUFunction *fn = getTopDownFunction(value);
  if (!fn)
    return own;
  ObjectSet set;
  auto leaf = [&](ArrayRef<RootGroup> groups) {
    for (Cell cell : groups[0].cells) {
      if (!llvm::is_contained(set.cells, cell))
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
    return true;
  };
  RootGroup group;
  group.cells.push_back(own.cells.front());
  unsigned count = 0;
  ContextWalkResult result = forEachContext(*fn, group, leaf, count);
  countContexts(result, count);
  if (result != ContextWalkResult::Done)
    return own;
  if (numContexts)
    *numContexts = count;
  return set;
}

std::optional<ModRefResult>
PointsToAnalysis::getModRefInContexts(const TUFunction &fn, Operation *call,
                                      Value location) const {
  const TUCallSite *site = findCallSite(call);
  ObjectSet set = pointsTo(location);
  if (!site || set.cells.empty())
    return std::nullopt;
  // Group 0: the location; group 1: what the call may reach from its
  // arguments and results, and the symbol nodes its callees may name
  // (shared, the same in every graph).
  RootGroup groups[2];
  groups[0].cells.push_back(set.cells.front());
  groups[1].reach = true;
  collectCallRoots(*site, groups[1].cells, groups[1].cells);
  std::optional<ModRefResult> join;
  auto leaf = [&](ArrayRef<RootGroup> leafGroups) {
    llvm::DenseSet<const Node *> reach;
    for (Cell cell : leafGroups[1].cells)
      reach.insert(cell.getNode());
    ModRefResult result = callModRefOf(leafGroups[0].cells, reach);
    join = join ? join->merge(result) : result;
    return !join->isModAndRef();
  };
  unsigned numContexts = 0;
  ContextWalkResult result = forEachContext(fn, groups, leaf, numContexts);
  countContexts(result, numContexts);
  if (result != ContextWalkResult::Done || !join)
    return std::nullopt;
  ModRefResult own = getCallModRef(call, location);
  ModRefResult both = own;
  both = both.intersect(*join);
  if (both.isMod() != own.isMod() || both.isRef() != own.isRef())
    ++queryStats.numRefined;
  return join;
}
