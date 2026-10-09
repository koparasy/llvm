//===- PointsToQueries.cpp - Alias and mod/ref queries over points-to -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file answers alias and mod/ref queries from the finished TU graph of
// PointsToAnalysis (alias, aliasWithRule, explainAlias, getModRef).
//
// Notation: a pointer value v points to the cell (N, o) of node N. With
// unification, everything v may point to is in N: two pointers to the same
// concrete object always have the same node. A node is "unknown-reachable"
// when it is External, Incomplete or UnknownObj after status propagation:
// code we don't see may hold a pointer into it, or may have made the pointer
// we hold.
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "clang/CIR/Dialect/Analysis/CIRPointerDecomposition.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToAnalysis.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Interfaces/CIROpInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace cir;
using namespace cir::pointsto;

StringRef cir::pointsto::getAliasRuleName(AliasRule rule) {
  switch (rule) {
  case AliasRule::SameValue:
    return "same-value";
  case AliasRule::NullPointer:
    return "null-pointer";
  case AliasRule::BothNull:
    return "both-null";
  case AliasRule::NotAnalysed:
    return "not-analysed";
  case AliasRule::DistinctObjects:
    return "distinct-objects";
  case AliasRule::UnescapedObject:
    return "unescaped-object";
  case AliasRule::BothUnknown:
    return "both-unknown";
  case AliasRule::SameNodeUnknown:
    return "same-node-unknown";
  case AliasRule::Collapsed:
    return "collapsed";
  case AliasRule::ArrayDisjoint:
    return "array-disjoint";
  case AliasRule::ArrayMayOverlap:
    return "array-may-overlap";
  case AliasRule::DisjointRanges:
    return "disjoint-ranges";
  case AliasRule::SameAddress:
    return "same-address";
  case AliasRule::SameOffsetNotSingleton:
    return "same-offset-not-singleton";
  case AliasRule::UnknownSize:
    return "unknown-size";
  case AliasRule::PartialOverlap:
    return "partial-overlap";
  case AliasRule::OverlapNotSingleton:
    return "overlap-not-singleton";
  case AliasRule::DifferentFunctions:
    return "different-functions";
  case AliasRule::DisjointSubobjects:
    return "disjoint-subobjects";
  case AliasRule::SubobjectOverlap:
    return "subobject-overlap";
  case AliasRule::CallingContexts:
    return "calling-contexts";
  case AliasRule::MayBeShared:
    return "may-be-shared";
  }
  llvm_unreachable("unknown alias rule");
}

static constexpr NodeFlags kUnknownReachable =
    NodeFlags::External | NodeFlags::Incomplete | NodeFlags::UnknownObj;

/// Whether code we don't see may hold a pointer into \p node or may have
/// made a pointer to it.
static bool isUnknownReachable(const Node *node) {
  return hasAny(node->getFlags(), kUnknownReachable);
}

/// Whether \p node stands for exactly one concrete object: one origin, not a
/// Summary (T10, including the re-entry rule N16), and complete. Only such a
/// node may give MustAlias or PartialAlias (N2).
static bool isSingleton(const Node *node) {
  return node->getOrigins().size() == 1 &&
         !hasAny(node->getFlags(),
                 NodeFlags::Summary | NodeFlags::MayBeShared) &&
         !isUnknownReachable(node);
}

/// The one parent of \p node if it is only a member of that parent: no
/// origin of its own (nothing else was merged into it) and exactly one
/// parent link, not on an array (whose child is the member of every
/// element). Null otherwise.
static const ParentLink *getOnlyParent(const Graph &graph, const Node *node,
                                       SmallVectorImpl<ParentLink> &storage) {
  if (!node->getOrigins().empty())
    return nullptr;
  storage = graph.getParents(node);
  if (storage.size() != 1 || storage.front().parent->isArray() ||
      storage.front().parent->isCollapsed())
    return nullptr;
  return &storage.front();
}

/// isSingleton, where a child that is only a member of its parent
/// (getOnlyParent) stands for one object if its parent does.
static bool isSingleton(const Graph &graph, const Node *node) {
  // MayBeShared: the node may also be a global's object, which is not among
  // its origins.
  if (hasAny(node->getFlags(), NodeFlags::Summary | NodeFlags::MayBeShared) ||
      isUnknownReachable(node))
    return false;
  for (unsigned depth = 0; depth < 64; ++depth) {
    if (node->getOrigins().size() == 1)
      return isSingleton(node);
    SmallVector<ParentLink, 1> parents;
    const ParentLink *parent = getOnlyParent(graph, node, parents);
    if (!parent)
      return false;
    node = parent->parent;
    if (hasAny(node->getFlags(), NodeFlags::Summary | NodeFlags::MayBeShared) ||
        isUnknownReachable(node))
      return false;
  }
  return false;
}

/// Whether \p value is a null pointer, or is computed from one by the steps
/// decomposePointer looks through (a cast, a member or a constant stride).
static bool isDerivedFromNull(Value value, const DataLayout &dataLayout) {
  if (isNullPointerConstant(value))
    return true;
  PointerDecomposition decomposed = decomposePointer(value, dataLayout);
  return decomposed.base && isNullPointerConstant(decomposed.base);
}

/// The function \p value belongs to, or null.
static Operation *getFunctionOf(Value value) {
  Region *region = value.getParentRegion();
  return region ? region->getParentOfType<cir::FuncOp>().getOperation()
                : nullptr;
}

uint64_t PointsToAnalysis::getAccessSize(Value ptr,
                                         std::optional<uint64_t> size) const {
  if (size)
    return *size;
  auto ptrTy = dyn_cast<cir::PointerType>(ptr.getType());
  if (!ptrTy)
    return kUnknownSize;
  std::optional<int64_t> bytes =
      getTypeSizeInBytes(ptrTy.getPointee(), *dataLayout);
  return bytes && *bytes >= 0 ? static_cast<uint64_t>(*bytes) : kUnknownSize;
}

//===----------------------------------------------------------------------===//
// alias
//===----------------------------------------------------------------------===//

AliasResult PointsToAnalysis::alias(Value a, Value b,
                                    std::optional<uint64_t> sizeA,
                                    std::optional<uint64_t> sizeB) const {
  return aliasWithRule(a, b, sizeA, sizeB).result;
}

AliasAnswer
PointsToAnalysis::aliasWithRule(Value a, Value b, std::optional<uint64_t> sizeA,
                                std::optional<uint64_t> sizeB) const {
  AliasAnswer own = aliasOwn(a, b, sizeA, sizeB);
  // BU: the own graph's MayAlias between cells of a covered function may be
  // NoAlias (or MustAlias) in each of its calling contexts.
  if (!own.result.isMay() || own.rule == AliasRule::BothNull ||
      own.rule == AliasRule::NotAnalysed ||
      own.rule == AliasRule::DifferentFunctions)
    return own;
  const TUFunction *fn = getTopDownFunction(a);
  if (!fn)
    return own;
  std::optional<AliasAnswer> answer =
      aliasInContexts(*fn, pointsTo(a).cells.front(), pointsTo(b).cells.front(),
                      getAccessSize(a, sizeA), getAccessSize(b, sizeB));
  return answer ? *answer : own;
}

AliasAnswer PointsToAnalysis::aliasOwn(Value a, Value b,
                                       std::optional<uint64_t> sizeA,
                                       std::optional<uint64_t> sizeB) const {
  // The same SSA value is the same address (the MLIR convention, which
  // LocalAliasAnalysis and CIRBasicAliasAnalysis follow too).
  if (a == b)
    return {AliasResult::MustAlias, AliasRule::SameValue};
  // BU: each function has its own graph, and a node of one stands for
  // objects in that function's activations only. A copy of a callee's
  // object and the callee's own node for it are two nodes.
  if (options.mode == PointsToMode::BU && getFunctionOf(a) != getFunctionOf(b))
    return {AliasResult::MayAlias, AliasRule::DifferentFunctions};

  ObjectSet setA = pointsTo(a);
  ObjectSet setB = pointsTo(b);

  // Rule 1, SOUNDNESS: a null pointer points to no object, and an access
  // through it is UB, so it overlaps no access. Two pointers that both come
  // from null are left to the other analyses: CIRBasicAliasAnalysis calls a
  // bitcast of null MustAlias with null, and both answers hold vacuously, so
  // MayAlias keeps the two from contradicting each other.
  bool nullA = setA.cells.empty() && setA.complete;
  bool nullB = setB.cells.empty() && setB.complete;
  if (nullA || nullB) {
    if (isDerivedFromNull(a, *dataLayout) && isDerivedFromNull(b, *dataLayout))
      return {AliasResult::MayAlias, AliasRule::BothNull};
    return {AliasResult::NoAlias, AliasRule::NullPointer};
  }
  // Rule 4: a value of a function this analysis does not walk (no body,
  // another module) has no cell. Nothing is known about it.
  if (setA.cells.empty() || setB.cells.empty())
    return {AliasResult::MayAlias, AliasRule::NotAnalysed};

  return aliasCells(setA.cells.front(), setB.cells.front(),
                    getAccessSize(a, sizeA), getAccessSize(b, sizeB));
}

namespace {
/// A node a cell is in, directly or through subobject links, and the cell's
/// offset in it. Not exact when an array or collapsed node on the way loses
/// the position.
struct AncestorCell {
  const Node *node;
  int64_t offset;
  bool exact;
  /// Every node below this one on the way is only a member of its parent
  /// (getOnlyParent): the cell is this node's offset, in its objects only.
  bool pure;
};
} // namespace

/// The cell \p cell (resolved) and the cells it is in its ancestors: for a
/// child at [o, o + size) of a parent whose start is the child's offset c,
/// the child's offset x is o + (x - c) in the parent.
static SmallVector<AncestorCell, 4> collectAncestors(const Graph &graph,
                                                     Cell cell) {
  constexpr size_t kMaxAncestors = 64;
  SmallVector<AncestorCell, 4> out;
  const Node *node = cell.getRawNode();
  out.push_back({node, cell.getRawOffset(),
                 !node->isCollapsed() && !node->isArray(), /*pure=*/true});
  for (size_t i = 0; i < out.size() && out.size() < kMaxAncestors; ++i) {
    AncestorCell at = out[i];
    SmallVector<ParentLink, 1> parents = graph.getParents(at.node);
    bool onlyMember = at.node->getOrigins().empty() && parents.size() == 1;
    for (const ParentLink &parent : parents) {
      bool exact = at.exact && !parent.parent->isArray() &&
                   !parent.parent->isCollapsed();
      int64_t offset = parent.offset + (at.offset - parent.childOffset);
      if (llvm::any_of(out, [&](const AncestorCell &c) {
            return c.node == parent.parent && c.offset == offset &&
                   c.exact == exact;
          }))
        continue;
      out.push_back({parent.parent, offset, exact, at.pure && onlyMember});
    }
  }
  return out;
}

AliasAnswer PointsToAnalysis::aliasCells(Cell cellA, Cell cellB, uint64_t sizeA,
                                         uint64_t sizeB) const {
  cellA = cellA.resolve();
  cellB = cellB.resolve();
  const Node *nodeA = cellA.getNode();
  const Node *nodeB = cellB.getNode();
  bool unknownA = isUnknownReachable(nodeA);
  bool unknownB = isUnknownReachable(nodeB);
  auto mayBeShared = [&](const Node *node) {
    return node->hasFlags(NodeFlags::MayBeShared) && !graph->isShared(node);
  };

  // Rule 2b: different nodes of one tree of subobjects.
  // SOUNDNESS: a child is the bytes [o, o + size) of each of its parents, so
  // two cells with a common ancestor overlap only where their ranges, placed
  // in that ancestor, do; with no common ancestor they are different objects
  // (rule 2). Every common ancestor is checked: a child with several parents
  // is in each. Unknown code may have made a pointer anywhere in an object
  // it reaches (the argument of rule 3), so then the offsets say nothing.
  if (nodeA != nodeB && graph->hasSubobjects()) {
    SmallVector<AncestorCell, 4> upA = collectAncestors(*graph, cellA);
    SmallVector<AncestorCell, 4> upB = collectAncestors(*graph, cellB);
    bool related = false, overlap = false;
    const AncestorCell *pureA = nullptr, *pureB = nullptr;
    for (const AncestorCell &a : upA)
      for (const AncestorCell &b : upB) {
        if (a.node != b.node)
          continue;
        related = true;
        if (!a.exact || !b.exact || sizeA == kUnknownSize ||
            sizeB == kUnknownSize ||
            (a.offset < b.offset + static_cast<int64_t>(sizeB) &&
             b.offset < a.offset + static_cast<int64_t>(sizeA)) ||
            a.offset == b.offset)
          overlap = true;
        if (!pureA && a.exact && b.exact && a.pure && b.pure) {
          pureA = &a;
          pureB = &b;
        }
      }
    if (related) {
      if (unknownA || unknownB)
        return {AliasResult::MayAlias, AliasRule::SubobjectOverlap};
      if (!overlap)
        return {AliasResult::NoAlias, AliasRule::DisjointSubobjects};
      // SOUNDNESS: on a pure way up (each node below is only a member of
      // its one parent) both cells are fixed bytes of the ancestor's
      // objects; if it is one object, the same-node rules below apply to
      // the offsets in it.
      if (pureA && isSingleton(*graph, pureA->node)) {
        if (pureA->offset == pureB->offset)
          return sizeA == kUnknownSize || sizeB == kUnknownSize
                     ? AliasAnswer{AliasResult::MayAlias,
                                   AliasRule::UnknownSize}
                     : AliasAnswer{AliasResult::MustAlias,
                                   AliasRule::SameAddress};
        if (sizeA != kUnknownSize && sizeB != kUnknownSize)
          return {AliasResult::PartialAlias, AliasRule::PartialOverlap};
      }
      return {AliasResult::MayAlias, AliasRule::SubobjectOverlap};
    }
  }

  // Rule 2: different nodes.
  if (nodeA != nodeB) {
    // SOUNDNESS (BU, V4 F2): a MayBeShared node may be the object of any
    // shared node (a caller passed a global's object, see projectClones), so
    // it may be the object of the other node if that one is shared, or if it
    // may be one too. This comes before the escape argument below: a global
    // that never escaped is the same object as the formal its caller gives
    // it to. Inside one tree of subobjects (rule 2b above) the layout holds
    // whichever object the tree is.
    if ((mayBeShared(nodeA) &&
         (graph->isShared(nodeB) || mayBeShared(nodeB))) ||
        (mayBeShared(nodeB) && graph->isShared(nodeA)))
      return {AliasResult::MayAlias, AliasRule::MayBeShared};
    // SOUNDNESS: with unification every pointer the analysed code may hold
    // to an object points into that object's node, so two different nodes
    // that unknown code cannot reach denote different objects.
    if (!unknownA && !unknownB)
      return {AliasResult::NoAlias, AliasRule::DistinctObjects};
    // SOUNDNESS (the BasicAA escape argument): say A's node may hold
    // pointers unknown code made. B's node is neither External nor
    // Incomplete: no pointer into it was ever passed to, stored where, or
    // returned to code we don't see (status propagation closes that over
    // the links), and none came from an unknown caller. Code we don't see
    // therefore never had B's address, and can't have made a pointer into
    // it. Every pointer the analysed code made into B's object is in B's
    // node, not A's, so A's pointer is not one of them either.
    if (!unknownA || !unknownB)
      return {AliasResult::NoAlias, AliasRule::UnescapedObject};
    // Unknown code may have made both pointers, to the same object.
    return {AliasResult::MayAlias, AliasRule::BothUnknown};
  }

  // Rule 3: the same node.
  // SOUNDNESS: the offsets of a node are meaningful only for pointers the
  // analysed code computed. Unknown code may have made a pointer into any
  // escaped object at any offset, and the analysis records it at the cell
  // it flows to: for example, an unknown callee that stores &x[0] where the
  // analysed code stored &x[2] gives the loaded pointer the cell of x[2].
  // An unknown caller may likewise pass f(p, p) to a function that unifies
  // p + 8 with its second parameter. On a node unknown code can reach, two
  // cells at different offsets may therefore still overlap.
  if (unknownA || unknownB)
    return {AliasResult::MayAlias, AliasRule::SameNodeUnknown};
  // A collapsed node keeps no offsets.
  if (nodeA->isCollapsed())
    return {AliasResult::MayAlias, AliasRule::Collapsed};

  int64_t offA = cellA.getOffset();
  int64_t offB = cellB.getOffset();

  if (std::optional<uint64_t> elem = nodeA->getArrayElemSize()) {
    // SOUNDNESS: offsets on an array node are modulo the element size E, in
    // [0, E). If each access lies within one element (offset + size <= E)
    // and the two element-relative ranges are disjoint, then two different
    // elements are disjoint, and in the same element the ranges are. Equal
    // offsets may be the same element: never NoAlias, even for a size of 0.
    uint64_t e = *elem;
    bool withinA = sizeA != kUnknownSize && static_cast<uint64_t>(offA) <= e &&
                   sizeA <= e - static_cast<uint64_t>(offA);
    bool withinB = sizeB != kUnknownSize && static_cast<uint64_t>(offB) <= e &&
                   sizeB <= e - static_cast<uint64_t>(offB);
    if (e != 0 && offA != offB && withinA && withinB &&
        (static_cast<uint64_t>(offA) + sizeA <= static_cast<uint64_t>(offB) ||
         static_cast<uint64_t>(offB) + sizeB <= static_cast<uint64_t>(offA)))
      return {AliasResult::NoAlias, AliasRule::ArrayDisjoint};
    return {AliasResult::MayAlias, AliasRule::ArrayMayOverlap};
  }

  bool singleton = isSingleton(*graph, nodeA);
  // SOUNDNESS: on a plain node of known code, a cell (N, o) is offset o of
  // one of the objects N stands for, laid out the same way (a conflicting
  // layout collapses the node). Equal offsets are the same address only if
  // N is one object; with a Summary, several or no origins, the two
  // pointers may be in two different objects. Equal offsets are checked
  // first so that an access of size 0 is never called disjoint from one at
  // the same address (CIRBasicAliasAnalysis calls that MustAlias).
  if (offA == offB) {
    if (!singleton)
      return {AliasResult::MayAlias, AliasRule::SameOffsetNotSingleton};
    if (sizeA == kUnknownSize || sizeB == kUnknownSize)
      return {AliasResult::MayAlias, AliasRule::UnknownSize};
    return {AliasResult::MustAlias, AliasRule::SameAddress};
  }
  // The byte ranges [o, o + size); an unknown size reaches to the end.
  auto endOf = [](int64_t off, uint64_t size) -> std::optional<int64_t> {
    if (size == kUnknownSize || size > static_cast<uint64_t>(INT64_MAX))
      return std::nullopt;
    auto [end, overflow] = llvm::AddOverflow(off, static_cast<int64_t>(size));
    if (overflow)
      return std::nullopt;
    return end;
  };
  std::optional<int64_t> endA = endOf(offA, sizeA);
  std::optional<int64_t> endB = endOf(offB, sizeB);
  if ((endA && *endA <= offB) || (endB && *endB <= offA))
    return {AliasResult::NoAlias, AliasRule::DisjointRanges};
  if (!endA || !endB)
    return {AliasResult::MayAlias, AliasRule::UnknownSize};
  if (!singleton)
    return {AliasResult::MayAlias, AliasRule::OverlapNotSingleton};
  return {AliasResult::PartialAlias, AliasRule::PartialOverlap};
}

//===----------------------------------------------------------------------===//
// explainAlias
//===----------------------------------------------------------------------===//

/// explain(why) without the leading "it is ".
static std::string describeReason(const Reason *why) {
  std::string text = explain(why);
  StringRef ref(text);
  if (ref.consume_front("it is "))
    return ref.str();
  return text;
}

/// A short description of \p origin: the symbol, or the op and its location.
static std::string describeOrigin(const Origin &origin) {
  std::string out;
  llvm::raw_string_ostream os(out);
  switch (origin.kind) {
  case Origin::Kind::Alloca:
    os << "stack";
    break;
  case Origin::Kind::GlobalSym:
    os << "global";
    break;
  case Origin::Kind::HeapCall:
    os << "heap";
    break;
  case Origin::Kind::GhostSite:
    os << "ghost";
    break;
  case Origin::Kind::FuncSym:
    os << "function";
    break;
  }
  if (origin.sym)
    os << " " << origin.sym;
  if (Operation *site = origin.site) {
    if (auto name = site->getAttrOfType<StringAttr>("name"))
      os << " '" << name.getValue() << "'";
    if (auto loc = site->getLoc()->findInstanceOf<FileLineColLoc>())
      os << " (" << llvm::sys::path::filename(loc.getFilename().getValue())
         << ":" << loc.getLine() << ")";
  }
  return out;
}

/// N17: for a node several objects were merged into, how many and, if it
/// collapsed, why; the first reason alone says little about such a node.
static std::string describeMerge(const Node *node) {
  std::string out;
  ArrayRef<Origin> origins = node->getOrigins();
  if (origins.size() > 1) {
    out += " [" + std::to_string(origins.size()) +
           " objects merged into one node, e.g. " + describeOrigin(origins[0]) +
           " and " + describeOrigin(origins[1]);
    if (const Reason *why = node->getReason(NodeFlags::Collapsed))
      out += "; collapsed: " + describeReason(why);
    out += "]";
  }
  return out;
}

/// What unknown code has to do with \p node.
static std::string describeUnknown(const Node *node) {
  const Reason *why = node->getReason(NodeFlags::Incomplete);
  if (!why)
    why = node->getReason(NodeFlags::External);
  if (!why)
    return "an unknown object";
  return describeReason(why);
}

std::string
PointsToAnalysis::explainAlias(Value a, Value b, std::optional<uint64_t> sizeA,
                               std::optional<uint64_t> sizeB) const {
  return explainAnswer(aliasWithRule(a, b, sizeA, sizeB), a, b);
}

std::string PointsToAnalysis::explainAnswer(const AliasAnswer &answer, Value a,
                                            Value b) const {
  ObjectSet setA = pointsTo(a);
  ObjectSet setB = pointsTo(b);
  const Node *nodeA = setA.cells.empty() ? nullptr : setA.cells[0].getNode();
  const Node *nodeB = setB.cells.empty() ? nullptr : setB.cells[0].getNode();
  auto side = [](StringRef name, const Node *node) {
    return (name + ": " + describeUnknown(node) + describeMerge(node)).str();
  };
  auto sameNode = [&](StringRef what) {
    return ("a, b: same node" + what + describeMerge(nodeA)).str();
  };
  switch (answer.rule) {
  case AliasRule::SameValue:
    return "a, b: the same value";
  case AliasRule::NullPointer:
    return setA.cells.empty() ? "a: null pointer" : "b: null pointer";
  case AliasRule::BothNull:
    return "a, b: both derived from a null pointer";
  case AliasRule::NotAnalysed: {
    std::string out;
    if (setA.cells.empty())
      out = "a: " + describeReason(setA.why);
    if (setB.cells.empty())
      out += (out.empty() ? "b: " : "; b: ") + describeReason(setB.why);
    return out;
  }
  case AliasRule::DistinctObjects:
    return "a, b: different objects, neither reachable by unknown code";
  case AliasRule::UnescapedObject:
    return isUnknownReachable(nodeA)
               ? side("a", nodeA) + "; b: an object that never escaped"
               : "a: an object that never escaped; " + side("b", nodeB);
  case AliasRule::BothUnknown:
    return side("a", nodeA) + "; " + side("b", nodeB);
  case AliasRule::SameNodeUnknown:
    return sameNode(", which may refer to unknown memory: " +
                    describeUnknown(nodeA));
  case AliasRule::Collapsed: {
    const Reason *why = nodeA->getReason(NodeFlags::Collapsed);
    // describeMerge already names the reason of a merged node.
    if (nodeA->getOrigins().size() > 1 || !why)
      return sameNode(" (collapsed)");
    return sameNode(" (collapsed: " + describeReason(why) + ")");
  }
  case AliasRule::ArrayDisjoint:
    return "a, b: disjoint ranges within an element of the same array node";
  case AliasRule::ArrayMayOverlap:
    return sameNode(" (an array of " +
                    std::to_string(nodeA->getArrayElemSize().value_or(0)) +
                    "-byte elements; the accesses may overlap in one "
                    "element)");
  case AliasRule::DisjointRanges:
    return "a, b: disjoint ranges of the same node";
  case AliasRule::SameAddress:
    return "a, b: the same address of one object";
  case AliasRule::SameOffsetNotSingleton:
  case AliasRule::OverlapNotSingleton: {
    StringRef what = answer.rule == AliasRule::SameOffsetNotSingleton
                         ? " at the same offset"
                         : " at overlapping ranges";
    if (const Reason *why = nodeA->getReason(NodeFlags::Summary))
      return sameNode((what + ", which may stand for several objects: " +
                       describeReason(why))
                          .str());
    return sameNode((what + ", which has no known origin").str());
  }
  case AliasRule::UnknownSize:
    return sameNode(", an access size is unknown");
  case AliasRule::PartialOverlap:
    return "a, b: overlapping ranges of one object";
  case AliasRule::DifferentFunctions:
    return "a, b: values of two functions, whose graphs are apart";
  case AliasRule::DisjointSubobjects:
    return "a, b: disjoint ranges of one object, in different subobjects";
  case AliasRule::SubobjectOverlap:
    return "a, b: subobjects of one object whose ranges may overlap";
  case AliasRule::MayBeShared: {
    const Node *flagged =
        nodeA->hasFlags(NodeFlags::MayBeShared) ? nodeA : nodeB;
    const Reason *why = flagged->getReason(NodeFlags::MayBeShared);
    return "a, b: one may be a global's object" +
           (why ? " (" + why->detail.str() + ")" : std::string()) +
           ", the other is or may be one";
  }
  case AliasRule::CallingContexts: {
    std::string out;
    llvm::raw_string_ostream os(out);
    os << "a, b: " << answer.result << " in each of the " << answer.contexts
       << " calling contexts of @"
       << a.getParentRegion()->getParentOfType<cir::FuncOp>().getSymName();
    return out;
  }
  }
  llvm_unreachable("unknown alias rule");
}

//===----------------------------------------------------------------------===//
// getModRef
//===----------------------------------------------------------------------===//

ModRefResult PointsToAnalysis::getAccessModRef(Value addr,
                                               std::optional<uint64_t> size,
                                               bool mod, bool ref,
                                               Value location) const {
  // SOUNDNESS: the access touches [addr, addr + size), so it reads or
  // writes the location only if the two may overlap.
  if (alias(addr, location, size, std::nullopt).isNo())
    return ModRefResult::getNoModRef();
  if (mod && ref)
    return ModRefResult::getModAndRef();
  return mod ? ModRefResult::getMod() : ModRefResult::getRef();
}

/// The length of a memory op if it is a non-negative constant.
static std::optional<uint64_t> getConstantLength(Value length) {
  std::optional<int64_t> len = getConstantIndex(length);
  if (!len || *len < 0)
    return PointsToAnalysis::kUnknownSize;
  return static_cast<uint64_t>(*len);
}

ModRefResult PointsToAnalysis::getModRef(Operation *op, Value location) const {
  ModRefResult own = getModRefOwn(op, location);
  // BU: what a call of a covered function may access in each of its calling
  // contexts. The other ops ask alias(), which enumerates the contexts.
  if (own.isNoModRef() || !isa<CIRCallOpInterface>(op))
    return own;
  if (op->getParentOfType<cir::FuncOp>().getOperation() !=
      getFunctionOf(location))
    return own;
  const TUFunction *fn = getTopDownFunction(location);
  if (!fn)
    return own;
  std::optional<ModRefResult> inContexts =
      getModRefInContexts(*fn, op, location);
  return inContexts ? own.intersect(*inContexts) : own;
}

ModRefResult PointsToAnalysis::getModRefOwn(Operation *op,
                                            Value location) const {
  using MRR = ModRefResult;
  // BU: an op and a location of two functions are in two graphs.
  if (options.mode == PointsToMode::BU &&
      op->getParentOfType<cir::FuncOp>().getOperation() !=
          getFunctionOf(location))
    return MRR::getModAndRef();
  auto access = [&](Value addr, std::optional<uint64_t> size, bool mod,
                    bool ref) {
    return getAccessModRef(addr, size, mod, ref, location);
  };
  auto copy = [&](Value dst, Value src, std::optional<uint64_t> size) {
    return access(dst, size, /*mod=*/true, /*ref=*/false)
        .merge(access(src, size, /*mod=*/false, /*ref=*/true));
  };
  std::optional<MRR> result =
      llvm::TypeSwitch<Operation *, std::optional<MRR>>(op)
          .Case<cir::LoadOp>([&](cir::LoadOp load) {
            return access(load.getAddr(),
                          getAccessSize(load.getAddr(), std::nullopt),
                          /*mod=*/false, /*ref=*/true);
          })
          .Case<cir::StoreOp>([&](cir::StoreOp store) {
            return access(store.getAddr(), std::nullopt, /*mod=*/true,
                          /*ref=*/false);
          })
          .Case<cir::CopyOp>([&](cir::CopyOp op) {
            return copy(op.getDst(), op.getSrc(),
                        op.getCopySizeInBytes(*dataLayout));
          })
          .Case<cir::MemCpyOp, cir::MemMoveOp>([&](auto op) {
            return copy(op.getDst(), op.getSrc(),
                        getConstantLength(op.getLen()));
          })
          .Case<cir::MemSetOp>([&](cir::MemSetOp op) {
            return access(op.getDst(), getConstantLength(op.getLen()),
                          /*mod=*/true, /*ref=*/false);
          })
          .Case<cir::AtomicFetchOp, cir::AtomicXchgOp, cir::AtomicCmpXchgOp,
                cir::AtomicTestAndSetOp, cir::AtomicClearOp>([&](auto op) {
            return access(op.getPtr(), std::nullopt, /*mod=*/true,
                          /*ref=*/true);
          })
          .Case<cir::GetBitfieldOp>([&](cir::GetBitfieldOp op) {
            return access(op.getAddr(), std::nullopt, /*mod=*/false,
                          /*ref=*/true);
          })
          .Case<cir::SetBitfieldOp>([&](cir::SetBitfieldOp op) {
            return access(op.getAddr(), std::nullopt, /*mod=*/true,
                          /*ref=*/false);
          })
          .Case<cir::VecMaskedLoadOp>([&](cir::VecMaskedLoadOp op) {
            return access(op.getAddr(), std::nullopt, /*mod=*/false,
                          /*ref=*/true);
          })
          .Case<cir::StrLenOp>([&](cir::StrLenOp op) {
            return access(op.getString(), kUnknownSize, /*mod=*/false,
                          /*ref=*/true);
          })
          .Case<cir::ClearPaddingOp>([&](cir::ClearPaddingOp op) {
            return access(op.getArg(), std::nullopt, /*mod=*/true,
                          /*ref=*/false);
          })
          .Case<cir::CIRCallOpInterface>([&](cir::CIRCallOpInterface call) {
            return getCallModRef(call, location);
          })
          .Default([](Operation *) { return std::nullopt; });
  if (result)
    return *result;

  // SOUNDNESS: an op without a rule touches no memory only if it says so:
  // no memory effect at all, or only allocations (a fresh object is no
  // existing location). Anything else, including an op with regions whose
  // effects are recursive, may read and write anything.
  if (isMemoryEffectFree(op))
    return MRR::getNoModRef();
  if (auto effects = dyn_cast<MemoryEffectOpInterface>(op);
      effects && !op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    SmallVector<MemoryEffects::EffectInstance> list;
    effects.getEffects(list);
    if (llvm::all_of(list, [](const MemoryEffects::EffectInstance &effect) {
          return isa<MemoryEffects::Allocate>(effect.getEffect());
        }))
      return MRR::getNoModRef();
  }
  return MRR::getModAndRef();
}

void PointsToAnalysis::buildQueryIndex() const {
  if (queryIndexBuilt)
    return;
  queryIndexBuilt = true;
  for (const std::unique_ptr<TUFunction> &fn : functions) {
    functionByName[cir::FuncOp(fn->func).getSymNameAttr()] = fn.get();
    for (const TUCallSite &site : fn->callSites) {
      if (!site.site.synthetic)
        callSiteOf[site.site.call] = &site;
      else if (isa<CIRCallOpInterface>(site.site.call))
        syntheticSitesOf[site.site.call].push_back(&site);
    }
  }
  for (auto [sym, node] : globals->getSymbolNodes())
    symbolNodeOf[sym.getAttr()] = node;
}

const TUCallSite *PointsToAnalysis::findCallSite(Operation *call) const {
  buildQueryIndex();
  return callSiteOf.lookup(call);
}

const TUFunction *PointsToAnalysis::lookupFunction(StringAttr name) const {
  buildQueryIndex();
  return functionByName.lookup(name);
}

Node *PointsToAnalysis::lookupSymbolNode(StringAttr name) const {
  buildQueryIndex();
  Node *node = symbolNodeOf.lookup(name);
  return node ? Cell(node).getNode() : nullptr;
}

bool PointsToAnalysis::collectNamedSymbols(
    const TUFunction &fn, llvm::DenseSet<const TUFunction *> &visited,
    SmallVectorImpl<Cell> &out) const {
  if (!visited.insert(&fn).second)
    return true;
  // The globals the body names (get_global, #cir.global_view constants).
  // A symbol that has no node was never asked for by the local phase, so the
  // body holds no pointer to it.
  if (std::optional<SymbolTable::UseRange> uses =
          SymbolTable::getSymbolUses(&cir::FuncOp(fn.func).getBody()))
    for (const SymbolTable::SymbolUse &use : *uses)
      if (Node *node = lookupSymbolNode(use.getSymbolRef().getRootReference()))
        out.push_back(Cell(node));
  for (const TUCallSite &call : fn.callSites) {
    // An unknown callee, or a model that clobbers the globals, may call
    // back into any externally callable function of the TU, which may name
    // any global.
    if (call.site.clobbersGlobals ||
        call.resolution == CallResolution::Unknown ||
        call.resolution == CallResolution::IndirectUnresolved)
      return false;
    if (call.resolution == CallResolution::Direct) {
      if (const TUFunction *callee =
              functionMap.lookup(cir::FuncOp(call.callee).getOperation()))
        if (!collectNamedSymbols(*callee, visited, out))
          return false;
      continue;
    }
    if (call.resolution == CallResolution::Indirect)
      for (FlatSymbolRefAttr target : call.targets)
        if (const TUFunction *callee = lookupFunction(target.getAttr()))
          if (!collectNamedSymbols(*callee, visited, out))
            return false;
    // CallResolution::Model: a model touches only what its arguments reach
    // (its effect is in the graph); one that may touch globals sets
    // clobbersGlobals.
  }
  return true;
}

void PointsToAnalysis::collectCallRoots(
    const TUCallSite &call, SmallVectorImpl<Cell> &locals,
    SmallVectorImpl<Cell> &symbolsOut) const {
  // The call's own site and the calls a model makes on its behalf (a
  // synthetic site on the same op): all of them run during the call.
  buildQueryIndex();
  SmallVector<const TUCallSite *, 2> sites{&call};
  auto extra = syntheticSitesOf.find(call.site.call);
  if (extra != syntheticSitesOf.end())
    llvm::append_range(sites, extra->second);

  bool allGlobals = false;
  for (const TUCallSite *one : sites) {
    const CallSiteRecord &site = one->site;
    for (const std::optional<Cell> &arg : site.args)
      if (arg)
        locals.push_back(*arg);
    for (const auto &[index, slots] : site.aggregateArgs)
      for (auto [off, cell] : slots)
        if (cell)
          locals.push_back(cell);
    // The result: the callee may write a fresh object and return it.
    if (site.result)
      locals.push_back(*site.result);
    if (site.resultSlots)
      for (auto [off, cell] : *site.resultSlots)
        if (cell)
          locals.push_back(cell);

    // The globals the callee may name.
    allGlobals |= site.clobbersGlobals ||
                  one->resolution == CallResolution::Unknown ||
                  one->resolution == CallResolution::IndirectUnresolved;
    if (allGlobals)
      continue;
    llvm::DenseSet<const TUFunction *> visited;
    if (one->resolution == CallResolution::Direct) {
      if (const TUFunction *callee =
              functionMap.lookup(cir::FuncOp(one->callee).getOperation()))
        allGlobals = !collectNamedSymbols(*callee, visited, symbolsOut);
    } else if (one->resolution == CallResolution::Indirect) {
      for (FlatSymbolRefAttr target : one->targets)
        if (const TUFunction *callee = lookupFunction(target.getAttr()))
          if (!collectNamedSymbols(*callee, visited, symbolsOut)) {
            allGlobals = true;
            break;
          }
    }
  }
  if (allGlobals)
    for (auto [name, node] : symbolNodeOf)
      symbolsOut.push_back(Cell(node));
}

void PointsToAnalysis::closeReach(ArrayRef<Cell> roots,
                                  llvm::DenseSet<const Node *> &reach) const {
  // Through links, ghost links, subobjects (a callee given an object may
  // access its members), and from a member at offset 0 to its object
  // (pointer-interconvertibility).
  SmallVector<const Node *, 16> worklist;
  for (Cell root : roots)
    if (const Node *node = root.getNode(); reach.insert(node).second)
      worklist.push_back(node);
  while (!worklist.empty()) {
    const Node *node = worklist.pop_back_val();
    graph->forEachSuccessor(node, [&](Node *next) {
      if (reach.insert(next).second)
        worklist.push_back(next);
    });
  }
}

const llvm::DenseSet<const Node *> &
PointsToAnalysis::getCallReach(const TUCallSite &call) const {
  std::unique_ptr<llvm::DenseSet<const Node *>> &slot =
      callReach[call.site.call];
  if (slot)
    return *slot;
  slot = std::make_unique<llvm::DenseSet<const Node *>>();
  SmallVector<Cell, 8> roots;
  collectCallRoots(call, roots, roots);
  closeReach(roots, *slot);
  return *slot;
}

ModRefResult PointsToAnalysis::getCallModRef(Operation *call,
                                             Value location) const {
  ObjectSet set = pointsTo(location);
  if (set.cells.empty())
    return set.complete ? ModRefResult::getNoModRef()
                        : ModRefResult::getModAndRef();
  const TUCallSite *site = findCallSite(call);
  if (!site)
    return ModRefResult::getModAndRef();
  return callModRefOf(set.cells, getCallReach(*site));
}

ModRefResult PointsToAnalysis::callModRefOf(
    ArrayRef<Cell> location, const llvm::DenseSet<const Node *> &reach) const {
  ModRefResult result = ModRefResult::getNoModRef();
  for (Cell cell : location) {
    const Node *node = cell.getNode();
    // Subobjects: the location's bytes include those of its children, which
    // have their own nodes; the rules below hold for each of them.
    SmallVector<const Node *, 4> parts;
    graph->forEachInSubtree(node, [&](const Node *n) { parts.push_back(n); });
    bool reached = false;
    NodeFlags flags = NodeFlags::None;
    for (const Node *part : parts) {
      // SOUNDNESS: code we don't see may run during any call (an unknown or
      // unresolved callee, or one further down), and it may write any object
      // it can reach. That is every External node, and also every
      // Incomplete one: an unknown caller may have kept the pointer it passed
      // us where that code finds it. Their Mod and Ref flags record only the
      // analysed code's accesses (status propagation spreads External, not
      // Mod), so they don't refine the answer either.
      // SOUNDNESS: a USM allocation is also accessed by device code, which
      // runs asynchronously and is synchronized with by calls (a wait, an
      // event, the next command) whose arguments need not reach the
      // allocation. So is the memory an asynchronous command reads or writes
      // (AsyncAccess).
      // MayBeShared (BU): the node may be a global's object, which the call
      // may reach by the global's name.
      if (isUnknownReachable(part) || part->hasFlags(NodeFlags::HasUSM) ||
          part->hasFlags(NodeFlags::AsyncAccess) ||
          (part->hasFlags(NodeFlags::MayBeShared) && !graph->isShared(part)))
        return ModRefResult::getModAndRef();
      // SOUNDNESS: a call (and everything it calls) reaches an existing
      // object only through a pointer: one it is passed, one it loads from
      // memory it reaches, or one it takes from a global by name. The binding
      // unified the callee's formals with these arguments, so the pointers
      // the callee derives from them are in nodes reachable from the
      // arguments. A fresh object the call allocates is reached through its
      // result, or through an argument or a global it is stored into.
      reached |= reach.contains(part);
      flags |= part->getFlags();
    }
    if (!reached)
      continue;
    // SOUNDNESS: the nodes are complete, so all code that accesses them is
    // analysed, and every analysed access set Mod or Ref on them (merges OR
    // the flags). A node never written anywhere is not written by this call.
    if (hasAny(flags, NodeFlags::Mod))
      result = result.merge(ModRefResult::getMod());
    if (hasAny(flags, NodeFlags::Ref))
      result = result.merge(ModRefResult::getRef());
  }
  return result;
}
