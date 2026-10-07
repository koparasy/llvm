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
         !hasAny(node->getFlags(), NodeFlags::Summary) &&
         !isUnknownReachable(node);
}

/// Whether \p value is a null pointer, or is computed from one by the steps
/// decomposePointer looks through (a cast, a member or a constant stride).
static bool isDerivedFromNull(Value value, const DataLayout &dataLayout) {
  if (isNullPointerConstant(value))
    return true;
  PointerDecomposition decomposed = decomposePointer(value, dataLayout);
  return decomposed.base && isNullPointerConstant(decomposed.base);
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
  // The same SSA value is the same address (the MLIR convention, which
  // LocalAliasAnalysis and CIRBasicAliasAnalysis follow too).
  if (a == b)
    return {AliasResult::MustAlias, AliasRule::SameValue};

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

AliasAnswer PointsToAnalysis::aliasCells(Cell cellA, Cell cellB, uint64_t sizeA,
                                         uint64_t sizeB) const {
  cellA = cellA.resolve();
  cellB = cellB.resolve();
  const Node *nodeA = cellA.getNode();
  const Node *nodeB = cellB.getNode();
  bool unknownA = isUnknownReachable(nodeA);
  bool unknownB = isUnknownReachable(nodeB);

  // Rule 2: different nodes.
  if (nodeA != nodeB) {
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

  bool singleton = isSingleton(nodeA);
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
  using MRR = ModRefResult;
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

const llvm::DenseSet<const Node *> &
PointsToAnalysis::getCallReach(const TUCallSite &call) const {
  std::unique_ptr<llvm::DenseSet<const Node *>> &slot =
      callReach[call.site.call];
  if (slot)
    return *slot;
  slot = std::make_unique<llvm::DenseSet<const Node *>>();
  llvm::DenseSet<const Node *> &reach = *slot;

  // The call's own site and the calls a model makes on its behalf (a
  // synthetic site on the same op): all of them run during the call.
  buildQueryIndex();
  SmallVector<const TUCallSite *, 2> sites{&call};
  auto extra = syntheticSitesOf.find(call.site.call);
  if (extra != syntheticSitesOf.end())
    llvm::append_range(sites, extra->second);

  SmallVector<Cell, 8> roots;
  bool allGlobals = false;
  for (const TUCallSite *one : sites) {
    const CallSiteRecord &site = one->site;
    for (const std::optional<Cell> &arg : site.args)
      if (arg)
        roots.push_back(*arg);
    for (const auto &[index, slots] : site.aggregateArgs)
      for (auto [off, cell] : slots)
        if (cell)
          roots.push_back(cell);
    // The result: the callee may write a fresh object and return it.
    if (site.result)
      roots.push_back(*site.result);
    if (site.resultSlots)
      for (auto [off, cell] : *site.resultSlots)
        if (cell)
          roots.push_back(cell);

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
        allGlobals = !collectNamedSymbols(*callee, visited, roots);
    } else if (one->resolution == CallResolution::Indirect) {
      for (FlatSymbolRefAttr target : one->targets)
        if (const TUFunction *callee = lookupFunction(target.getAttr()))
          if (!collectNamedSymbols(*callee, visited, roots)) {
            allGlobals = true;
            break;
          }
    }
  }
  if (allGlobals) {
    buildQueryIndex();
    for (auto [name, node] : symbolNodeOf)
      roots.push_back(Cell(node));
  }

  SmallVector<const Node *, 16> worklist;
  for (Cell root : roots)
    if (const Node *node = root.getNode(); reach.insert(node).second)
      worklist.push_back(node);
  while (!worklist.empty()) {
    const Node *node = worklist.pop_back_val();
    for (auto [off, target] : node->getLinks())
      if (const Node *next = target.getNode(); reach.insert(next).second)
        worklist.push_back(next);
    for (const GhostLink &ghost : node->getGhostLinks())
      if (const Node *next = ghost.target.getNode(); reach.insert(next).second)
        worklist.push_back(next);
  }
  return reach;
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
  const Node *node = set.cells.front().getNode();
  // SOUNDNESS: code we don't see may run during any call (an unknown or
  // unresolved callee, or one further down), and it may write any object it
  // can reach. That is every External node, and also every Incomplete one:
  // an unknown caller may have kept the pointer it passed us where that
  // code finds it. Their Mod and Ref flags record only the analysed code's
  // accesses (status propagation spreads External, not Mod), so they don't
  // refine the answer either.
  if (isUnknownReachable(node))
    return ModRefResult::getModAndRef();
  // SOUNDNESS: a USM allocation is also accessed by device code, which runs
  // asynchronously and is synchronized with by calls (a wait, an event, the
  // next command) whose arguments need not reach the allocation. So is the
  // memory an asynchronous command reads or writes (AsyncAccess).
  if (node->hasFlags(NodeFlags::HasUSM) ||
      node->hasFlags(NodeFlags::AsyncAccess))
    return ModRefResult::getModAndRef();
  // SOUNDNESS: a call (and everything it calls) reaches an existing object
  // only through a pointer: one it is passed, one it loads from memory it
  // reaches, or one it takes from a global by name. Context-insensitive
  // binding unified the callee's formals with these arguments, so the
  // pointers the callee derives from them are in nodes reachable from the
  // arguments. A fresh object the call allocates is reached through its
  // result, or through an argument or a global it is stored into.
  if (!getCallReach(*site).contains(node))
    return ModRefResult::getNoModRef();
  // SOUNDNESS: the node is complete, so all code that accesses it is
  // analysed, and every analysed access set Mod or Ref on it (merges OR the
  // flags). A node never written anywhere is not written by this call.
  bool mod = node->hasFlags(NodeFlags::Mod);
  bool ref = node->hasFlags(NodeFlags::Ref);
  if (mod && ref)
    return ModRefResult::getModAndRef();
  if (mod)
    return ModRefResult::getMod();
  if (ref)
    return ModRefResult::getRef();
  return ModRefResult::getNoModRef();
}
