//===- PointsToGraph.cpp - DSA-style points-to memory graph ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unification and the rules that make a node field-insensitive (collapse it):
//
//  - Misaligned links. A pointer slot is pointerSize bytes. Two links whose
//    offsets differ but are less than pointerSize apart overlap, so the node
//    can't be described by byte offsets any more. On an array node the last
//    link and the first link of the next element are also compared, so a
//    pointer slot can't straddle an element boundary, and an array whose
//    element is smaller than a pointer collapses as soon as it holds one.
//  - Self-unification at a delta. unify((n, a), (n, b)) with a != b (after
//    normalization) says the object contains itself at a shift: a recursive
//    or array structure. It collapses. On an array node, a and b are compared
//    modulo the element size, so the matching-stride case is a no-op.
//  - Arrays of different element size. Merging two array nodes with different
//    element sizes, or calling markArray with a size other than the node's,
//    collapses. markArray with element size 0 (unknown) collapses too.
//  - Array with non-array. The non-array side is taken modulo the array's
//    element size; links that then collide at the same offset are unified,
//    links that then overlap trigger the misalignment rule.
//  - Collapsed with anything. The result is collapsed.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToGraph.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstring>

using namespace llvm;
using namespace cir::pointsto;

//===----------------------------------------------------------------------===//
// NodeFlags
//===----------------------------------------------------------------------===//

static constexpr unsigned numFlagBits = 13;

StringRef cir::pointsto::getFlagName(NodeFlags bit) {
  switch (bit) {
  case NodeFlags::Stack:
    return "Stack";
  case NodeFlags::Heap:
    return "Heap";
  case NodeFlags::Global:
    return "Global";
  case NodeFlags::Ghost:
    return "Ghost";
  case NodeFlags::Function:
    return "Function";
  case NodeFlags::UnknownObj:
    return "UnknownObj";
  case NodeFlags::Incomplete:
    return "Incomplete";
  case NodeFlags::External:
    return "External";
  case NodeFlags::Collapsed:
    return "Collapsed";
  case NodeFlags::Summary:
    return "Summary";
  case NodeFlags::Mod:
    return "Mod";
  case NodeFlags::Ref:
    return "Ref";
  case NodeFlags::HasUSM:
    return "HasUSM";
  default:
    return "?";
  }
}

std::string cir::pointsto::toString(NodeFlags flags) {
  std::string result;
  for (unsigned i = 0; i < numFlagBits; ++i) {
    auto bit = static_cast<NodeFlags>(1u << i);
    if (!hasAny(flags, bit))
      continue;
    if (!result.empty())
      result += '|';
    result += getFlagName(bit);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Reason
//===----------------------------------------------------------------------===//

StringRef cir::pointsto::getReasonKindName(Reason::Kind kind) {
  switch (kind) {
  case Reason::Kind::MergedAt:
    return "MergedAt";
  case Reason::Kind::EscapedTo:
    return "EscapedTo";
  case Reason::Kind::UnknownCaller:
    return "UnknownCaller";
  case Reason::Kind::CollapsedAt:
    return "CollapsedAt";
  case Reason::Kind::Budget:
    return "Budget";
  case Reason::Kind::UnmodelledRole:
    return "UnmodelledRole";
  case Reason::Kind::IndirectUnresolved:
    return "IndirectUnresolved";
  case Reason::Kind::SummaryObject:
    return "SummaryObject";
  case Reason::Kind::UnhandledOp:
    return "UnhandledOp";
  }
  llvm_unreachable("unknown reason kind");
}

/// The details of the reasons Graph::markReachable derives.
static constexpr StringLiteral kReachableFromEscaped =
    "reachable from an escaped pointer";
static constexpr StringLiteral kReachableFromIncomplete =
    "reachable from an incomplete node";

bool cir::pointsto::isDerivedReason(const Reason *why) {
  return why && why->cause &&
         (why->detail == kReachableFromEscaped ||
          why->detail == kReachableFromIncomplete);
}

const Reason *cir::pointsto::getRootReason(const Reason *why) {
  while (isDerivedReason(why))
    why = why->cause;
  return why;
}

/// The name of the symbol \p op defines, or of the nearest enclosing op that
/// defines one (the function an op is in), as "@name"; empty if none.
static std::string getEnclosingSymbol(mlir::Operation *op) {
  for (; op; op = op->getParentOp())
    if (auto name = op->getAttrOfType<mlir::StringAttr>(
            mlir::SymbolTable::getSymbolAttrName()))
      return ("@" + name.getValue()).str();
  return {};
}

/// " (file:line)" for \p op, " (in @function)" if it has no file location,
/// or nothing.
static std::string getLocationSuffix(mlir::Operation *op) {
  if (!op)
    return {};
  if (auto loc = op->getLoc()->findInstanceOf<mlir::FileLineColLoc>())
    return (" (" + sys::path::filename(loc.getFilename().getValue()) + ":" +
            Twine(loc.getLine()) + ")")
        .str();
  std::string sym = getEnclosingSymbol(op);
  return sym.empty() ? std::string() : " (in " + sym + ")";
}

/// Whether \p detail is a single symbol reference, "@name".
static bool isSymbolDetail(StringRef detail) {
  return detail.size() > 1 && detail.front() == '@' && !detail.contains(' ');
}

/// How a pointer escaped, for the detail of an EscapedTo reason: a phrase
/// that follows "an escaped pointer".
static std::string describeEscape(StringRef detail) {
  if (isSymbolDetail(detail))
    return ("passed to " + detail).str();
  if (StringRef callee = detail;
      callee.consume_back(" not analysed") && isSymbolDetail(callee))
    return ("passed to " + callee + ", whose body is not analysed").str();
  if (detail.consume_back(" may access the global"))
    return detail == "an indirect callee"
               ? std::string("stored in a global that an unresolved indirect "
                             "callee may access")
               : ("stored in a global that " + detail + " may access").str();
  if (detail.consume_front("result of "))
    return ("returned by " + detail).str();
  if (detail.starts_with("kept by ") ||
      detail.starts_with("may be returned by ") ||
      detail.starts_with("written by "))
    return detail.str();
  return StringSwitch<std::string>(detail)
      .Case("global visible to other TUs",
            "stored in a global visible to other translation units")
      .Case("variadic argument", "passed as a variadic argument")
      .Case("address escaped as an integer", "converted to an integer")
      .Case("pointer made from an integer", "made from an integer")
      .Case("absolute address", "made from an absolute address")
      .Case("symbol not found", "naming a symbol that does not exist")
      .Case("alias chain too deep", "naming an alias chain that is too deep")
      .Case("search of a null pointer", "returned by a search of a null "
                                        "pointer")
      .Default(("(\"" + detail + "\")").str());
}

/// A noun phrase for the root reason \p why, without its location.
static std::string describeRoot(const Reason &why) {
  StringRef detail = why.detail;
  auto quoted = [&](StringRef prefix) {
    return detail.empty() ? prefix.str()
                          : (prefix + " (\"" + detail + "\")").str();
  };
  std::string opName =
      why.op ? ("`" + why.op->getName().getStringRef() + "`").str()
             : std::string("an operation");
  std::string sym = getEnclosingSymbol(why.op);
  if (sym.empty())
    sym = "a function";
  switch (why.kind) {
  case Reason::Kind::EscapedTo:
    return "an escaped pointer " + describeEscape(detail);
  case Reason::Kind::UnknownCaller:
    if (detail.consume_front("formal #"))
      return ("parameter #" + detail + " of " + sym +
              ", which may have callers that are not analysed")
          .str();
    if (detail == "returned to an unknown caller")
      return "the return value of " + sym +
             ", which may have callers that are not analysed";
    if (detail == "dynamic initialization not analysed")
      return "the global " + sym +
             ", whose dynamic initialization is not analysed";
    return quoted("a value that a caller that is not analysed may pass");
  case Reason::Kind::IndirectUnresolved:
    return quoted("a pointer passed to or returned from an indirect call "
                  "whose targets are not known");
  case Reason::Kind::Budget:
    return quoted("a pointer the analysis stopped tracking when it ran out "
                  "of budget");
  case Reason::Kind::UnmodelledRole:
    return quoted("a pointer passed to a runtime call that has no model");
  case Reason::Kind::UnhandledOp:
    if (!why.op)
      return quoted("a value the analysis did not see");
    return quoted("a pointer through " + opName +
                  ", which the analysis does not model");
  case Reason::Kind::CollapsedAt:
    return quoted("an object made field-insensitive at " + opName);
  case Reason::Kind::MergedAt:
    if (isSymbolDetail(detail))
      return ("an object merged with others by binding a call to " + detail)
          .str();
    if (detail == "one node per global symbol")
      return "an object merged with the other copies of a global";
    return quoted("an object merged with others at " + opName);
  case Reason::Kind::SummaryObject:
    return StringSwitch<std::string>(detail)
        .Case("loop", "an object allocated in a loop")
        .Case("recursion", "an object of a recursive function")
        .Case("several callers",
              "an object allocated in a function that may run several times")
        .Case("merged origins", "an object with several origins")
        .Default(quoted("an object that stands for several objects"));
  }
  llvm_unreachable("unknown reason kind");
}

std::string cir::pointsto::explain(const Reason *why) {
  if (!why)
    return "no reason was recorded";
  const Reason *root = getRootReason(why);
  return (root == why ? "it is " : "it is reachable from ") +
         describeRoot(*root) + getLocationSuffix(root->op);
}

const Reason *ReasonTable::make(Reason::Kind kind, mlir::Operation *op,
                                StringRef detail, const Reason *cause) {
  StringRef ownedDetail;
  if (!detail.empty()) {
    char *buf = allocator.Allocate<char>(detail.size());
    std::memcpy(buf, detail.data(), detail.size());
    ownedDetail = StringRef(buf, detail.size());
  }
  return new (allocator.Allocate<Reason>())
      Reason{kind, op, ownedDetail, cause};
}

//===----------------------------------------------------------------------===//
// Cell and Node
//===----------------------------------------------------------------------===//

/// Normalize \p offset on the live node \p node: 0 if collapsed, modulo the
/// element size on an array.
static int64_t normalizeOffset(const Node *node, int64_t offset) {
  if (node->isCollapsed())
    return 0;
  if (std::optional<uint64_t> elem = node->getArrayElemSize()) {
    auto e = static_cast<int64_t>(*elem);
    int64_t r = offset % e;
    return r < 0 ? r + e : r;
  }
  return offset;
}

Cell Cell::resolve() const {
  if (!node)
    return *this;
  Node *root = node;
  int64_t off = offset;
  if (root->forward) {
    // Find the live node, then point every node on the chain straight at it
    // with its accumulated delta (path compression).
    SmallVector<Node *, 8> chain;
    while (root->forward) {
      chain.push_back(root);
      root = root->forward;
    }
    int64_t acc = 0;
    for (Node *n : llvm::reverse(chain)) {
      acc += n->forwardDelta;
      n->forwardDelta = acc;
      n->forward = root;
    }
    off += node->forwardDelta;
  }
  return Cell(root, normalizeOffset(root, off));
}

SmallVector<std::pair<int64_t, Cell>, 4> Node::getLinks() const {
  SmallVector<std::pair<int64_t, Cell>, 4> result;
  result.reserve(links.size());
  for (const auto &[off, target] : links)
    result.emplace_back(off, target.resolve());
  return result;
}

const Reason *Node::getReason(NodeFlags flag) const {
  for (const auto &[bit, why] : reasons)
    if (bit == flag)
      return why;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// CloneMap
//===----------------------------------------------------------------------===//

Cell CloneMap::lookup(const Node *node) const {
  if (!node)
    return Cell();
  Node *live = Cell(const_cast<Node *>(node)).resolve().getRawNode();
  auto it = nodeMap.find(live);
  if (it == nodeMap.end())
    return Cell();
  return it->second.resolve();
}

Cell CloneMap::map(Cell srcCell) const {
  Cell src = srcCell.resolve();
  if (!src)
    return Cell();
  auto it = nodeMap.find(src.getRawNode());
  if (it == nodeMap.end())
    return Cell();
  Cell dst = it->second;
  return Cell(dst.getRawNode(), dst.getRawOffset() + src.getRawOffset())
      .resolve();
}

//===----------------------------------------------------------------------===//
// Graph: construction and simple updates
//===----------------------------------------------------------------------===//

Graph::Graph(unsigned pointerSize, ReasonTable *reasons)
    : pointerSize(pointerSize) {
  assert(pointerSize > 0 && "pointer size must be positive");
  if (!reasons) {
    ownedReasons = std::make_unique<ReasonTable>();
    reasons = ownedReasons.get();
  }
  reasonTable = reasons;
}

Graph::~Graph() = default;

Node *Graph::newNode() {
  auto id = static_cast<uint32_t>(allNodes.size());
  Node *node = new (nodeAllocator.Allocate()) Node(id);
  allNodes.push_back(node);
  return node;
}

Node *Graph::makeNode(NodeFlags flags) {
  Node *node = newNode();
  if (hasAny(flags, NodeFlags::Collapsed)) {
    collapse(node);
    flags &= ~NodeFlags::Collapsed;
  }
  node->flags |= flags;
  return node;
}

Node *Graph::makeNode(const Origin &origin, NodeFlags flags) {
  Node *node = makeNode(flags);
  node->origins.push_back(origin);
  return node;
}

void Graph::recordReason(Node *node, NodeFlags bit, const Reason *why) {
  if (!why)
    return;
  auto *it = llvm::lower_bound(node->reasons, bit,
                               [](const std::pair<NodeFlags, const Reason *> &e,
                                  NodeFlags b) { return e.first < b; });
  if (it != node->reasons.end() && it->first == bit)
    return; // The first reason wins.
  node->reasons.insert(it, {bit, why});
}

void Graph::addFlags(Cell cell, NodeFlags flags, const Reason *why) {
  Node *node = cell.getNode();
  assert(node && "addFlags on an empty cell");
  if (hasAny(flags, NodeFlags::Collapsed)) {
    collapse(node, why);
    node = cell.getNode();
  }
  node->flags |= flags;
  for (unsigned i = 0; i < numFlagBits; ++i) {
    auto bit = static_cast<NodeFlags>(1u << i);
    if (hasAny(flags & NodeFlags::StatusMask, bit))
      recordReason(node, bit, why);
  }
}

size_t Graph::markReachable(Cell from, NodeFlags flags, const Reason *why,
                            bool stopAtFlagged) {
  assert(!hasAny(flags, NodeFlags::Collapsed) &&
         "markReachable must not collapse nodes");
  Node *root = from.getNode();
  if (!root)
    return 0;
  const Reason *derived = nullptr;
  llvm::SmallPtrSet<Node *, 16> visited;
  llvm::SmallVector<Node *, 16> worklist{root};
  visited.insert(root);
  while (!worklist.empty()) {
    Node *node = worklist.pop_back_val();
    // Already flagged: whatever it reaches is the job of the call made for
    // it.
    if (stopAtFlagged && node != root && node->hasFlags(flags))
      continue;
    const Reason *r = why;
    if (node != root && why) {
      if (!derived)
        derived =
            makeReason(why->kind, why->op,
                       flags == NodeFlags::Incomplete ? kReachableFromIncomplete
                                                      : kReachableFromEscaped,
                       why);
      r = derived;
    }
    addFlags(Cell(node), flags, r);
    for (const auto &link : node->links) {
      Node *target = link.second.getNode();
      if (target && visited.insert(target).second)
        worklist.push_back(target);
    }
  }
  return visited.size();
}

void Graph::addOrigin(Cell cell, const Origin &origin) {
  Node *node = cell.getNode();
  assert(node && "addOrigin on an empty cell");
  if (!llvm::is_contained(node->origins, origin))
    node->origins.push_back(origin);
}

void Graph::growSize(Node *node, uint64_t size) {
  node = Cell(node).getNode();
  node->size = std::max(node->size, size);
}

size_t Graph::getNumLiveNodes() const {
  return llvm::count_if(allNodes,
                        [](const Node *n) { return !n->isForwarded(); });
}

//===----------------------------------------------------------------------===//
// Graph: links
//===----------------------------------------------------------------------===//

static auto findLink(SmallVectorImpl<std::pair<int64_t, Cell>> &links,
                     int64_t offset) {
  return llvm::lower_bound(
      links, offset,
      [](const std::pair<int64_t, Cell> &e, int64_t o) { return e.first < o; });
}

void Graph::insertLink(Node *node, int64_t offset, Cell target,
                       Worklist &worklist) {
  offset = normalizeOffset(node, offset);
  auto *it = findLink(node->links, offset);
  if (it != node->links.end() && it->first == offset) {
    // The slot already points somewhere: both targets are the same cell now.
    worklist.emplace_back(it->second, target);
    return;
  }
  node->links.insert(it, {offset, target});
}

bool Graph::isMisaligned(const Node *node) const {
  const auto &links = node->links;
  if (node->isCollapsed() || links.empty())
    return false;
  for (size_t i = 1; i < links.size(); ++i)
    if (links[i].first - links[i - 1].first < int64_t(pointerSize))
      return true;
  // On an array, the last slot of one element must not run into the first
  // slot of the next one.
  if (std::optional<uint64_t> elem = node->arrayElemSize)
    if (links.front().first + int64_t(*elem) - links.back().first <
        int64_t(pointerSize))
      return true;
  return false;
}

void Graph::collapseIfMisaligned(Node *node, const Reason *why,
                                 Worklist &worklist) {
  if (isMisaligned(node))
    collapseImpl(node, why, worklist);
}

Cell Graph::getLink(Cell from, const Reason *why) {
  if (std::optional<Cell> existing = lookupLink(from))
    return *existing;
  Cell src = from.resolve();
  assert(src && "getLink on an empty cell");
  Cell target(newNode(), 0);
  Worklist worklist;
  insertLink(src.getRawNode(), src.getRawOffset(), target, worklist);
  collapseIfMisaligned(src.getRawNode(), why, worklist);
  runWorklist(worklist, why);
  return target.resolve();
}

std::optional<Cell> Graph::lookupLink(Cell from) const {
  Cell src = from.resolve();
  if (!src)
    return std::nullopt;
  auto &links = src.getRawNode()->links;
  auto *it = findLink(links, src.getRawOffset());
  if (it == links.end() || it->first != src.getRawOffset())
    return std::nullopt;
  return it->second.resolve();
}

void Graph::setLink(Cell from, Cell to, const Reason *why) {
  Cell src = from.resolve();
  assert(src && to && "setLink on an empty cell");
  Worklist worklist;
  insertLink(src.getRawNode(), src.getRawOffset(), to, worklist);
  collapseIfMisaligned(src.getRawNode(), why, worklist);
  runWorklist(worklist, why);
}

//===----------------------------------------------------------------------===//
// Graph: unification, collapse, arrays
//===----------------------------------------------------------------------===//

void Graph::unify(Cell a, Cell b, const Reason *why) {
  Worklist worklist;
  worklist.emplace_back(a, b);
  runWorklist(worklist, why);
}

void Graph::runWorklist(Worklist &worklist, const Reason *why) {
  while (!worklist.empty()) {
    auto [a, b] = worklist.pop_back_val();
    Cell ra = a.resolve();
    Cell rb = b.resolve();
    assert(ra && rb && "unify on an empty cell");
    Node *na = ra.getRawNode();
    Node *nb = rb.getRawNode();
    int64_t oa = ra.getRawOffset();
    int64_t ob = rb.getRawOffset();

    if (na == nb) {
      // The object contains itself at a shift: recursive or array structure.
      if (oa != ob)
        collapseImpl(na, why, worklist);
      continue;
    }

    // The node at the larger offset survives, so the shift is non-negative.
    // On a tie the older node survives.
    if (oa > ob || (oa == ob && na->id < nb->id))
      merge(na, nb, oa - ob, worklist, why);
    else
      merge(nb, na, ob - oa, worklist, why);
  }
}

void Graph::merge(Node *survivor, Node *merged, int64_t delta,
                  Worklist &worklist, const Reason *why) {
  assert(survivor != merged && !survivor->isForwarded() &&
         !merged->isForwarded() && "merge of non-live or identical nodes");

  bool collapseResult = survivor->isCollapsed() || merged->isCollapsed() ||
                        (survivor->arrayElemSize && merged->arrayElemSize &&
                         *survivor->arrayElemSize != *merged->arrayElemSize);
  std::optional<uint64_t> elem =
      survivor->arrayElemSize ? survivor->arrayElemSize : merged->arrayElemSize;

  // Move the state out of the merged node and forward it. From here on every
  // cell on `merged` resolves to `survivor`.
  auto mergedLinks = std::move(merged->links);
  auto mergedOrigins = std::move(merged->origins);
  auto mergedReasons = std::move(merged->reasons);
  NodeFlags mergedFlags = merged->flags;
  uint64_t mergedSize = merged->size;
  merged->links.clear();
  merged->origins.clear();
  merged->reasons.clear();
  merged->flags = NodeFlags::None;
  merged->size = 0;
  merged->arrayElemSize.reset();
  merged->forward = survivor;
  merged->forwardDelta = delta;

  // Collapsed is applied by collapseImpl below, which also moves the links.
  survivor->flags |= mergedFlags & ~NodeFlags::Collapsed;
  for (const Origin &origin : mergedOrigins)
    if (!llvm::is_contained(survivor->origins, origin))
      survivor->origins.push_back(origin);
  // The survivor's reasons win; the merged node fills in the missing bits.
  for (const auto &[bit, r] : mergedReasons)
    recordReason(survivor, bit, r);
  if (mergedSize) {
    int64_t end = static_cast<int64_t>(mergedSize) + delta;
    if (end > 0)
      survivor->size = std::max(survivor->size, static_cast<uint64_t>(end));
  }

  if (collapseResult)
    collapseImpl(survivor, why, worklist);
  else if (elem && !survivor->arrayElemSize)
    markArrayImpl(survivor, *elem, why, worklist);

  for (const auto &[off, target] : mergedLinks)
    insertLink(survivor, off + delta, target, worklist);
  collapseIfMisaligned(survivor, why, worklist);
}

void Graph::collapseImpl(Node *node, const Reason *why, Worklist &worklist) {

  if (node->isCollapsed())
    return;
  node->flags |= NodeFlags::Collapsed;
  recordReason(node, NodeFlags::Collapsed, why);
  node->arrayElemSize.reset();
  if (node->links.empty())
    return;
  // Every slot is the slot at 0 now: unify all targets into the first one.
  Cell first = node->links.front().second;
  for (const auto &link : llvm::drop_begin(node->links))
    worklist.emplace_back(first, link.second);
  node->links.clear();
  node->links.emplace_back(0, first);
}

void Graph::collapse(Node *node, const Reason *why) {
  node = Cell(node).getNode();
  assert(node && "collapse of a null node");
  Worklist worklist;
  collapseImpl(node, why, worklist);
  runWorklist(worklist, why);
}

void Graph::markArrayImpl(Node *node, uint64_t elemSize, const Reason *why,
                          Worklist &worklist) {
  if (node->isCollapsed())
    return;
  if (elemSize == 0 ||
      (node->arrayElemSize && *node->arrayElemSize != elemSize)) {
    collapseImpl(node, why, worklist);
    return;
  }
  if (node->arrayElemSize)
    return;
  node->arrayElemSize = elemSize;
  // Re-insert the links at their offsets modulo the element size.
  auto oldLinks = std::move(node->links);
  node->links.clear();
  for (const auto &[off, target] : oldLinks)
    insertLink(node, off, target, worklist);
  collapseIfMisaligned(node, why, worklist);
}

void Graph::markArray(Node *node, uint64_t elemSize, const Reason *why) {
  node = Cell(node).getNode();
  assert(node && "markArray of a null node");
  Worklist worklist;
  markArrayImpl(node, elemSize, why, worklist);
  runWorklist(worklist, why);
}

//===----------------------------------------------------------------------===//
// Graph: cloning
//===----------------------------------------------------------------------===//

CloneMap Graph::cloneInto(Graph &dst, ArrayRef<Cell> roots,
                          NodeFlags keepMask) const {
  // Collapsed describes the shape of the links; it is always kept.
  keepMask |= NodeFlags::Collapsed;
  bool sharedReasons = &dst.getReasonTable() == reasonTable;
  DenseMap<const Reason *, const Reason *> reasonMap;
  auto copyReason = [&](const Reason *r) -> const Reason * {
    if (!r || sharedReasons)
      return r;
    // Copy the not yet copied prefix of the cause chain, innermost first.
    SmallVector<const Reason *, 4> chain;
    for (const Reason *c = r; c && !reasonMap.count(c); c = c->cause)
      chain.push_back(c);
    for (const Reason *c : llvm::reverse(chain))
      reasonMap[c] = dst.reasonTable->make(c->kind, c->op, c->detail,
                                           c->cause ? reasonMap.lookup(c->cause)
                                                    : nullptr);
    return reasonMap.lookup(r);
  };

  CloneMap result;
  SmallVector<Node *, 16> order;
  auto visit = [&](Cell cell) {
    Node *src = cell.getNode();
    if (!src || result.nodeMap.count(src))
      return;
    Node *copy = dst.newNode();
    copy->flags = src->flags & keepMask;
    copy->size = src->size;
    copy->arrayElemSize = src->arrayElemSize;
    copy->origins = src->origins;
    for (const auto &[bit, r] : src->reasons)
      if (hasAny(keepMask, bit))
        copy->reasons.emplace_back(bit, copyReason(r));
    result.nodeMap[src] = Cell(copy, 0);
    order.push_back(src);
  };

  // Breadth-first over the links, so the clone ids follow the roots and the
  // link order deterministically.
  for (Cell root : roots)
    visit(root);
  for (size_t i = 0; i < order.size(); ++i)
    for (const auto &link : order[i]->links)
      visit(link.second);

  // The copies have the same collapse and array state as their sources, so
  // the (already normalized and sorted) links carry over unchanged.
  for (Node *src : order) {
    Node *copy = result.nodeMap[src].getRawNode();
    for (const auto &[off, target] : src->links) {
      Cell t = target.resolve();
      copy->links.emplace_back(
          off,
          Cell(result.nodeMap[t.getRawNode()].getRawNode(), t.getRawOffset()));
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Graph: printing
//===----------------------------------------------------------------------===//

/// Print \p op as its name, its "name" attribute, and its file location.
static void printOp(raw_ostream &os, mlir::Operation *op) {
  if (!op) {
    os << '?';
    return;
  }
  os << op->getName().getStringRef();
  if (auto name = op->getAttrOfType<mlir::StringAttr>("name"))
    os << " '" << name.getValue() << "'";
  if (auto loc = op->getLoc()->findInstanceOf<mlir::FileLineColLoc>())
    os << ' ' << sys::path::filename(loc.getFilename().getValue()) << ':'
       << loc.getLine() << ':' << loc.getColumn();
}

static StringRef getOriginKindName(Origin::Kind kind) {
  switch (kind) {
  case Origin::Kind::Alloca:
    return "stack";
  case Origin::Kind::GlobalSym:
    return "global";
  case Origin::Kind::HeapCall:
    return "heap";
  case Origin::Kind::GhostSite:
    return "ghost";
  case Origin::Kind::FuncSym:
    return "func";
  }
  llvm_unreachable("unknown origin kind");
}

static void printOrigin(raw_ostream &os, const Origin &origin) {
  os << getOriginKindName(origin.kind) << '(';
  if (origin.sym) {
    origin.sym.print(os);
    if (origin.site)
      os << ", ";
  }
  if (origin.site || !origin.sym)
    printOp(os, origin.site);
  os << ')';
  if (origin.ctx)
    os << '#' << origin.ctx;
}

/// Print \p r alone, without its causes.
static void printOneReason(raw_ostream &os, const Reason *r) {
  os << getReasonKindName(r->kind) << '(';
  printOp(os, r->op);
  if (!r->detail.empty()) {
    os << ", \"";
    printEscapedString(r->detail, os);
    os << '"';
  }
  os << ')';
}

static void printReason(raw_ostream &os, const Reason *r) {
  constexpr unsigned maxCauses = 4;
  for (unsigned depth = 0; r; r = r->cause, ++depth) {
    if (depth > maxCauses) {
      os << " <- ...";
      return;
    }
    if (depth)
      os << " <- ";
    printOneReason(os, r);
  }
}

SmallVector<std::string, 4> cir::pointsto::explainChain(const Reason *why) {
  SmallVector<std::string, 4> chain;
  for (; why; why = why->cause) {
    std::string line;
    raw_string_ostream os(line);
    printOneReason(os, why);
    chain.push_back(std::move(line));
  }
  return chain;
}

void Graph::printCell(raw_ostream &os, Cell cell) {
  Cell c = cell.resolve();
  if (!c) {
    os << "null";
    return;
  }
  os << 'n' << c.getRawNode()->getId();
  if (c.getRawOffset() >= 0)
    os << '+';
  os << c.getRawOffset();
}

void Graph::print(raw_ostream &os) const {
  for (const Node *node : nodes()) {
    os << 'n' << node->id << " [" << toString(node->flags)
       << "] size=" << node->size;
    if (node->arrayElemSize)
      os << " arr=" << *node->arrayElemSize;
    os << " origins={";
    llvm::interleave(
        node->origins, os, [&](const Origin &o) { printOrigin(os, o); }, ", ");
    os << "} links={";
    llvm::interleave(
        node->links, os,
        [&](const std::pair<int64_t, Cell> &link) {
          os << link.first << "->";
          printCell(os, link.second);
        },
        ", ");
    os << '}';
    if (!node->reasons.empty()) {
      os << " reasons={";
      llvm::interleave(
          node->reasons, os,
          [&](const std::pair<NodeFlags, const Reason *> &e) {
            os << getFlagName(e.first) << ": ";
            printReason(os, e.second);
          },
          ", ");
      os << '}';
    }
    os << '\n';
  }
}

LLVM_DUMP_METHOD void Graph::dump() const { print(llvm::errs()); }

//===----------------------------------------------------------------------===//
// Graph: verification
//===----------------------------------------------------------------------===//

bool Graph::verify(raw_ostream *errs) const {
  bool ok = true;
  auto fail = [&](const Node *node, const Twine &msg) {
    ok = false;
    if (errs)
      *errs << "points-to graph: n" << node->id << ": " << msg << '\n';
  };

  for (const Node *node : allNodes) {
    if (node->isForwarded()) {
      if (!node->links.empty() || !node->origins.empty() ||
          !node->reasons.empty() || node->flags != NodeFlags::None)
        fail(node, "forwarded node still holds state");
      // A cycle in the forwarding chain would make resolve() loop.
      const Node *n = node;
      size_t steps = 0;
      while (n->forward && steps <= allNodes.size()) {
        n = n->forward;
        ++steps;
      }
      if (n->forward)
        fail(node, "forwarding cycle");
      continue;
    }

    const auto &links = node->links;
    for (size_t i = 0; i < links.size(); ++i) {
      if (!links[i].second)
        fail(node, "null link target");
      else if (links[i].second.getNode()->isForwarded())
        fail(node, "link resolves to a forwarded node");
      if (i && links[i - 1].first >= links[i].first)
        fail(node, "links not sorted and unique");
    }
    if (isMisaligned(node))
      fail(node, "misaligned links on a non-collapsed node");
    if (node->isCollapsed()) {
      if (links.size() > 1 || (links.size() == 1 && links[0].first != 0))
        fail(node, "collapsed node has a link at a non-zero offset");
      if (node->arrayElemSize)
        fail(node, "collapsed node has an element size");
    }
    if (std::optional<uint64_t> elem = node->arrayElemSize) {
      if (*elem == 0)
        fail(node, "array element size 0");
      for (const auto &link : links)
        if (link.first < 0 || link.first >= int64_t(*elem))
          fail(node, "array link offset outside [0, elemSize)");
    }
    for (size_t i = 0; i < node->reasons.size(); ++i) {
      NodeFlags bit = node->reasons[i].first;
      if (!llvm::isPowerOf2_32(static_cast<uint32_t>(bit)) ||
          !hasAny(bit, NodeFlags::StatusMask) || !hasAny(node->flags, bit))
        fail(node, "reason for a bit that is not a set status bit");
      if (i && node->reasons[i - 1].first >= bit)
        fail(node, "reasons not sorted and unique");
    }
    for (size_t i = 0; i < node->origins.size(); ++i)
      for (size_t j = 0; j < i; ++j)
        if (node->origins[i] == node->origins[j])
          fail(node, "duplicate origin");
  }
  return ok;
}
