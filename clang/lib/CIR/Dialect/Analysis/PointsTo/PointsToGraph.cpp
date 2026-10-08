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
//  - Subobjects (see PointsToGraph.h): a link of a node that runs into one
//    of its children; a child that resolves to its parent at another offset,
//    or a cycle of subobject links (the object holds itself). A collapse
//    unifies the children into the node; markArray keeps the children that
//    fit one element, re-keyed modulo its size, and unifies the others.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToGraph.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstring>
#include <mutex>

using namespace llvm;
using namespace cir::pointsto;

//===----------------------------------------------------------------------===//
// NodeFlags
//===----------------------------------------------------------------------===//

static constexpr unsigned numFlagBits = 14;

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
  case NodeFlags::AsyncAccess:
    return "AsyncAccess";
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
// GhostField
//===----------------------------------------------------------------------===//

namespace {
/// The process-wide table of ghost field names. Id 0 is the null field.
struct GhostFieldTable {
  std::mutex mutex;
  llvm::StringMap<uint32_t> ids;
  std::vector<StringRef> names{StringRef()};
};
} // namespace

static GhostFieldTable &getGhostFieldTable() {
  static GhostFieldTable table;
  return table;
}

GhostField GhostField::get(StringRef name) {
  assert(!name.empty() && "a ghost field needs a name");
  GhostFieldTable &table = getGhostFieldTable();
  std::lock_guard<std::mutex> lock(table.mutex);
  auto [it, inserted] =
      table.ids.try_emplace(name, static_cast<uint32_t>(table.names.size()));
  if (inserted)
    table.names.push_back(it->getKey());
  return GhostField(it->second);
}

StringRef GhostField::getName() const {
  GhostFieldTable &table = getGhostFieldTable();
  std::lock_guard<std::mutex> lock(table.mutex);
  return id < table.names.size() ? table.names[id] : StringRef();
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

/// The flags that flow from a parent to its children, from a child to its
/// parents, and from a child at offset 0 to its parents (see the file comment
/// of PointsToGraph.h).
static constexpr NodeFlags kDownFlags = NodeFlags::All & ~NodeFlags::Collapsed;
static constexpr NodeFlags kUpFlags = NodeFlags::Mod | NodeFlags::Ref |
                                      NodeFlags::HasUSM |
                                      NodeFlags::AsyncAccess;
static constexpr NodeFlags kUpAtZeroFlags =
    kUpFlags | NodeFlags::External | NodeFlags::Incomplete;

/// How deep a chain of subobject links is followed (a guard; real nesting is
/// far shallower).
static constexpr unsigned kMaxSubobjectDepth = 64;

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

SmallVector<GhostLink, 2> Node::getGhostLinks() const {
  SmallVector<GhostLink, 2> result;
  result.reserve(ghosts.size());
  for (const GhostLink &ghost : ghosts)
    result.push_back({ghost.offset, ghost.field, ghost.target.resolve()});
  return result;
}

SmallVector<SubobjectLink, 2> Node::getSubobjects() const {
  SmallVector<SubobjectLink, 2> result;
  result.reserve(subs.size());
  for (const SubobjectLink &sub : subs)
    result.push_back({sub.offset, sub.size, sub.child.resolve()});
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
  NodeFlags before = node->flags;
  node->flags |= flags;
  for (unsigned i = 0; i < numFlagBits; ++i) {
    auto bit = static_cast<NodeFlags>(1u << i);
    if (hasAny(flags & NodeFlags::StatusMask, bit))
      recordReason(node, bit, why);
  }
  if (node->flags != before)
    propagateTreeFlags(node);
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
    // The flags may have spread to the tree of subobjects around the node
    // (propagateTreeFlags), but the walk still visits each of them, for the
    // nodes their links reach.
    forEachSuccessor(node, [&](Node *target) {
      if (target && visited.insert(target).second)
        worklist.push_back(target);
    });
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
  if (node->shared)
    sharedFrontier.push_back(target);
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
  // A pointer slot of the node itself must not run into a subobject (its
  // own slots are in the child).
  for (const auto &link : links) {
    const SubobjectLink *next = llvm::upper_bound(
        node->subs, link.first,
        [](int64_t o, const SubobjectLink &sub) { return o < sub.offset; });
    if (next != node->subs.end() &&
        link.first + int64_t(pointerSize) > next->offset)
      return true;
  }
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
  Cell src = descend(from);
  assert(src && "getLink on an empty cell");
  Cell target(newNode(), 0);
  Worklist worklist;
  insertLink(src.getRawNode(), src.getRawOffset(), target, worklist);
  collapseIfMisaligned(src.getRawNode(), why, worklist);
  runWorklist(worklist, why);
  return target.resolve();
}

std::optional<Cell> Graph::lookupLink(Cell from) const {
  Cell src = descend(from);
  if (!src)
    return std::nullopt;
  auto &links = src.getRawNode()->links;
  auto *it = findLink(links, src.getRawOffset());
  if (it == links.end() || it->first != src.getRawOffset())
    return std::nullopt;
  return it->second.resolve();
}

void Graph::setLink(Cell from, Cell to, const Reason *why) {
  Cell src = descend(from);
  assert(src && to && "setLink on an empty cell");
  Worklist worklist;
  insertLink(src.getRawNode(), src.getRawOffset(), to, worklist);
  collapseIfMisaligned(src.getRawNode(), why, worklist);
  runWorklist(worklist, why);
}

//===----------------------------------------------------------------------===//
// Graph: ghost links
//===----------------------------------------------------------------------===//

static auto findGhost(SmallVectorImpl<GhostLink> &ghosts, int64_t offset,
                      GhostField field) {
  return llvm::lower_bound(
      ghosts, std::make_pair(offset, field.getId()),
      [](const GhostLink &g, const std::pair<int64_t, uint32_t> &key) {
        return std::make_pair(g.offset, g.field.getId()) < key;
      });
}

void Graph::insertGhost(Node *node, int64_t offset, GhostField field,
                        Cell target, Worklist &worklist) {
  offset = normalizeOffset(node, offset);
  if (node->shared)
    sharedFrontier.push_back(target);
  auto *it = findGhost(node->ghosts, offset, field);
  if (it != node->ghosts.end() && it->offset == offset && it->field == field) {
    worklist.emplace_back(it->target, target);
    return;
  }
  node->ghosts.insert(it, {offset, field, target});
}

void Graph::renormalizeGhosts(Node *node, Worklist &worklist) {
  if (node->ghosts.empty())
    return;
  auto old = std::move(node->ghosts);
  node->ghosts.clear();
  for (const GhostLink &ghost : old)
    insertGhostDeep(node, ghost.offset, ghost.field, ghost.target, worklist);
}

Cell Graph::getGhostLink(Cell from, GhostField field, const Reason *why) {
  if (std::optional<Cell> existing = lookupGhostLink(from, field))
    return *existing;
  Cell src = descend(from);
  assert(src && field && "getGhostLink on an empty cell or field");
  Cell target(newNode(), 0);
  Worklist worklist;
  insertGhost(src.getRawNode(), src.getRawOffset(), field, target, worklist);
  runWorklist(worklist, why);
  return target.resolve();
}

std::optional<Cell> Graph::lookupGhostLink(Cell from, GhostField field) const {
  Cell src = descend(from);
  if (!src || !field)
    return std::nullopt;
  auto &ghosts = src.getRawNode()->ghosts;
  auto *it = findGhost(ghosts, src.getRawOffset(), field);
  if (it == ghosts.end() || it->offset != src.getRawOffset() ||
      it->field != field)
    return std::nullopt;
  return it->target.resolve();
}

void Graph::setGhostLink(Cell from, GhostField field, Cell to,
                         const Reason *why) {
  Cell src = descend(from);
  assert(src && to && field && "setGhostLink on an empty cell or field");
  Worklist worklist;
  insertGhost(src.getRawNode(), src.getRawOffset(), field, to, worklist);
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
  auto mergedGhosts = std::move(merged->ghosts);
  auto mergedOrigins = std::move(merged->origins);
  auto mergedReasons = std::move(merged->reasons);
  auto mergedSubs = std::move(merged->subs);
  auto mergedParents = std::move(merged->parents);
  NodeFlags mergedFlags = merged->flags;
  uint64_t mergedSize = merged->size;
  merged->links.clear();
  merged->ghosts.clear();
  merged->origins.clear();
  merged->reasons.clear();
  merged->subs.clear();
  merged->parents.clear();
  merged->flags = NodeFlags::None;
  merged->size = 0;
  merged->arrayElemSize.reset();
  merged->forward = survivor;
  merged->forwardDelta = delta;
  // The survivor of a merge with a shared node is shared. It is marked at
  // the next flushShared, which then walks its links (those it had, and
  // those inserted below, which insertLink records only for a node already
  // shared).
  if (merged->shared && !survivor->shared)
    sharedFrontier.push_back(Cell(survivor));
  merged->shared = false;

  // A subobject link of the survivor to the merged node now names the
  // survivor itself: the same bytes (drop it), or the object holds itself
  // at a shift (collapse, below).
  bool selfShift = false;
  llvm::erase_if(survivor->subs, [&](const SubobjectLink &sub) {
    Cell child = sub.child.resolve();
    if (child.getRawNode() != survivor)
      return false;
    if (child.getRawOffset() != normalizeOffset(survivor, sub.offset))
      selfShift = true;
    return true;
  });
  llvm::append_range(survivor->parents, mergedParents);

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

  if (collapseResult || selfShift)
    collapseImpl(survivor, why, worklist);
  else if (elem && !survivor->arrayElemSize)
    markArrayImpl(survivor, *elem, why, worklist);

  // The children first, so that the links of the merged node land in them.
  for (const SubobjectLink &sub : mergedSubs)
    attachSub(survivor, sub.offset + delta, sub.size, sub.child, worklist, why);
  for (const auto &[off, target] : mergedLinks)
    insertLinkDeep(survivor, off + delta, target, worklist, why);
  for (const GhostLink &ghost : mergedGhosts)
    insertGhostDeep(survivor, ghost.offset + delta, ghost.field, ghost.target,
                    worklist);
  collapseIfMisaligned(survivor, why, worklist);
  // The merge may have closed a cycle of subobject links (a child merged
  // with an ancestor): the object holds itself.
  if (!survivor->subs.empty() && !survivor->isCollapsed())
    for (const SubobjectLink &sub : survivor->subs)
      if (Node *child = sub.child.getNode();
          child != survivor && isInSubtree(child, survivor)) {
        collapseImpl(survivor, why, worklist);
        break;
      }
  propagateTreeFlags(survivor);
}

void Graph::collapseImpl(Node *node, const Reason *why, Worklist &worklist) {

  if (node->isCollapsed())
    return;
  node->flags |= NodeFlags::Collapsed;
  recordReason(node, NodeFlags::Collapsed, why);
  node->arrayElemSize.reset();
  // A field-insensitive node has no subobjects: its children join it.
  absorbSubs(node, std::nullopt, worklist, why);
  renormalizeGhosts(node, worklist);
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
  // The children that fit one element move to their offset modulo its size,
  // the others join the node.
  absorbSubs(node, elemSize, worklist, why);
  // Re-insert the links at their offsets modulo the element size.
  auto oldLinks = std::move(node->links);
  node->links.clear();
  for (const auto &[off, target] : oldLinks)
    insertLinkDeep(node, off, target, worklist, why);
  renormalizeGhosts(node, worklist);
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
// Graph: subobjects
//===----------------------------------------------------------------------===//

const SubobjectLink *Graph::findSubAt(const Node *node, int64_t offset) {
  // Sorted by offset and disjoint: the last link starting at or before it.
  const SubobjectLink *it = llvm::upper_bound(
      node->subs, offset,
      [](int64_t o, const SubobjectLink &sub) { return o < sub.offset; });
  if (it == node->subs.begin())
    return nullptr;
  --it;
  return offset < it->offset + static_cast<int64_t>(it->size) ? it : nullptr;
}

Cell Graph::descend(Cell cell) const {
  Cell cur = cell.resolve();
  for (unsigned depth = 0; cur && depth < kMaxSubobjectDepth; ++depth) {
    Node *node = cur.getRawNode();
    if (node->subs.empty())
      break;
    const SubobjectLink *hit = findSubAt(node, cur.getRawOffset());
    if (!hit)
      break;
    Cell next =
        Cell(hit->child.getRawNode(),
             hit->child.getRawOffset() + cur.getRawOffset() - hit->offset)
            .resolve();
    if (next.getRawNode() == node)
      break;
    cur = next;
  }
  return cur;
}

SmallVector<ParentLink, 1> Graph::getParents(const Node *node) const {
  SmallVector<ParentLink, 1> result;
  const Node *live = Cell(const_cast<Node *>(node)).getNode();
  if (!live)
    return result;
  for (Cell back : live->parents) {
    Cell at = back.resolve();
    Node *parent = at.getRawNode();
    if (!parent || parent == live)
      continue;
    const SubobjectLink *sub = findSubAt(parent, at.getRawOffset());
    if (!sub || sub->offset != at.getRawOffset())
      continue;
    Cell child = sub->child.resolve();
    if (child.getRawNode() != live)
      continue;
    if (llvm::any_of(result, [&](const ParentLink &p) {
          return p.parent == parent && p.offset == sub->offset;
        }))
      continue;
    result.push_back({parent, sub->offset, sub->size, child.getRawOffset()});
  }
  return result;
}

void Graph::forEachInSubtree(const Node *node,
                             function_ref<void(const Node *)> fn) const {
  const Node *root = Cell(const_cast<Node *>(node)).getNode();
  if (!root)
    return;
  SmallPtrSet<const Node *, 8> seen;
  SmallVector<const Node *, 8> worklist{root};
  seen.insert(root);
  while (!worklist.empty()) {
    const Node *n = worklist.pop_back_val();
    fn(n);
    for (const SubobjectLink &sub : n->subs)
      if (const Node *child = sub.child.getNode();
          child && seen.insert(child).second)
        worklist.push_back(child);
  }
}

bool Graph::isInSubtree(const Node *ancestor, const Node *node) const {
  bool found = false;
  forEachInSubtree(ancestor, [&](const Node *n) { found |= n == node; });
  return found;
}

void Graph::forEachSuccessor(const Node *node, function_ref<void(Node *)> fn,
                             bool allParents) const {
  for (const auto &link : node->links)
    fn(link.second.getNode());
  for (const GhostLink &ghost : node->ghosts)
    fn(ghost.target.getNode());
  for (const SubobjectLink &sub : node->subs)
    fn(sub.child.getNode());
  if (node->parents.empty())
    return;
  for (const ParentLink &parent : getParents(node))
    if (allParents || parent.offset == 0)
      fn(parent.parent);
}

void Graph::propagateTreeFlags(Node *start) {
  start = Cell(start).getNode();
  if (!start || (start->subs.empty() && start->parents.empty()))
    return;
  // Add the flags \p mask of \p from to \p to, with their reasons.
  auto transfer = [&](const Node *from, Node *to, NodeFlags mask) {
    NodeFlags add = from->flags & mask & ~to->flags;
    if (add == NodeFlags::None)
      return false;
    to->flags |= add;
    for (unsigned i = 0; i < numFlagBits; ++i) {
      auto bit = static_cast<NodeFlags>(1u << i);
      if (hasAny(add & NodeFlags::StatusMask, bit))
        recordReason(to, bit, from->getReason(bit));
    }
    return true;
  };
  SmallVector<Node *, 8> worklist{start};
  SmallPtrSet<Node *, 8> queued;
  queued.insert(start);
  auto push = [&](Node *n) {
    if (queued.insert(n).second)
      worklist.push_back(n);
  };
  while (!worklist.empty()) {
    Node *node = worklist.pop_back_val();
    queued.erase(node);
    if (node->isForwarded())
      continue;
    SmallVector<ParentLink, 1> parents = getParents(node);
    // Take what the neighbours have, then hand on what this node has.
    for (const ParentLink &parent : parents)
      transfer(parent.parent, node, kDownFlags);
    for (const SubobjectLink &sub : node->subs)
      if (Node *child = sub.child.getNode(); child != node)
        transfer(child, node, sub.offset == 0 ? kUpAtZeroFlags : kUpFlags);
    for (const SubobjectLink &sub : node->subs)
      if (Node *child = sub.child.getNode();
          child != node && transfer(node, child, kDownFlags))
        push(child);
    for (const ParentLink &parent : parents)
      if (transfer(node, parent.parent,
                   parent.offset == 0 ? kUpAtZeroFlags : kUpFlags))
        push(parent.parent);
  }
}

void Graph::insertLinkDeep(Node *node, int64_t offset, Cell target,
                           Worklist &worklist, const Reason *why) {
  if (node->subs.empty()) {
    insertLink(node, offset, target, worklist);
    return;
  }
  Cell at = descend(Cell(node, offset));
  insertLink(at.getRawNode(), at.getRawOffset(), target, worklist);
  if (at.getRawNode() != node)
    collapseIfMisaligned(at.getRawNode(), why, worklist);
}

void Graph::insertGhostDeep(Node *node, int64_t offset, GhostField field,
                            Cell target, Worklist &worklist) {
  if (node->subs.empty()) {
    insertGhost(node, offset, field, target, worklist);
    return;
  }
  Cell at = descend(Cell(node, offset));
  insertGhost(at.getRawNode(), at.getRawOffset(), field, target, worklist);
}

void Graph::attachSub(Node *node, int64_t offset, uint64_t size, Cell child,
                      Worklist &worklist, const Reason *why) {
  anySubobjects = true;
  Cell at = Cell(node, offset).resolve();
  Node *n = at.getRawNode();
  int64_t o = at.getRawOffset();
  Cell c = child.resolve();
  auto fallback = [&] { worklist.emplace_back(c, Cell(n, o)); };
  // Find the node the range belongs to: inside an existing child, go down.
  for (unsigned depth = 0;; ++depth) {
    if (c.getRawNode() == n) {
      // The child is the node itself: the same bytes, or itself at a shift.
      if (c.getRawOffset() != o)
        collapseImpl(n, why, worklist);
      return;
    }
    int64_t end = o + static_cast<int64_t>(size);
    if (depth >= kMaxSubobjectDepth || n->isCollapsed() ||
        (n->arrayElemSize &&
         (o < 0 || end > static_cast<int64_t>(*n->arrayElemSize)))) {
      fallback();
      return;
    }
    const SubobjectLink *container = nullptr;
    bool partial = false;
    for (const SubobjectLink &sub : n->subs) {
      int64_t subEnd = sub.offset + static_cast<int64_t>(sub.size);
      if (sub.offset >= end || o >= subEnd)
        continue;
      if (sub.offset == o && sub.size == size) {
        // The same range: one object.
        worklist.emplace_back(sub.child, c);
        return;
      }
      if (sub.offset <= o && end <= subEnd) {
        container = &sub;
        break;
      }
      if (o <= sub.offset && subEnd <= end)
        continue; // Moves into the new child below.
      partial = true;
    }
    if (partial) {
      fallback();
      return;
    }
    if (!container)
      break;
    Cell next = Cell(container->child.getRawNode(),
                     container->child.getRawOffset() + o - container->offset)
                    .resolve();
    if (next.getRawNode() == n) {
      fallback();
      return;
    }
    n = next.getRawNode();
    o = next.getRawOffset();
  }
  Node *cn = c.getRawNode();
  int64_t co = c.getRawOffset();
  // A child that holds its parent would make a cycle: the object would
  // contain itself.
  if (isInSubtree(cn, n)) {
    fallback();
    return;
  }

  // Move what the node holds in the range into the child.
  int64_t end = o + static_cast<int64_t>(size);
  auto inRange = [&](int64_t off) { return off >= o && off < end; };
  SmallVector<std::pair<int64_t, Cell>, 4> movedLinks;
  llvm::erase_if(n->links, [&](const std::pair<int64_t, Cell> &link) {
    if (!inRange(link.first))
      return false;
    movedLinks.push_back(link);
    return true;
  });
  SmallVector<GhostLink, 2> movedGhosts;
  llvm::erase_if(n->ghosts, [&](const GhostLink &ghost) {
    if (!inRange(ghost.offset))
      return false;
    movedGhosts.push_back(ghost);
    return true;
  });
  SmallVector<SubobjectLink, 2> movedSubs;
  llvm::erase_if(n->subs, [&](const SubobjectLink &sub) {
    if (!inRange(sub.offset))
      return false;
    movedSubs.push_back(sub);
    return true;
  });
  SubobjectLink *pos =
      llvm::lower_bound(n->subs, o, [](const SubobjectLink &sub, int64_t off) {
        return sub.offset < off;
      });
  n->subs.insert(pos, {o, size, c});
  cn->parents.push_back(Cell(n, o));
  if (o == 0)
    zeroOffsetParents.push_back(Cell(n));
  if (n->shared && !cn->shared)
    sharedFrontier.push_back(c);
  else if (cn->shared && !n->shared)
    sharedFrontier.push_back(Cell(n));
  if (end > 0)
    n->size = std::max(n->size, static_cast<uint64_t>(end));
  if (int64_t childEnd = co + static_cast<int64_t>(size); childEnd > 0)
    cn->size = std::max(cn->size, static_cast<uint64_t>(childEnd));

  for (const auto &[off, target] : movedLinks) {
    Cell to = Cell(cn, co + off - o).resolve();
    insertLinkDeep(to.getRawNode(), to.getRawOffset(), target, worklist, why);
  }
  for (const GhostLink &ghost : movedGhosts) {
    Cell to = Cell(cn, co + ghost.offset - o).resolve();
    insertGhostDeep(to.getRawNode(), to.getRawOffset(), ghost.field,
                    ghost.target, worklist);
  }
  for (const SubobjectLink &sub : movedSubs)
    attachSub(cn, co + sub.offset - o, sub.size, sub.child, worklist, why);
  if (Node *live = Cell(cn).getNode(); live && !live->isForwarded())
    collapseIfMisaligned(live, why, worklist);
  propagateTreeFlags(cn);
}

Cell Graph::getSubobject(Cell at, uint64_t size, const Reason *why) {
  Cell start = at.resolve();
  if (!start || size == 0)
    return start;
  // An existing child for exactly this range, at any depth.
  Cell cur = start;
  for (unsigned depth = 0; depth < kMaxSubobjectDepth; ++depth) {
    Node *node = cur.getRawNode();
    int64_t off = cur.getRawOffset();
    const SubobjectLink *hit = findSubAt(node, off);
    if (!hit)
      break;
    if (hit->offset == off && hit->size == size)
      return hit->child.resolve();
    if (off + static_cast<int64_t>(size) >
        hit->offset + static_cast<int64_t>(hit->size))
      break;
    Cell next = Cell(hit->child.getRawNode(),
                     hit->child.getRawOffset() + off - hit->offset)
                    .resolve();
    if (next.getRawNode() == node)
      break;
    cur = next;
  }
  Node *child = newNode();
  child->size = size;
  Worklist worklist;
  attachSub(start.getRawNode(), start.getRawOffset(), size, Cell(child, 0),
            worklist, why);
  runWorklist(worklist, why);
  return Cell(child, 0).resolve();
}

void Graph::absorbSubs(Node *node, std::optional<uint64_t> elemSize,
                       Worklist &worklist, const Reason *why) {
  if (node->subs.empty())
    return;
  auto subs = std::move(node->subs);
  node->subs.clear();
  for (const SubobjectLink &sub : subs) {
    if (elemSize) {
      auto e = static_cast<int64_t>(*elemSize);
      int64_t norm = ((sub.offset % e) + e) % e;
      if (norm + static_cast<int64_t>(sub.size) <= e) {
        attachSub(node, norm, sub.size, sub.child, worklist, why);
        continue;
      }
    }
    worklist.emplace_back(Cell(node, sub.offset), sub.child);
  }
}

bool Graph::absorbOverflowingSubobjects(const Reason *why) {
  if (!anySubobjects)
    return false;
  // Whether the child cell \p child of a slot of \p size bytes has bytes
  // outside it: then a pointer to the member was used as a pointer to the
  // object (or the child stands for a larger object too, which the folding
  // covers as well).
  auto overflows = [&](Cell child, uint64_t size) {
    const Node *node = child.getRawNode();
    int64_t start = child.getRawOffset();
    auto slot = static_cast<int64_t>(size);
    if (start != 0 || node->size > size)
      return true;
    if (std::optional<uint64_t> elem = node->arrayElemSize;
        elem && *elem > size)
      return true;
    for (const auto &link : node->links)
      if (link.first < 0 || link.first + int64_t(pointerSize) > slot)
        return true;
    for (const GhostLink &ghost : node->ghosts)
      if (ghost.offset < 0 || ghost.offset >= slot)
        return true;
    for (const SubobjectLink &sub : node->subs)
      if (sub.offset < 0 || sub.offset + static_cast<int64_t>(sub.size) > slot)
        return true;
    return false;
  };
  bool changed = false;
  for (bool again = true; again;) {
    again = false;
    Worklist worklist;
    // The nodes that may still have a child at offset 0, each once.
    SmallPtrSet<Node *, 32> seen;
    llvm::erase_if(zeroOffsetParents, [&](Cell cell) {
      Node *node = cell.getNode();
      if (!seen.insert(node).second || node->subs.empty() ||
          node->subs.front().offset != 0)
        return true;
      const SubobjectLink &sub = node->subs.front();
      Cell child = sub.child.resolve();
      if (child.getRawNode() != node && overflows(child, sub.size))
        worklist.emplace_back(Cell(node, 0), child);
      return false;
    });
    if (worklist.empty())
      break;
    runWorklist(worklist, why);
    again = changed = true;
  }
  return changed;
}

//===----------------------------------------------------------------------===//
// Graph: cloning
//===----------------------------------------------------------------------===//

void Graph::markShared(Cell cell) {
  assert(trackShared && "markShared without shared tracking");
  if (cell)
    sharedFrontier.push_back(cell);
}

void Graph::flushShared() {
  while (!sharedFrontier.empty()) {
    Node *node = sharedFrontier.back().getNode();
    sharedFrontier.pop_back();
    if (!node || node->shared)
      continue;
    node->shared = true;
    // A tree of subobjects is shared as a whole: a copy of part of it would
    // hold bytes the shared rest also holds.
    forEachSuccessor(
        node, [&](Node *next) { sharedFrontier.push_back(Cell(next)); },
        /*allParents=*/true);
  }
}

bool Graph::isShared(const Node *node) const {
  const Node *live = Cell(const_cast<Node *>(node)).getNode();
  return live && live->shared;
}

CloneMap Graph::cloneInto(Graph &dst, ArrayRef<Cell> roots, NodeFlags keepMask,
                          const CloneOptions &options) const {
  assert((!options.isShared || &dst == this) &&
         "shared nodes map to themselves only within one graph");
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
    if (options.isShared && options.isShared(src)) {
      result.nodeMap[src] = Cell(src, 0);
      return;
    }
    Node *copy = dst.newNode();
    copy->flags = src->flags & keepMask;
    copy->size = src->size;
    copy->arrayElemSize = src->arrayElemSize;
    copy->origins = src->origins;
    if (options.rewriteOrigin)
      for (Origin &origin : copy->origins)
        options.rewriteOrigin(origin);
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
  // A node comes with its whole tree of subobjects, parents included.
  for (size_t i = 0; i < order.size(); ++i)
    forEachSuccessor(
        order[i], [&](Node *next) { visit(Cell(next)); },
        /*allParents=*/true);

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
    for (const GhostLink &ghost : src->ghosts) {
      Cell t = ghost.target.resolve();
      copy->ghosts.push_back({ghost.offset, ghost.field,
                              Cell(result.nodeMap[t.getRawNode()].getRawNode(),
                                   t.getRawOffset())});
    }
    for (const SubobjectLink &sub : src->subs) {
      Cell t = sub.child.resolve();
      Node *child = result.nodeMap[t.getRawNode()].getRawNode();
      copy->subs.push_back(
          {sub.offset, sub.size, Cell(child, t.getRawOffset())});
      // The back link (on a shared child too, which is in this graph).
      child->parents.push_back(Cell(copy, sub.offset));
      dst.anySubobjects = true;
      if (sub.offset == 0)
        dst.zeroOffsetParents.push_back(Cell(copy));
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
    if (!node->ghosts.empty()) {
      os << " ghost={";
      llvm::interleave(
          node->ghosts, os,
          [&](const GhostLink &ghost) {
            os << ghost.offset << ':' << ghost.field.getName() << "->";
            printCell(os, ghost.target);
          },
          ", ");
      os << '}';
    }
    if (!node->subs.empty()) {
      os << " sub={";
      llvm::interleave(
          node->subs, os,
          [&](const SubobjectLink &sub) {
            os << sub.offset << ':' << sub.size << "->";
            printCell(os, sub.child);
          },
          ", ");
      os << '}';
    }
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
      if (!node->links.empty() || !node->ghosts.empty() ||
          !node->origins.empty() || !node->reasons.empty() ||
          !node->subs.empty() || !node->parents.empty() ||
          node->flags != NodeFlags::None)
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
    const auto &ghosts = node->ghosts;
    for (size_t i = 0; i < ghosts.size(); ++i) {
      if (!ghosts[i].field)
        fail(node, "ghost link without a field");
      if (!ghosts[i].target)
        fail(node, "null ghost link target");
      else if (ghosts[i].target.getNode()->isForwarded())
        fail(node, "ghost link resolves to a forwarded node");
      if (i &&
          std::make_pair(ghosts[i - 1].offset, ghosts[i - 1].field.getId()) >=
              std::make_pair(ghosts[i].offset, ghosts[i].field.getId()))
        fail(node, "ghost links not sorted and unique");
      if (normalizeOffset(node, ghosts[i].offset) != ghosts[i].offset)
        fail(node, "ghost link offset not normalized");
    }
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

    // Subobject links.
    const auto &subs = node->subs;
    if (!subs.empty() && node->isCollapsed())
      fail(node, "collapsed node has subobject links");
    for (size_t i = 0; i < subs.size(); ++i) {
      const SubobjectLink &sub = subs[i];
      int64_t end = sub.offset + static_cast<int64_t>(sub.size);
      if (sub.size == 0)
        fail(node, "subobject link of size 0");
      if (i && subs[i - 1].offset + static_cast<int64_t>(subs[i - 1].size) >
                   sub.offset)
        fail(node, "subobject links not sorted and disjoint");
      if (normalizeOffset(node, sub.offset) != sub.offset)
        fail(node, "subobject link offset not normalized");
      if (std::optional<uint64_t> elem = node->arrayElemSize)
        if (sub.offset < 0 || end > static_cast<int64_t>(*elem))
          fail(node, "subobject link outside one array element");
      const Node *child = sub.child.getNode();
      if (!child) {
        fail(node, "null subobject link target");
        continue;
      }
      if (child->isForwarded())
        fail(node, "subobject link resolves to a forwarded node");
      if (child == node || isInSubtree(child, node))
        fail(node, "subobject link cycle");
      NodeFlags down = node->flags & kDownFlags & ~child->flags;
      NodeFlags up = child->flags &
                     (sub.offset == 0 ? kUpAtZeroFlags : kUpFlags) &
                     ~node->flags;
      if (down != NodeFlags::None)
        fail(node, "child n" + Twine(child->id) + " lacks the flags " +
                       toString(down));
      if (up != NodeFlags::None)
        fail(node, "lacks the flags " + toString(up) + " of child n" +
                       Twine(child->id));
    }
    for (const auto &link : links)
      if (findSubAt(node, link.first))
        fail(node, "link inside a subobject link's range");
    for (const GhostLink &ghost : ghosts)
      if (findSubAt(node, ghost.offset))
        fail(node, "ghost link inside a subobject link's range");
  }
  return ok;
}
