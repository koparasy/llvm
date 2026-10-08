//===- PointsToGraph.h - DSA-style points-to memory graph -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the memory graph of the CIR points-to analysis: abstract
// memory objects (nodes), byte-offset cells into them, the pointer links
// between cells, and the unification-based union-find that merges nodes. The
// design follows Data Structure Analysis (Lattner et al., PLDI 2007) and
// SeaDSA: fields are byte offsets and type-agnostic, and merged nodes are
// forwarded to their survivor rather than kept in an equivalence-class side
// table, so a Cell taken before a merge stays valid after it.
//
// Forwarding and offset convention
// --------------------------------
// When node M is merged into node S, M stores (forward = S, forwardDelta = d).
// A cell (M, o) then denotes the cell (S, o + d). Chains are followed with path
// compression, summing the deltas. After the live node is reached its offset
// is normalized: 0 on a Collapsed node, `o mod elemSize` (in [0, elemSize)) on
// an array node, unchanged otherwise. Every public accessor returns such a
// resolved, normalized value.
//
// Missing links
// -------------
// A node has an outgoing link only at the offsets where a pointer has been
// seen. A missing link means "points to nothing seen yet", not "points to
// unknown memory". Graph::getLink creates an empty target node on demand,
// Graph::lookupLink reads without creating.
//
// Ghost links
// -----------
// Besides its byte-offset links, a node has named ghost links: abstract
// properties that a library model attaches to an object and that no byte of
// it holds, such as the storage a SYCL buffer handle refers to. A ghost link
// is keyed by (offset, GhostField): the offset of the sub-object it describes
// (a closure holding three accessors has one `$storage` ghost per accessor
// field) and an interned name. Ghost links never occupy bytes: they are kept
// apart from the links, so they never collide with a real pointer slot or
// make a node misaligned, whatever the array element size. Their offsets
// follow the node's normalization (0 on a collapsed node, modulo the element
// size on an array), and two ghost links that land on the same key unify
// their targets. Unification, cloning, reachability and printing treat them
// like links.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_POINTSTOGRAPH_H
#define CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_POINTSTOGRAPH_H

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mlir {
class Operation;
} // namespace mlir

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace cir::pointsto {

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

//===----------------------------------------------------------------------===//
// NodeFlags
//===----------------------------------------------------------------------===//

/// The kind and status bits of a node. Kinds form a set because unification
/// can merge objects of different kinds.
enum class NodeFlags : uint32_t {
  None = 0,

  // Kind bits.
  Stack = 1u << 0,
  Heap = 1u << 1,
  Global = 1u << 2,
  Ghost = 1u << 3,
  Function = 1u << 4,
  UnknownObj = 1u << 5,

  // Status bits. Each records the first Reason that set it.
  /// Reachable from something not analysed yet (a formal, the return of an
  /// unprocessed caller).
  Incomplete = 1u << 6,
  /// Escaped to, or came from, code we don't see.
  External = 1u << 7,
  /// Field-insensitive: every offset is 0.
  Collapsed = 1u << 8,
  /// May denote more than one live concrete object.
  Summary = 1u << 9,
  Mod = 1u << 10,
  Ref = 1u << 11,
  HasUSM = 1u << 12,
  /// Accessed by code we don't see while the host runs, until a later call
  /// synchronizes with it (the host memory of an asynchronous SYCL memory
  /// command). Not spread along links: the access is to the node's bytes.
  AsyncAccess = 1u << 13,

  KindMask = Stack | Heap | Global | Ghost | Function | UnknownObj,
  StatusMask = Incomplete | External | Collapsed | Summary | Mod | Ref |
      HasUSM | AsyncAccess,
  All = KindMask | StatusMask,

  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/AsyncAccess)
};

/// Return true if any bit of \p mask is set in \p flags.
inline bool hasAny(NodeFlags flags, NodeFlags mask) {
  return (flags & mask) != NodeFlags::None;
}

/// Return true if every bit of \p mask is set in \p flags.
inline bool hasAll(NodeFlags flags, NodeFlags mask) {
  return (flags & mask) == mask;
}

/// Print \p flags as the names of its set bits in bit order, joined with '|'
/// (for example "Heap|Summary"). Return the empty string for None.
std::string toString(NodeFlags flags);

/// Return the name of a single flag bit, for example "Collapsed".
llvm::StringRef getFlagName(NodeFlags bit);

//===----------------------------------------------------------------------===//
// Reason
//===----------------------------------------------------------------------===//

/// Why a status flag was set. Reasons are immutable, allocated from a
/// ReasonTable, and may chain to the reason that caused them.
struct Reason {
  enum class Kind : uint8_t {
    MergedAt,
    EscapedTo,
    UnknownCaller,
    CollapsedAt,
    Budget,
    UnmodelledRole,
    IndirectUnresolved,
    SummaryObject,
    UnhandledOp,
  };

  Kind kind;
  /// The op responsible, or null.
  mlir::Operation *op;
  /// Free-form detail. Owned by the ReasonTable that made the reason.
  llvm::StringRef detail;
  /// The reason this one follows from, or null.
  const Reason *cause;
};

/// Return the name of \p kind, for example "EscapedTo".
llvm::StringRef getReasonKindName(Reason::Kind kind);

/// Whether \p why is a reason Graph::markReachable derived for a node it
/// reached: its detail is "reachable from ..." and its cause is the reason of
/// the node the walk started from.
bool isDerivedReason(const Reason *why);

/// The reason \p why follows from after skipping every derived reason
/// (isDerivedReason): the first cause of the chain.
const Reason *getRootReason(const Reason *why);

/// One sentence, the text remarks print, for why a pointer has the status
/// \p why records, for example
///   "it is reachable from an escaped pointer passed to @foo (t.cpp:12)".
/// A derived reason reads "it is reachable from <root>", any other one
/// "it is <root>", where <root> describes the first cause, ending with the
/// file:line of its op (or "in @function" if the op has no file location).
/// Each Reason::Kind has its own template; an unknown detail is quoted. The
/// text is deterministic: it names ops, symbols and locations, never node ids
/// or addresses. Null gives "no reason was recorded".
std::string explain(const Reason *why);

/// The chain of \p why, one entry per reason from \p why to its last cause,
/// each as `<Kind>(<op> <file:line:col>[, "<detail>"])`, the format
/// Graph::print uses for reasons.
llvm::SmallVector<std::string, 4> explainChain(const Reason *why);

/// Owns Reason objects so that pointers to them stay stable for the lifetime
/// of the table. A table is either owned by one Graph or shared by many.
class ReasonTable {
public:
  ReasonTable() = default;
  ReasonTable(const ReasonTable &) = delete;
  ReasonTable &operator=(const ReasonTable &) = delete;

  /// Allocate a new reason. \p detail is copied into the table.
  const Reason *make(Reason::Kind kind, mlir::Operation *op = nullptr,
                     llvm::StringRef detail = {},
                     const Reason *cause = nullptr);

private:
  llvm::BumpPtrAllocator allocator;
};

//===----------------------------------------------------------------------===//
// Origin
//===----------------------------------------------------------------------===//

/// Where an abstract object comes from. A node collects the deduplicated
/// origins of every node merged into it.
struct Origin {
  enum class Kind : uint8_t { Alloca, GlobalSym, HeapCall, GhostSite, FuncSym };

  Kind kind;
  /// The allocating op (cir.alloca, the heap call, the model site), or null.
  mlir::Operation *site = nullptr;
  /// The symbol for globals and functions, or null.
  mlir::SymbolRefAttr sym = {};
  /// Call-path id of the context the object was cloned into; 0 is none.
  uint32_t ctx = 0;

  friend bool operator==(const Origin &a, const Origin &b) {
    return a.kind == b.kind && a.site == b.site && a.sym == b.sym &&
           a.ctx == b.ctx;
  }
  friend bool operator!=(const Origin &a, const Origin &b) { return !(a == b); }
};

//===----------------------------------------------------------------------===//
// GhostField
//===----------------------------------------------------------------------===//

/// The interned name of a ghost link (see the file comment). A model plug-in
/// gets its fields once, for example `GhostField::get("$storage")`. The same
/// name always gives the same field, in every graph; the ids order fields by
/// first registration.
class GhostField {
public:
  GhostField() = default;

  /// The field named \p name, registering it the first time. Thread-safe.
  static GhostField get(llvm::StringRef name);

  /// The registered name; empty for the null field.
  llvm::StringRef getName() const;
  uint32_t getId() const { return id; }
  explicit operator bool() const { return id != 0; }

  friend bool operator==(GhostField a, GhostField b) { return a.id == b.id; }
  friend bool operator!=(GhostField a, GhostField b) { return a.id != b.id; }
  friend bool operator<(GhostField a, GhostField b) { return a.id < b.id; }

private:
  explicit GhostField(uint32_t id) : id(id) {}
  uint32_t id = 0;
};

//===----------------------------------------------------------------------===//
// Cell
//===----------------------------------------------------------------------===//

class Node;

/// A byte offset into a node. A Cell can be held across unifications: it may
/// name a forwarded node, and every accessor resolves it first (see the file
/// comment for the forwarding convention).
class Cell {
public:
  Cell() = default;
  Cell(Node *node, int64_t offset = 0) : node(node), offset(offset) {}

  /// Follow forwarding to the live node, with path compression, and return
  /// the equivalent cell on it with a normalized offset.
  Cell resolve() const;

  /// The resolved node, or null for an empty cell.
  Node *getNode() const { return resolve().node; }
  /// The resolved, normalized offset.
  int64_t getOffset() const { return resolve().offset; }

  /// The stored, unresolved fields. Mostly for hashing and debugging.
  Node *getRawNode() const { return node; }
  int64_t getRawOffset() const { return offset; }

  explicit operator bool() const { return node != nullptr; }

  /// Compare the stored fields. Resolve both sides to ask whether two cells
  /// denote the same location: `a.resolve() == b.resolve()`.
  friend bool operator==(const Cell &a, const Cell &b) {
    return a.node == b.node && a.offset == b.offset;
  }
  friend bool operator!=(const Cell &a, const Cell &b) { return !(a == b); }

private:
  Node *node = nullptr;
  int64_t offset = 0;
};

//===----------------------------------------------------------------------===//
// Node
//===----------------------------------------------------------------------===//

/// A ghost link: the target of the ghost field \p field of the sub-object at
/// \p offset.
struct GhostLink {
  int64_t offset;
  GhostField field;
  Cell target;
};

/// An abstract memory object. Nodes are created and mutated only by Graph.
class Node {
public:
  /// Creation id, unique within the graph. Printing orders nodes by it.
  uint32_t getId() const { return id; }
  /// True if this node was merged into another one and is no longer live.
  bool isForwarded() const { return forward != nullptr; }

  NodeFlags getFlags() const { return flags; }
  bool hasFlags(NodeFlags mask) const { return hasAll(flags, mask); }
  bool isCollapsed() const { return hasAny(flags, NodeFlags::Collapsed); }

  /// Maximum known extent in bytes (0 when unknown). For an array node this
  /// is the extent of the whole region seen, not the element size.
  uint64_t getSize() const { return size; }
  /// The element size if offsets are taken modulo it.
  std::optional<uint64_t> getArrayElemSize() const { return arrayElemSize; }
  bool isArray() const { return arrayElemSize.has_value(); }

  /// The outgoing links as (offset, resolved target), sorted by offset.
  llvm::SmallVector<std::pair<int64_t, Cell>, 4> getLinks() const;
  size_t getNumLinks() const { return links.size(); }

  /// The ghost links with resolved targets, sorted by (offset, field).
  llvm::SmallVector<GhostLink, 2> getGhostLinks() const;
  size_t getNumGhostLinks() const { return ghosts.size(); }

  llvm::ArrayRef<Origin> getOrigins() const { return origins; }

  /// The first reason recorded for the status bit \p flag, or null.
  const Reason *getReason(NodeFlags flag) const;
  /// The (status bit, reason) pairs, sorted by bit.
  llvm::ArrayRef<std::pair<NodeFlags, const Reason *>> getReasons() const {
    return reasons;
  }

private:
  friend class Cell;
  friend class Graph;

  explicit Node(uint32_t id) : id(id) {}

  /// The survivor this node was merged into, and the offset shift: (this, o)
  /// denotes (forward, o + forwardDelta).
  Node *forward = nullptr;
  int64_t forwardDelta = 0;
  uint32_t id;
  NodeFlags flags = NodeFlags::None;
  uint64_t size = 0;
  std::optional<uint64_t> arrayElemSize;
  /// Pointer slots: (byte offset, target), sorted and unique by offset.
  llvm::SmallVector<std::pair<int64_t, Cell>, 2> links;
  /// Ghost links, sorted and unique by (offset, field id).
  llvm::SmallVector<GhostLink, 0> ghosts;
  llvm::SmallVector<Origin, 1> origins;
  /// First reason per status bit, sorted by bit.
  llvm::SmallVector<std::pair<NodeFlags, const Reason *>, 0> reasons;
  /// Reachable from a node Graph::markShared was given (see there).
  bool shared = false;
};

//===----------------------------------------------------------------------===//
// CloneMap
//===----------------------------------------------------------------------===//

/// The result of Graph::cloneInto: each source node reachable from the roots
/// maps to the cell of its copy in the destination graph.
class CloneMap {
public:
  /// The copy of \p node (resolved first), or an empty cell if it wasn't
  /// cloned.
  Cell lookup(const Node *node) const;

  /// Map a source cell to the destination: the copy of its node at the same
  /// (resolved) offset. Return an empty cell if the node wasn't cloned.
  Cell map(Cell srcCell) const;

  size_t size() const { return nodeMap.size(); }

  /// Every (source node, copy) pair, the source as it was when it was
  /// cloned (resolve it before use; it may have been merged since). A node
  /// CloneOptions::isShared kept maps to itself.
  const llvm::DenseMap<const Node *, Cell> &entries() const { return nodeMap; }

private:
  friend class Graph;
  llvm::DenseMap<const Node *, Cell> nodeMap;
};

/// How Graph::cloneInto treats some nodes (both optional).
struct CloneOptions {
  /// A node for which this returns true is not copied: it maps to itself,
  /// and the walk does not go past it. Only valid when cloning into the
  /// source graph (the bottom-up phase's globals graph, whose nodes every
  /// function's graph shares).
  llvm::function_ref<bool(const Node *)> isShared;
  /// Called on every origin of every copy: the hook that gives a cloned
  /// object its calling context (Origin::ctx).
  llvm::function_ref<void(Origin &)> rewriteOrigin;
};

//===----------------------------------------------------------------------===//
// Graph
//===----------------------------------------------------------------------===//

/// A points-to memory graph: the nodes of one function (local, bottom-up or
/// top-down) or of the globals.
///
/// Collapse rules (see the .cpp file for the details):
///  - two links whose offsets differ by less than the pointer size;
///  - unifying a node with itself at a different (normalized) offset;
///  - merging two arrays of different element size;
///  - markArray with a different element size than the node already has;
///  - an explicit collapse().
class Graph {
public:
  /// \p pointerSize is the size of a pointer slot in bytes. \p reasons is a
  /// shared table; when null the graph owns its own.
  explicit Graph(unsigned pointerSize = 8, ReasonTable *reasons = nullptr);
  ~Graph();
  Graph(const Graph &) = delete;
  Graph &operator=(const Graph &) = delete;

  unsigned getPointerSize() const { return pointerSize; }
  ReasonTable &getReasonTable() const { return *reasonTable; }

  /// Convenience for getReasonTable().make(...).
  const Reason *makeReason(Reason::Kind kind, mlir::Operation *op = nullptr,
                           llvm::StringRef detail = {},
                           const Reason *cause = nullptr) const {
    return reasonTable->make(kind, op, detail, cause);
  }

  /// Create a node with one origin and the given flags.
  Node *makeNode(const Origin &origin, NodeFlags flags = NodeFlags::None);
  /// Create a node without an origin.
  Node *makeNode(NodeFlags flags = NodeFlags::None);

  /// Set \p flags on the node of \p cell. For each status bit that has no
  /// reason yet, record \p why (if non-null).
  void addFlags(Cell cell, NodeFlags flags, const Reason *why = nullptr);
  /// Set the status bits \p flags on the node of \p from and on every node
  /// reachable from it through links and ghost links (all of a node's links,
  /// whatever the offset of \p from). The node of \p from records \p why; the
  /// nodes reached from it record one derived reason, "reachable from an
  /// escaped pointer", whose cause is \p why. The walk is iterative with a
  /// visited set, so cycles and long chains are fine. \p flags must not contain
  /// Collapsed (collapsing would rewrite the links being walked). Returns the
  /// number of nodes visited.
  ///
  /// With \p stopAtFlagged the walk does not go past a node other than the
  /// one of \p from that already has all of \p flags: propagating a flag from
  /// every node that has it, one call per node, is then linear overall. When
  /// \p flags is exactly Incomplete the derived reason reads "reachable from
  /// an incomplete node" instead.
  size_t markReachable(Cell from, NodeFlags flags, const Reason *why = nullptr,
                       bool stopAtFlagged = false);

  /// Add \p origin to the node of \p cell unless it is already there.
  void addOrigin(Cell cell, const Origin &origin);
  /// Raise the known extent of \p node (resolved first) to at least \p size
  /// bytes.
  void growSize(Node *node, uint64_t size);

  /// The target of the pointer slot at \p from. If there is no link yet,
  /// create an empty target node (no flags, no origin) and link to it.
  Cell getLink(Cell from, const Reason *why = nullptr);
  /// The target of the pointer slot at \p from, without creating anything.
  std::optional<Cell> lookupLink(Cell from) const;
  /// Make the slot at \p from point to \p to. If it already points somewhere,
  /// unify the old target with \p to.
  void setLink(Cell from, Cell to, const Reason *why = nullptr);

  /// The target of the ghost field \p field of the sub-object at \p from.
  /// If there is none yet, create an empty target node and link to it.
  Cell getGhostLink(Cell from, GhostField field, const Reason *why = nullptr);
  /// The target of that ghost link, without creating anything.
  std::optional<Cell> lookupGhostLink(Cell from, GhostField field) const;
  /// Make the ghost field \p field of \p from point to \p to. If it already
  /// points somewhere, unify the old target with \p to.
  void setGhostLink(Cell from, GhostField field, Cell to,
                    const Reason *why = nullptr);

  /// Make \p a and \p b denote the same cell. Merges their nodes with the
  /// right offset shift, ORs the flags, unions the origins, keeps the
  /// survivor's reasons (and the merged-in node's for bits the survivor has
  /// none for), grows the size, and unifies the targets of colliding links
  /// and of colliding ghost links.
  /// The work runs off a worklist, so arbitrarily long chains are fine.
  /// \p why is recorded if the merge collapses a node.
  ///
  /// Survivor choice: the node whose cell has the larger offset survives, so
  /// the shift is never negative; on equal offsets the older node survives.
  /// Unifying a node with itself at two different offsets (for example `p`
  /// and `p+8`) means a recursive or array structure the graph can't express
  /// field-sensitively, and collapses the node. On an array node offsets are
  /// compared modulo the element size, so `p` and `p+elemSize` are equal.
  void unify(Cell a, Cell b, const Reason *why = nullptr);

  /// Make the node field-insensitive: every offset maps to 0, all link
  /// targets are unified into the single link at 0, the array element size
  /// is dropped, and the Collapsed flag is set (with \p why). Ghost links
  /// move to offset 0 too; those of one field unify, those of different
  /// fields stay apart.
  void collapse(Node *node, const Reason *why = nullptr);

  /// Take offsets on \p node modulo \p elemSize from now on. Existing links
  /// are normalized and the ones that land on the same offset unified. A node
  /// that already is an array of another element size collapses; a collapsed
  /// node stays collapsed.
  void markArray(Node *node, uint64_t elemSize, const Reason *why = nullptr);

  /// Deep-copy the nodes reachable from \p roots into \p dst. Flags are
  /// masked with \p keepMask (status reasons follow their bits), origins are
  /// copied with their ctx, and reasons are copied into dst's table unless it
  /// is shared with this graph. Links and ghost links, including cycles, are
  /// reproduced.
  CloneMap cloneInto(Graph &dst, llvm::ArrayRef<Cell> roots,
                     NodeFlags keepMask = NodeFlags::All,
                     const CloneOptions &options = {}) const;

  /// Shared nodes (the bottom-up phase): a node is shared when it is
  /// reachable from a node passed to markShared, through links and ghost
  /// links, as the graph is now. Off unless enabled; then the graph keeps
  /// track as links are added and nodes merge, and flushShared brings
  /// isShared up to date.
  void enableSharedTracking() { trackShared = true; }
  /// Make the node of \p cell, and what it reaches, shared (after the next
  /// flushShared).
  void markShared(Cell cell);
  /// Mark everything the shared nodes reach as shared.
  void flushShared();
  /// Whether the node of \p node (resolved) is shared, as of the last
  /// flushShared.
  bool isShared(const Node *node) const;

  /// The live (not forwarded) nodes, in creation order.
  auto nodes() const {
    return llvm::make_filter_range(
        allNodes, [](const Node *n) { return !n->isForwarded(); });
  }
  /// Number of live nodes.
  size_t getNumLiveNodes() const;

  /// Print one line per live node in creation-id order:
  /// \code
  ///   n<id> [<flags>] size=<n>[ arr=<elem>] origins={<origin>, ...}
  ///     links={<off>-><cell>, ...}[ ghost={<off>:<field>-><cell>, ...}]
  ///     [ reasons={<Flag>: <reason>, ...}]
  /// \endcode
  /// (on one line). A cell prints as `n<id>+<off>` or `n<id>-<off>`. An origin
  /// prints as `<kind>(<what>)`, where kind is stack, global, heap, ghost or
  /// func, and what is `@sym` and/or the site; a non-zero ctx appends
  /// `#<ctx>`. An op prints as its name, its "name" string attribute if any
  /// (`cir.alloca 'x'`), and `file:line:col` with the file's basename if it
  /// has a file location. A reason prints as `<Kind>(<op>[, "<detail>"])`,
  /// followed by ` <- <cause>` for up to 4 causes and ` <- ...` beyond.
  void print(llvm::raw_ostream &os) const;
  /// Print \p cell (resolved) as `n<id>+<off>`, or `null`.
  static void printCell(llvm::raw_ostream &os, Cell cell);
  void dump() const;

  /// Check the structural invariants: links sorted and unique per offset and
  /// at least a pointer apart (also across an array element boundary); live
  /// nodes reachable from links are not forwarded; forwarded nodes hold no
  /// state; collapsed nodes have at most one link, at 0, and no element size;
  /// array offsets are in [0, elemSize); ghost links are sorted and unique,
  /// normalized like offsets, and resolve to live nodes. Report problems to
  /// \p errs if given.
  bool verify(llvm::raw_ostream *errs = nullptr) const;

private:
  using Worklist = llvm::SmallVector<std::pair<Cell, Cell>, 8>;

  Node *newNode();
  void runWorklist(Worklist &worklist, const Reason *why);
  void merge(Node *survivor, Node *merged, int64_t delta, Worklist &worklist,
             const Reason *why);
  void collapseImpl(Node *node, const Reason *why, Worklist &worklist);
  void markArrayImpl(Node *node, uint64_t elemSize, const Reason *why,
                     Worklist &worklist);
  void insertLink(Node *node, int64_t offset, Cell target, Worklist &worklist);
  void insertGhost(Node *node, int64_t offset, GhostField field, Cell target,
                   Worklist &worklist);
  /// Re-key the ghost links of \p node after its normalization changed
  /// (collapse, markArray).
  void renormalizeGhosts(Node *node, Worklist &worklist);
  void collapseIfMisaligned(Node *node, const Reason *why, Worklist &worklist);
  bool isMisaligned(const Node *node) const;
  static void recordReason(Node *node, NodeFlags bit, const Reason *why);

  unsigned pointerSize;
  bool trackShared = false;
  /// Cells whose nodes are to become shared at the next flushShared: the
  /// targets of links added to shared nodes, and the survivors of merges
  /// with a shared node.
  std::vector<Cell> sharedFrontier;
  std::unique_ptr<ReasonTable> ownedReasons;
  ReasonTable *reasonTable;
  llvm::SpecificBumpPtrAllocator<Node> nodeAllocator;
  std::vector<Node *> allNodes;
};

} // namespace cir::pointsto

#endif // CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_POINTSTOGRAPH_H
