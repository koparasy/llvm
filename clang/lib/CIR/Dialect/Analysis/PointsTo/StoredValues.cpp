//===- StoredValues.cpp - Stored-value queries over points-to cells -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The stored-value rule (plan 2.6). What a cell (node, offset) may hold, read
// as `size` bytes, is the union of the writes that may reach it:
//
//  - Writes. Every op of an analysed function that writes memory: cir.store
//    (a scalar, or an aggregate whose range covers the cell), cir.copy,
//    cir.libc.memcpy and cir.libc.memmove (a copy from a source cell),
//    cir.libc.memset (a byte pattern), the atomics, cir.set_bitfield and any
//    other op with a write effect (an unknown value). The models' writes
//    (TUFunction::modelWrites: unknown values; modelTransfers: copies). The
//    initial value of every global (a definition without one is zero).
//
//  - Where. A write is indexed at its node and at every ancestor of it in its
//    tree of subobjects, located in each (WriteLocation). A cell compares
//    with a write in each node they share: disjoint ranges in an exact view
//    rule the write out; the same range gives the write's value; a write
//    whose range contains the cell gives the part of it at the cell's offset
//    (an aggregate store read from its source, a copy read from its source, a
//    constant extracted); a partial overlap makes the answer incomplete. On an
//    array node offsets are element-relative, so a write to one element counts
//    for every element. On a collapsed node (or across an array or collapsed
//    ancestor) a write may land at any offset of the node: it counts if it
//    may overlap, and only a write of the same size as the read gives its
//    value. SOUNDNESS of that last rule: a read of T at an address where a
//    write of a same-size object did not start reads an object whose storage
//    the write reused, which ended its lifetime ([basic.life]p1, p8): UB, so a
//    UB-free program reads the value of a write that starts where it reads, or
//    of no write at all. A different size, or a copy or memset whose offset
//    in the node is not known, gives an incomplete answer.
//
//  - Contexts (BU mode). A cell of a caller's graph that is the copy of a
//    node of a callee's graph (a TUClone) also gets the writes of the
//    callee's graph to that node (StoredValueIndex::getInverses), with the
//    clone on the walk's chain. The chain turns a formal of the callee,
//    stored by the callee, into the actual operand at the call of the clone,
//    and maps the cells a callee's load or copy reads out into the caller's
//    graph (mapThroughClone, faithful images only; else the load's value is
//    kept as a def, and a copy makes the answer incomplete). A callee bound
//    to its own graph (in an SCC, over budget) or the CI mode share the nodes:
//    their writes are found at the node itself.
//
//  - Unknown writers. The answer is incomplete if code we don't see may write
//    the cell: its node (or a subobject overlapping the read) is External,
//    Incomplete, an UnknownObj, HasUSM or AsyncAccess, a Ghost or a Function;
//    or (BU) it is reachable from the formals or the return cell of a function
//    some caller copied the graph of (the caller writes its copy, not this
//    node; the top-down query asks the callers); or it is shared (reachable
//    from a global) but was copied before it became shared, so writes to the
//    copies are not on it. A model write of unknown value (ModelWrite)
//    overlapping the cell makes the answer incomplete too.
//
//  - The indeterminate-value rule. A stack or heap object starts with an
//    indeterminate value, and reading one is UB (C++20 [basic.indet]p2, C17
//    6.7.9p10 and 7.22.3.4p2 for malloc), so "never written" contributes no
//    value: a cell no write reaches is complete and empty. A global starts
//    with its initializer or zero ([basic.start.static]p2), always a write; a
//    declaration (`external` with no initializer) is External, an unknown
//    writer. With PointsToAnalysisOptions::assumeIndeterminateReadsUB off
//    (N41: the rule is not sound for C, where an indeterminate value may be
//    copied and reading one of an address-taken object is unspecified, nor
//    for C++26 erroneous behaviour), the initial value of an object with a
//    stack or heap origin is a possible value the analysis does not know: a
//    read of one is incomplete unless a strong update (below) kills it.
//
//  - Values. A written value defined by cir.const contributes the constant
//    (an integer of another signedness of the same width is reinterpreted;
//    another type makes the answer incomplete); an integral cast of a
//    constant is folded; a value loaded from memory contributes what that
//    memory may hold if that answer is complete (a spill slot), and the load
//    itself as a def otherwise; any other value is a def. A def stands for
//    the run-time values of an SSA value, possibly of another function.
//
//  - Budget. A query visits at most maxQueryVisits cells (copies and loads
//    recursing, nodes walked through the copies); past it the answer is
//    incomplete, Reason Budget. A cycle of copies is incomplete too.
//
// The answer above is flow-insensitive (the union over the whole program),
// which is sound for every point.
//
// Strong updates (T24). A read at a query point `at` (a load, a copy reading
// its source, a launch reading the closure) gets the value of one write W
// alone when W must have stored the bytes read last before `at`:
//
//  1. The object. The nodes of the read (the node, its ancestors and the
//     subobjects overlapping it) are clean: none is External, Incomplete,
//     UnknownObj, HasUSM, AsyncAccess, Ghost, Function, Global, Heap,
//     Summary, MayBeShared or MayOverlap, shared, copied-shared or reachable
//     by a copied function's callers. The origins among them are one
//     cir.alloca of the function F holding `at`, in no calling context
//     (Origin::ctx 0), and F is not recursive. So (markSummaries) the node
//     stands for at most one live object: the alloca's object of the running
//     activation of F.
//  2. The read. `at` reads through an address derived from that alloca's
//     result (by get_member, get_element, ptr_stride and pointer casts). Or
//     the read is a callee's, mapped out through copies of a calling context
//     (readOut, "lifted"): it happens during the calls of those copies, so
//     the query point is the call of the copy whose caller is F (a call op of
//     F itself, not one a copy inherited): the outermost one, or, when step
//     3's exception walks down the copies, the one it reaches (walking down
//     any other copy ends the search: that callee's activation is over). A
//     call as the query point must not itself be a write of step 3.
//  3. The writes. Every write the index has that may overlap the read is an
//     op of F: a store, a copy, an op writing an unknown value, or the call of
//     a model writing (ModelWrite, MemTransferRecord). A write of another
//     function (on a shared CI node), or a copy of a callee's graph into the
//     read's nodes (an Inverse) whose graph, or a graph copied into it
//     recursively, has a write overlapping the read as cellValues' walk
//     locates it, makes the rule not apply; a copy with no such write (a
//     callee that only reads the object) adds nothing. One exception: if the
//     read's nodes have no write of their own and one copy's graph writes
//     them, the rule is tried in that graph, with that copy on the chain (the
//     read is of a copy of the callee's local, e.g. a SYCL closure built in a
//     CGF lambda).
//  4. W. A cir.store of exact location, in a node that is not collapsed or an
//     array, that equals or contains the read (so it writes every byte read),
//     through an address derived from the same alloca.
//  5. The path. W is found by a backward walk from `at` over structured CIR
//     (the region-based form, not the flattened CFG): the ops before `at` in
//     its block, then before its parent op in the enclosing block, and so
//     on. The first op met that is a write of step 3 must be W; any other
//     such write, an op with regions holding one, or a cir.lifetime.start or
//     .end of the alloca, ends the search (no strong update). Going up out of
//     a cir.scope, cir.if or cir.ternary, whose region runs at most once per
//     execution of the op and starts at the top of its one block, only the
//     ops before the op matter; going up out of any other region op (a loop,
//     a switch, a try, an op we don't know) the whole op must hold no such
//     write, since its regions may run again before `at`. A region with more
//     than one block, or reaching F's body without W, ends the search. So
//     does a function with cir.goto, cir.label, cir.indirect_br or cir.await,
//     or a call of setjmp-like (returns twice) functions, and the search
//     budget (maxStrongScanOps).
//
// SOUNDNESS. By 1, every write of step 3 that reaches the object, and the
// read, are to the one object of the running activation of F: an object of
// another activation is dead (a stack object of a non-recursive function that
// is not re-entered, markSummaries), so writing or reading it is UB
// ([basic.life]); that holds for a read through memory (2, lifted) too. By
// the rules of the flow-insensitive answer, the writes that may reach the
// bytes read are the index's writes on the read's nodes and those in the
// graphs copied into them (cellValues' walk), so by 3 every one is an op of
// F. Calls between W and `at` write the object only through those: a model's
// writes are ops of F (the call), a bound callee's writes reach it only
// through a copy of its graph (with a write, excluded by 3) or through a
// shared node (excluded by 1), and code we don't see only through an
// External node (excluded by 1); F is not recursive, so no other activation
// of F runs its own writes on the object meanwhile. A lifted read happens
// during the call `at`: the callee's writes would be in its copied graph
// (excluded by 3) and the call's own model writes are excluded by 2, so the
// callee reads what the object holds when the call starts. Concurrency: a
// std::thread, a host task or an async handoff is a call of code we don't
// see (or a model marking AsyncAccess), so an object it could reach is
// External or AsyncAccess, excluded by 1. Then step 5 shows that every
// execution of `at` is preceded, in the same activation, by an execution of
// W after which no write of step 3 executes before `at`: W precedes the
// ancestor of `at` in a block entered only at its top, and every op that runs
// between them is either before that ancestor in a block the walk scanned or
// inside an op the walk checked whole; without gotos, labels, indirect
// branches, coroutine suspension or setjmp, control reaches `at` from W only
// along those ops (a break, continue or return leaves the path; a loop
// re-entering the block runs W again). A lifetime marker between them would
// start a new lifetime with an indeterminate value. So the bytes read hold
// W's value (written by its last execution), and W's value is resolved like
// any other write's, which is sound for each of its executions. With the
// chain of step 3's exception, the object is the callee's local in the
// context of that copy: the caller's graph adds no write to it, and the
// argument above holds in the callee's activation.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/PointsTo/StoredValues.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "clang/CIR/Dialect/Analysis/CIRPointerDecomposition.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/OpClassification.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToAnalysis.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Interfaces/CIROpInterfaces.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

#include <functional>

using namespace mlir;
using namespace cir;
using namespace cir::pointsto;

static constexpr uint64_t kUnknownSize = PointsToAnalysis::kUnknownSize;
static constexpr int64_t kUnbounded = WriteLocation::kUnbounded;

//===----------------------------------------------------------------------===//
// StoredValues
//===----------------------------------------------------------------------===//

void StoredValues::addConst(TypedAttr value) {
  if (!llvm::is_contained(consts, value))
    consts.push_back(value);
}

void StoredValues::addDef(Value value) {
  if (!llvm::is_contained(defs, value))
    defs.push_back(value);
}

void StoredValues::setIncomplete(const Reason *reason) {
  if (complete || !why)
    why = reason;
  complete = false;
}

void StoredValues::merge(const StoredValues &other) {
  for (TypedAttr c : other.consts)
    addConst(c);
  for (Value v : other.defs)
    addDef(v);
  if (!other.complete)
    setIncomplete(other.why);
  if (!defWhy)
    defWhy = other.defWhy;
}

//===----------------------------------------------------------------------===//
// Locations
//===----------------------------------------------------------------------===//

WriteLocation StoredValueIndex::makeLocation(const Node *node, int64_t offset,
                                             uint64_t size) {
  if (node->isCollapsed())
    return {0, kUnbounded, false};
  if (std::optional<uint64_t> elem = node->getArrayElemSize()) {
    if (size != kUnknownSize && offset >= 0 &&
        static_cast<uint64_t>(offset) + size <= *elem)
      return {offset, offset + static_cast<int64_t>(size), true};
    return {0, kUnbounded, false};
  }
  if (size == kUnknownSize || size > static_cast<uint64_t>(INT64_MAX / 2))
    return {offset, kUnbounded, true};
  return {offset, offset + static_cast<int64_t>(size), true};
}

bool StoredValueIndex::locate(
    Cell cell, uint64_t size,
    SmallVectorImpl<std::pair<const Node *, WriteLocation>> &out,
    bool anywhere) const {
  size_t maxAncestors = analysis.options.maxAncestors;
  Cell at = cell.resolve();
  const Node *base = at.getNode();
  out.push_back({base, anywhere ? WriteLocation{0, kUnbounded, false}
                                : makeLocation(base, at.getOffset(), size)});
  const Graph &graph = analysis.getGraph();
  size_t i = 0;
  for (; i < out.size() && out.size() < maxAncestors; ++i) {
    auto [node, loc] = out[i];
    for (const ParentLink &link : graph.getParents(node)) {
      WriteLocation up;
      if (loc.exact && !node->isCollapsed() && !node->isArray()) {
        int64_t shift = link.offset - link.childOffset;
        up = {loc.lo + shift,
              loc.hi == kUnbounded ? kUnbounded : loc.hi + shift, true};
      } else {
        up = {link.offset, link.offset + static_cast<int64_t>(link.size),
              false};
      }
      if (link.parent->isCollapsed())
        up = {0, kUnbounded, false};
      // SOUNDNESS (G3, Csmith seed 407679): a node with several parents (two
      // subobjects merged) is reached at several places of a common
      // ancestor; the access may be at any of them, so each is kept.
      if (llvm::any_of(out, [&](const auto &entry) {
            return entry.first == link.parent && entry.second.lo == up.lo &&
                   entry.second.hi == up.hi && entry.second.exact == up.exact;
          }))
        continue;
      out.push_back({link.parent, up});
    }
  }
  // Past maxAncestors entries, some ancestors may be missing.
  return i == out.size();
}

namespace {
/// How a write relates to a read, in one node.
enum Relation : int {
  /// No byte in common.
  Disjoint = 0,
  /// The same bytes.
  Equal = 1,
  /// The write's bytes contain the read's, which start `delta` bytes in.
  Contains = 2,
  /// Some bytes in common, not all.
  Partial = 3,
  /// The offsets are not known: they may overlap.
  Anywhere = 4,
};
} // namespace

static Relation compare(const WriteLocation &w, const WriteLocation &q,
                        int64_t &delta) {
  if (w.hi <= q.lo || q.hi <= w.lo)
    return Disjoint;
  if (!w.exact || !q.exact)
    return Anywhere;
  if (w.lo == q.lo && w.hi == q.hi)
    return Equal;
  if (w.lo <= q.lo && q.hi <= w.hi) {
    delta = q.lo - w.lo;
    return Contains;
  }
  return Partial;
}

namespace {
/// How a write relates to a read over the nodes they share (relateWrites).
struct WriteRelation {
  /// The relations (with their deltas) the read may have to the write: one
  /// for a node the read is at one place of, several if it may be at
  /// several. Empty: disjoint.
  SmallVector<std::pair<int, int64_t>, 2> rels;
};
} // namespace

/// The relation of every write indexed at \p nodes (the read's nodes, each
/// with every location the read may have in it) to the read. In one node,
/// the read may be at any of its locations and the write at any of its
/// entries: the relation is the union over the pairs (disjoint only if every
/// pair is). Over the nodes, the views of one access agree up to precision
/// (an exact view is no less precise than an inexact one; a node with
/// several locations is the union of the possibilities), so a disjoint view
/// rules the write out and otherwise the most precise one is used: the one
/// whose worst relation is the least, then the one with fewest relations.
static void relateWrites(const StoredValueIndex &index,
                         ArrayRef<std::pair<const Node *, WriteLocation>> nodes,
                         llvm::MapVector<unsigned, WriteRelation> &out) {
  out.clear();
  // The read's locations per node, in order.
  llvm::MapVector<const Node *, SmallVector<WriteLocation, 1>> reads;
  for (auto [node, qloc] : nodes)
    reads[node].push_back(qloc);
  llvm::DenseSet<unsigned> disjoint;
  for (auto &[node, qlocs] : reads) {
    llvm::MapVector<unsigned, WriteRelation> here;
    for (const StoredValueIndex::Entry &entry : index.getEntries(node)) {
      WriteRelation &rel = here[entry.write];
      for (const WriteLocation &qloc : qlocs) {
        int64_t delta = 0;
        int r = compare(entry.loc, qloc, delta);
        if (r == Disjoint)
          continue;
        if (r != Contains)
          delta = 0;
        if (!llvm::is_contained(rel.rels, std::make_pair(r, delta)))
          rel.rels.push_back({r, delta});
      }
    }
    for (auto &[id, rel] : here) {
      if (rel.rels.empty()) {
        disjoint.insert(id);
        out[id].rels.clear();
        continue;
      }
      if (disjoint.contains(id))
        continue;
      auto worst = [](const WriteRelation &r) {
        int w = Disjoint;
        for (auto [k, d] : r.rels)
          w = std::max(w, k);
        return w;
      };
      auto [it, inserted] = out.try_emplace(id, rel);
      if (!inserted && (worst(rel) < worst(it->second) ||
                        (worst(rel) == worst(it->second) &&
                         rel.rels.size() < it->second.rels.size())))
        it->second = rel;
    }
  }
}

//===----------------------------------------------------------------------===//
// StoredValueIndex
//===----------------------------------------------------------------------===//

StoredValueIndex::StoredValueIndex(const PointsToAnalysis &analysis)
    : analysis(analysis) {
  indexWrites();
  indexInverses();
  indexCallerWriters();
}

ArrayRef<StoredValueIndex::Entry>
StoredValueIndex::getEntries(const Node *node) const {
  auto it = entries.find(node);
  return it == entries.end() ? ArrayRef<Entry>() : ArrayRef<Entry>(it->second);
}

ArrayRef<StoredValueIndex::Inverse>
StoredValueIndex::getInverses(const Node *node) const {
  auto it = inverses.find(node);
  return it == inverses.end() ? ArrayRef<Inverse>()
                              : ArrayRef<Inverse>(it->second);
}

const Reason *StoredValueIndex::getCallerWriter(const Node *node) const {
  return callerWriters.lookup(node);
}

void StoredValueIndex::addWrite(WriteRecord write) {
  if (!write.dst)
    return;
  unsigned id = writes.size();
  SmallVector<std::pair<const Node *, WriteLocation>, 4> where;
  // SOUNDNESS (budget maxAncestors): a write not indexed at every ancestor
  // would be missed by a read of one; then no read is complete.
  if (!locate(write.dst, write.size, where, write.anywhere) && !globalUnknown)
    globalUnknown =
        analysis.getGraph().makeReason(Reason::Kind::Budget, write.op,
                                       "a write with too many ancestors to "
                                       "index (-max-ancestors)");
  for (auto [node, loc] : where) {
    entries[node].push_back({id, loc});
    ++numEntries;
  }
  writes.push_back(std::move(write));
}

void StoredValueIndex::indexWrites() {
  const DataLayout &dataLayout = *analysis.dataLayout;
  const Graph &graph = analysis.getGraph();
  auto sizeOf = [&](Type type) -> uint64_t {
    std::optional<int64_t> size = getTypeSizeInBytes(type, dataLayout);
    return size && *size >= 0 ? static_cast<uint64_t>(*size) : kUnknownSize;
  };
  auto unknownWhy = [&](Operation *op, StringRef detail) {
    return graph.makeReason(Reason::Kind::UnhandledOp, op, detail);
  };
  // The cell \p addr points to; a missing cell of a non-null address makes
  // every query incomplete (it should not happen).
  auto cellOf = [&](Operation *op, Value addr) -> Cell {
    Cell cell = analysis.cellOf(addr);
    if (!cell && !isNullPointerConstant(addr) && !globalUnknown)
      globalUnknown = unknownWhy(op, "written through an address with no cell");
    return cell;
  };
  auto unknownAt = [&](const TUFunction *fn, Operation *op, Value addr) {
    ++unknownWriteOps[op->getName().getStringRef()];
    WriteRecord write{WriteRecord::Kind::Unknown, op, cellOf(op, addr)};
    write.owner = fn;
    if (auto ptr = dyn_cast<cir::PointerType>(addr.getType()))
      write.size = sizeOf(ptr.getPointee());
    else
      write.size = kUnknownSize;
    write.anywhere = write.size == kUnknownSize;
    addWrite(write);
  };

  for (const std::unique_ptr<TUFunction> &fn : analysis.getFunctions()) {
    cast<cir::FuncOp>(fn->func).walk([&](Operation *op) {
      if (auto store = dyn_cast<cir::StoreOp>(op)) {
        WriteRecord write{WriteRecord::Kind::Store, op,
                          cellOf(op, store.getAddr())};
        write.size = sizeOf(store.getValue().getType());
        write.value = store.getValue();
        write.owner = fn.get();
        if (write.size == kUnknownSize)
          write.kind = WriteRecord::Kind::Unknown, write.anywhere = true;
        addWrite(write);
        return;
      }
      if (auto copy = dyn_cast<cir::CopyOp>(op)) {
        WriteRecord write{WriteRecord::Kind::Copy, op,
                          cellOf(op, copy.getDst())};
        write.owner = fn.get();
        write.copySrc = analysis.cellOf(copy.getSrc());
        if (getTypeSizeInBytes(copy.getType().getPointee(), dataLayout))
          write.size = copy.getCopySizeInBytes(dataLayout);
        else
          write.kind = WriteRecord::Kind::Unknown, write.anywhere = true,
          write.size = kUnknownSize;
        addWrite(write);
        return;
      }
      if (isa<cir::MemCpyOp, cir::MemMoveOp>(op)) {
        Value dst, src, len;
        if (auto mc = dyn_cast<cir::MemCpyOp>(op))
          dst = mc.getDst(), src = mc.getSrc(), len = mc.getLen();
        else if (auto mm = dyn_cast<cir::MemMoveOp>(op))
          dst = mm.getDst(), src = mm.getSrc(), len = mm.getLen();
        WriteRecord write{WriteRecord::Kind::Copy, op, cellOf(op, dst)};
        write.owner = fn.get();
        write.copySrc = analysis.cellOf(src);
        std::optional<int64_t> n = getConstantIndex(len);
        write.size = n && *n >= 0 ? static_cast<uint64_t>(*n) : kUnknownSize;
        if (write.size == 0)
          return;
        addWrite(write);
        return;
      }
      if (auto set = dyn_cast<cir::MemSetOp>(op)) {
        WriteRecord write{WriteRecord::Kind::Memset, op,
                          cellOf(op, set.getDst())};
        write.owner = fn.get();
        std::optional<int64_t> n = getConstantIndex(set.getLen());
        write.size = n && *n >= 0 ? static_cast<uint64_t>(*n) : kUnknownSize;
        if (write.size == 0)
          return;
        if (std::optional<int64_t> byte = getConstantIndex(set.getVal()))
          write.pattern = static_cast<uint8_t>(*byte);
        addWrite(write);
        return;
      }
      // Calls: the bodies they are bound to (found through the copies, or at
      // the shared nodes), their models (TUFunction::modelWrites), or the
      // unknown-call rule (External).
      if (isa<CIRCallOpInterface>(op) || isa<cir::LoadOp>(op))
        return;
      // SOUNDNESS: the lifetime markers write no value. After
      // cir.lifetime.start the object's value is indeterminate, and after
      // cir.lifetime.end it is gone; reading it in either state is UB
      // ([basic.indet]p2, [basic.life]p7), so neither adds a value a
      // UB-free read can see.
      if (isa<cir::LifetimeStartOp, cir::LifetimeEndOp>(op))
        return;
      // Any other op that writes memory writes an unknown value: the
      // atomics, cir.set_bitfield, cir.clear_padding, and the ops this file
      // does not know.
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(op)) {
        SmallVector<MemoryEffects::EffectInstance> list;
        effects.getEffects(list);
        for (const MemoryEffects::EffectInstance &effect : list) {
          if (!isa<MemoryEffects::Write>(effect.getEffect()))
            continue;
          Value value = effect.getValue();
          if (value && isPointerValueType(value.getType())) {
            unknownAt(fn.get(), op, value);
          } else if (!value) {
            // A write to no value in particular: every pointer operand.
            for (Value operand : op->getOperands())
              if (isPointerValueType(operand.getType()))
                unknownAt(fn.get(), op, operand);
          }
        }
        return;
      }
      if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
        return; // The ops in its regions are walked.
      // No interface. An op LocalBuilder handles writes memory only if its
      // rule says so (OpClassification.cpp: the atomics, the bitfield store,
      // clear_padding, and the intrinsics it lets through); a call-like op
      // (cir.delete_array) writes through its synthetic call sites, and a
      // region op through the ops in its regions, which are walked.
      std::optional<OpClass> cls = lookupOpClass(op->getName().getStringRef());
      if (cls == OpClass::Handled &&
          !isa<cir::AtomicClearOp, cir::AtomicCmpXchgOp, cir::AtomicFetchOp,
               cir::AtomicTestAndSetOp, cir::AtomicXchgOp, cir::SetBitfieldOp,
               cir::ClearPaddingOp, cir::LLVMIntrinsicCallOp>(op))
        return;
      // Any other op may write through any pointer operand.
      for (Value operand : op->getOperands())
        if (isPointerValueType(operand.getType()))
          unknownAt(fn.get(), op, operand);
    });

    for (const ModelWrite &model : fn->modelWrites) {
      if (!model.cell)
        continue;
      auto add = [&](Cell cell) {
        WriteRecord write{WriteRecord::Kind::Unknown, model.op, cell};
        write.owner = fn.get();
        write.size = kUnknownSize;
        write.anywhere = true;
        addWrite(write);
      };
      if (!model.reachable && model.size != ModelWrite::kAnywhere) {
        // The bytes [cell, cell + size) (ModelContext::writeUnknown).
        if (model.size == 0)
          continue;
        WriteRecord write{WriteRecord::Kind::Unknown, model.op, model.cell};
        write.owner = fn.get();
        write.size = model.size;
        addWrite(write);
        continue;
      }
      if (!model.reachable) {
        add(model.cell);
        continue;
      }
      llvm::DenseSet<const Node *> reach;
      analysis.closeReach(model.cell, reach);
      for (const Node *node : reach)
        add(Cell(const_cast<Node *>(node)));
    }
    for (const MemTransferRecord &record : fn->modelTransfers) {
      WriteRecord write{WriteRecord::Kind::Copy, record.op, record.dst};
      write.owner = fn.get();
      write.copySrc = record.src;
      write.size = record.length == MemTransferRecord::kUnknown ? kUnknownSize
                                                                : record.length;
      if (write.size == 0)
        continue;
      addWrite(write);
    }
  }

  // The globals: their initial values.
  // SOUNDNESS: at the cell where the symbol's storage starts. A global
  // unified into another one's node (a pointer that may point to either)
  // is at an offset in it; at offset 0 its initializer would be read for
  // the other global's bytes, and missing from its own (V4fix, Csmith seed
  // 100570).
  for (auto [sym, cell] : analysis.getSymbolCells()) {
    auto global = analysis.symbols.lookupNearestSymbolFrom<cir::GlobalOp>(
        analysis.getModule(), sym);
    if (!global)
      continue;
    WriteRecord write{WriteRecord::Kind::GlobalInit, global, cell};
    write.size = sizeOf(global.getSymType());
    if (global.isDeclaration()) {
      // Another TU defines it (External anyway).
      write.kind = WriteRecord::Kind::Unknown;
      write.anywhere = true;
    } else if (std::optional<Attribute> init = global.getInitialValue()) {
      write.init = *init;
    }
    if (write.size == kUnknownSize &&
        write.kind == WriteRecord::Kind::GlobalInit)
      write.kind = WriteRecord::Kind::Unknown, write.anywhere = true;
    addWrite(write);
  }

  if (analysis.unownedModelWrites && !globalUnknown)
    globalUnknown = graph.makeReason(Reason::Kind::EscapedTo, nullptr,
                                     "a model wrote outside any graph");
}

void StoredValueIndex::indexInverses() {
  const Graph &graph = analysis.getGraph();
  for (const std::unique_ptr<TUClone> &clone : analysis.getClones()) {
    for (const auto &[src, dst] : clone->map.entries()) {
      Cell from = Cell(const_cast<Node *>(src)).resolve();
      Cell to = dst.resolve();
      if (!to || to.getNode() == from.getNode())
        continue;
      const Node *source = from.getNode(), *copy = to.getNode();
      // A shared node that a copy was made of (it became shared after the
      // copy): writes to it through the globals are not on the copy, and
      // writes to the copy are not on it. Both read as incomplete.
      if (graph.isShared(source)) {
        copiedShared.insert(source);
        copiedShared.insert(copy);
      }
      int64_t shift = from.getOffset() - to.getOffset();
      SmallVector<Inverse, 1> &list = inverses[copy];
      if (!llvm::any_of(list, [&](const Inverse &inv) {
            return inv.clone == clone.get() && inv.source == source &&
                   inv.shift == shift;
          }))
        list.push_back({clone.get(), source, shift});
    }
  }
}

void StoredValueIndex::indexCallerWriters() {
  if (analysis.getMode() != PointsToMode::BU)
    return;
  const Graph &graph = analysis.getGraph();
  llvm::DenseSet<const TUFunction *> copied;
  for (const std::unique_ptr<TUClone> &clone : analysis.getClones())
    copied.insert(clone->callee);
  for (const TUFunction *fn : copied) {
    SmallVector<Cell, 8> roots;
    for (const std::optional<Cell> &formal : fn->formals)
      if (formal)
        roots.push_back(*formal);
    for (const std::optional<AggregateSlots> &slots : fn->formalSlots)
      if (slots)
        for (auto [off, cell] : *slots)
          if (cell)
            roots.push_back(cell);
    if (fn->returnCell)
      roots.push_back(*fn->returnCell);
    if (fn->returnSlots)
      for (auto [off, cell] : *fn->returnSlots)
        if (cell)
          roots.push_back(cell);
    if (roots.empty())
      continue;
    llvm::DenseSet<const Node *> reach;
    analysis.closeReach(roots, reach);
    const Reason *why = nullptr;
    for (const Node *node : reach) {
      // A shared node is the same node in the callers' graphs: their writes
      // are on it.
      if (graph.isShared(node) || callerWriters.contains(node))
        continue;
      if (!why)
        why = graph.makeReason(Reason::Kind::UnknownCaller, fn->func,
                               "may be written by a caller that copied the "
                               "function's graph");
      callerWriters[node] = why;
    }
  }
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

/// \p value read as \p type, or null if that changes what it means (an
/// integer of the other signedness of the same width is the same bits).
static TypedAttr reconcile(TypedAttr value, Type type) {
  if (!type || value.getType() == type)
    return value;
  auto from = dyn_cast<cir::IntAttr>(value);
  auto to = dyn_cast<cir::IntType>(type);
  if (from && to && from.getValue().getBitWidth() == to.getWidth())
    return cir::IntAttr::get(type, from.getValue());
  return {};
}

/// The constant of \p type all of whose bytes are \p byte, or null.
static TypedAttr makePattern(Type type, uint8_t byte) {
  if (!type)
    return {};
  if (auto intTy = dyn_cast<cir::IntType>(type)) {
    unsigned width = intTy.getWidth();
    if (width % 8)
      return {};
    return cir::IntAttr::get(type, APInt::getSplat(width, APInt(8, byte)));
  }
  if (auto boolTy = dyn_cast<cir::BoolType>(type)) {
    if (byte > 1)
      return {};
    return cir::BoolAttr::get(type.getContext(), byte != 0);
  }
  if (auto fpTy = dyn_cast<cir::FPTypeInterface>(type)) {
    const llvm::fltSemantics &sem = fpTy.getFloatSemantics();
    unsigned width = APFloat::getSizeInBits(sem);
    if (width % 8)
      return {};
    return cir::FPAttr::get(
        type, APFloat(sem, APInt::getSplat(width, APInt(8, byte))));
  }
  if (auto ptrTy = dyn_cast<cir::PointerType>(type)) {
    if (byte)
      return {};
    return cir::ConstPtrAttr::get(
        type, IntegerAttr::get(IntegerType::get(type.getContext(), 64), 0));
  }
  return {};
}

/// Whether the bytes [offset, offset + size) of a value of type \p type may
/// include bytes that are no member's (padding), or it can't be told. An SSA
/// value of aggregate type has no padding bytes: a cir.store of it (an LLVM
/// first-class aggregate store) leaves them undefined, unlike cir.copy, a
/// byte copy (G2, Csmith seed 309470).
static bool mayCoverPadding(Type type, int64_t offset, uint64_t size,
                            const DataLayout &dataLayout, unsigned depth = 0) {
  if (size == 0)
    return false;
  if (depth > 16)
    return true;
  std::optional<int64_t> whole = getTypeSizeInBytes(type, dataLayout);
  if (!whole || offset < 0 || offset + static_cast<int64_t>(size) > *whole)
    return true;
  int64_t end = offset + static_cast<int64_t>(size);
  if (auto rec = dyn_cast<cir::RecordType>(type)) {
    if (rec.isUnion())
      return true;
    // The members in offset order cover the range without a gap.
    int64_t at = offset;
    for (auto [i, member] : llvm::enumerate(rec.getMembers())) {
      int64_t start = rec.getElementOffset(dataLayout, i);
      std::optional<int64_t> len = getTypeSizeInBytes(member, dataLayout);
      if (!len)
        return true;
      int64_t stop = start + *len;
      if (stop <= at)
        continue;
      if (start >= end)
        break;
      if (start > at)
        return true; // A gap before this member.
      int64_t lo = std::max(at, start), hi = std::min(end, stop);
      if (mayCoverPadding(member, lo - start, static_cast<uint64_t>(hi - lo),
                          dataLayout, depth + 1))
        return true;
      at = hi;
      if (at >= end)
        return false;
    }
    return at < end;
  }
  if (auto arr = dyn_cast<cir::ArrayType>(type)) {
    Type elem = arr.getElementType();
    std::optional<int64_t> e = getTypeSizeInBytes(elem, dataLayout);
    if (!e || *e <= 0)
      return true;
    // Every element is laid out alike: an element's padding, if any part of
    // the range covers one, is covered whole or in part in some element.
    int64_t first = offset / *e, last = (end - 1) / *e;
    if (last - first >= 2)
      return mayCoverPadding(elem, 0, static_cast<uint64_t>(*e), dataLayout,
                             depth + 1);
    for (int64_t k = first; k <= last; ++k) {
      int64_t lo = std::max(offset, k * *e), hi = std::min(end, (k + 1) * *e);
      if (mayCoverPadding(elem, lo - k * *e, static_cast<uint64_t>(hi - lo),
                          dataLayout, depth + 1))
        return true;
    }
    return false;
  }
  return false; // A scalar: all its bytes are its value.
}

/// The part of the constant \p init, of type \p initType, that holds the
/// \p size bytes at \p offset, read as \p type; null if it can't be told.
static TypedAttr extractConstant(Attribute init, Type initType, int64_t offset,
                                 uint64_t size, Type type,
                                 const DataLayout &dataLayout) {
  auto sizeOf = [&](Type t) -> std::optional<int64_t> {
    return getTypeSizeInBytes(t, dataLayout);
  };
  for (unsigned depth = 0; depth < 32; ++depth) {
    std::optional<int64_t> whole = sizeOf(initType);
    if (!whole || offset < 0 || offset + static_cast<int64_t>(size) > *whole)
      return {};
    if (!init || isa<cir::ZeroAttr>(init))
      return makePattern(type, 0);
    if (isa<cir::UndefAttr, cir::PoisonAttr>(init))
      return {};
    if (isa<cir::IntAttr, cir::FPAttr, cir::BoolAttr, cir::ConstPtrAttr>(
            init)) {
      if (offset != 0 || static_cast<int64_t>(size) != *whole)
        return {};
      if (auto ptr = dyn_cast<cir::ConstPtrAttr>(init);
          ptr && !ptr.isNullValue())
        return {};
      auto typed = cast<TypedAttr>(init);
      return type ? reconcile(typed, type) : typed;
    }
    if (auto rec = dyn_cast<cir::ConstRecordAttr>(init)) {
      auto recTy = dyn_cast<cir::RecordType>(initType);
      if (!recTy || recTy.isUnion())
        return {};
      ArrayAttr members = rec.getMembers();
      bool found = false;
      for (auto [i, member] : llvm::enumerate(members)) {
        auto typed = dyn_cast<TypedAttr>(member);
        if (!typed)
          return {};
        int64_t start = recTy.getElementOffset(dataLayout, i);
        std::optional<int64_t> len = sizeOf(typed.getType());
        if (!len)
          return {};
        if (offset >= start &&
            offset + static_cast<int64_t>(size) <= start + *len) {
          init = member;
          initType = typed.getType();
          offset -= start;
          found = true;
          break;
        }
      }
      if (!found)
        return {};
      continue;
    }
    if (auto arr = dyn_cast<cir::ConstArrayAttr>(init)) {
      auto arrTy = dyn_cast<cir::ArrayType>(initType);
      if (!arrTy)
        return {};
      Type elemTy = arrTy.getElementType();
      std::optional<int64_t> elemSize = sizeOf(elemTy);
      if (!elemSize || *elemSize <= 0)
        return {};
      int64_t index = offset / *elemSize;
      int64_t inner = offset - index * *elemSize;
      if (inner + static_cast<int64_t>(size) > *elemSize)
        return {};
      Attribute elts = arr.getElts();
      int64_t numElts = 0;
      if (auto list = dyn_cast<ArrayAttr>(elts))
        numElts = list.size();
      else if (auto str = dyn_cast<StringAttr>(elts))
        numElts = str.size();
      else
        return {};
      if (index >= numElts) {
        init = nullptr; // A trailing zero.
        initType = elemTy;
        offset = inner;
        continue;
      }
      if (auto list = dyn_cast<ArrayAttr>(elts)) {
        init = list[index];
      } else {
        auto str = cast<StringAttr>(elts);
        auto charTy = dyn_cast<cir::IntType>(elemTy);
        if (!charTy)
          return {};
        init = cir::IntAttr::get(
            elemTy, APInt(charTy.getWidth(),
                          static_cast<uint8_t>(str.getValue()[index])));
      }
      initType = elemTy;
      offset = inner;
      continue;
    }
    return {};
  }
  return {};
}

//===----------------------------------------------------------------------===//
// StoredValueQuery
//===----------------------------------------------------------------------===//

StoredValueQuery::StoredValueQuery(const PointsToAnalysis &analysis,
                                   unsigned maxQueryVisits)
    : analysis(analysis), index(std::make_unique<StoredValueIndex>(analysis)),
      maxQueryVisits(maxQueryVisits) {}

StoredValueQuery::~StoredValueQuery() = default;

const Reason *StoredValueQuery::reason(Reason::Kind kind, Operation *op,
                                       StringRef detail) {
  return analysis.getGraph().makeReason(kind, op, detail);
}

StoredValues StoredValueQuery::valuesAt(Cell cell, uint64_t size, Operation *at,
                                        Type type) {
  ++stats.numQueries;
  visitsLeft = maxQueryVisits;
  overBudget = false;
  StoredValues out;
  if (!cell || size == kUnknownSize || size == 0) {
    out.setIncomplete(
        reason(Reason::Kind::UnhandledOp, at, "no cell or no access size"));
    return out;
  }
  out = cellValues(cell, size, type, {}, at);
  if (overBudget)
    ++stats.numBudgetHits;
  return out;
}

StoredValues
StoredValueQuery::valuesInContext(Cell cell, uint64_t size, Type type,
                                  ArrayRef<const TUClone *> context,
                                  Operation *at) {
  ++stats.numQueries;
  visitsLeft = maxQueryVisits;
  overBudget = false;
  StoredValues out;
  bool strong = false;
  readOut(cell, size, type, context, out, at, &strong);
  out.strong = strong;
  if (overBudget)
    ++stats.numBudgetHits;
  return out;
}

/// The nodes that hold bytes of the read [lo, hi) of \p node: it and the
/// subobjects overlapping the range (all of them on an array or collapsed
/// node). Past \p max nodes it stops and sets \p truncated.
/// SOUNDNESS (G3): a node reached twice, at different ranges (a child of two
/// parents), is walked for each range: its subobjects overlapping either are
/// holders.
static void collectHolders(
    const Node *node, int64_t lo, int64_t hi,
    SmallVectorImpl<const Node *> &out, size_t max, bool &truncated,
    llvm::DenseSet<std::tuple<const Node *, int64_t, int64_t>> &seen) {
  if (!seen.insert({node, lo, hi}).second)
    return;
  if (!llvm::is_contained(out, node)) {
    if (out.size() >= max) {
      truncated = true;
      return;
    }
    out.push_back(node);
  }
  if (seen.size() > 4 * max) {
    truncated = true;
    return;
  }
  bool all = node->isCollapsed() || node->isArray();
  for (const SubobjectLink &sub : node->getSubobjects()) {
    int64_t end = sub.offset + static_cast<int64_t>(sub.size);
    if (!all && (end <= lo || hi <= sub.offset))
      continue;
    Cell child = sub.child.resolve();
    int64_t shift = child.getOffset() - sub.offset;
    // On an array or collapsed node the range says nothing about the child.
    int64_t clo = all ? child.getOffset() : lo + shift;
    int64_t chi =
        all ? child.getOffset() + static_cast<int64_t>(sub.size) : hi + shift;
    collectHolders(child.getNode(), clo, chi, out, max, truncated, seen);
  }
}

static void collectHolders(const Node *node, int64_t lo, int64_t hi,
                           SmallVectorImpl<const Node *> &out, size_t max,
                           bool &truncated) {
  llvm::DenseSet<std::tuple<const Node *, int64_t, int64_t>> seen;
  collectHolders(node, lo, hi, out, max, truncated, seen);
}

//===----------------------------------------------------------------------===//
// Strong updates (file comment)
//===----------------------------------------------------------------------===//

/// The address \p addr is derived from by members, elements, strides and
/// pointer casts: it points into the object that one points into.
static Value addressRoot(Value addr) {
  for (unsigned depth = 0; depth < 64 && addr; ++depth) {
    Operation *def = addr.getDefiningOp();
    if (!def)
      return addr;
    if (auto member = dyn_cast<cir::GetMemberOp>(def)) {
      addr = member.getAddr();
      continue;
    }
    if (auto elem = dyn_cast<cir::GetElementOp>(def)) {
      addr = elem.getBase();
      continue;
    }
    if (auto stride = dyn_cast<cir::PtrStrideOp>(def)) {
      addr = stride.getBase();
      continue;
    }
    if (auto cast = dyn_cast<cir::CastOp>(def);
        cast && (cast.getKind() == cir::CastKind::bitcast ||
                 cast.getKind() == cir::CastKind::address_space)) {
      addr = cast.getSrc();
      continue;
    }
    return addr;
  }
  return {};
}

/// Whether \p at reads memory through an address derived from \p root.
static bool readsThrough(Operation *at, Value root) {
  if (auto load = dyn_cast<cir::LoadOp>(at))
    return addressRoot(load.getAddr()) == root;
  if (auto copy = dyn_cast<cir::CopyOp>(at))
    return addressRoot(copy.getSrc()) == root;
  if (auto mc = dyn_cast<cir::MemCpyOp>(at))
    return addressRoot(mc.getSrc()) == root;
  if (auto mm = dyn_cast<cir::MemMoveOp>(at))
    return addressRoot(mm.getSrc()) == root;
  if (isa<CIRCallOpInterface>(at))
    return llvm::any_of(at->getOperands(), [&](Value operand) {
      return isPointerValueType(operand.getType()) &&
             addressRoot(operand) == root;
    });
  return false;
}

bool StoredValueQuery::allowsStrongUpdates(Operation *func) {
  if (auto it = strongFunctions.find(func); it != strongFunctions.end())
    return it->second;
  bool ok = !analysis.isRecursive(cast<cir::FuncOp>(func));
  if (ok)
    func->walk([&](Operation *op) {
      // Control the structured walk does not see: a jump into the path, a
      // coroutine resuming at a suspension point, a second return from a
      // setjmp-like call.
      if (isa<cir::GotoOp, cir::LabelOp, cir::IndirectBrOp, cir::AwaitOp>(op)) {
        ok = false;
        return WalkResult::interrupt();
      }
      if (auto call = dyn_cast<CIRCallOpInterface>(op)) {
        auto sym =
            dyn_cast_if_present<SymbolRefAttr>(call.getCallableForCallee());
        StringRef name = sym ? sym.getLeafReference().getValue() : "";
        if (op->hasAttr("returns_twice") ||
            llvm::is_contained({"setjmp", "_setjmp", "__setjmp", "sigsetjmp",
                                "__sigsetjmp", "_setjmp3", "savectx", "vfork",
                                "getcontext"},
                               name)) {
          ok = false;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
  strongFunctions[func] = ok;
  return ok;
}

bool StoredValueQuery::calleeWrites(
    const TUClone *clone,
    ArrayRef<std::pair<const Node *, WriteLocation>> nodes, size_t depth) {
  // SOUNDNESS: past a budget, the callee may write.
  if (!visitsLeft || depth >= analysis.options.maxContextDepth)
    return true;
  --visitsLeft;
  ++stats.numVisits;
  // The nodes of the callee's graph the read's nodes are copies of, located
  // as cellValues' walk locates them.
  SmallVector<std::pair<const Node *, WriteLocation>, 4> sub;
  for (auto [node, qloc] : nodes)
    for (const StoredValueIndex::Inverse &inv : index->getInverses(node)) {
      if (inv.clone != clone)
        continue;
      WriteLocation sloc{0, kUnbounded, false};
      const Node *src = inv.source;
      bool plain = !node->isCollapsed() && !node->isArray() &&
                   !src->isCollapsed() && !src->isArray();
      if (plain && qloc.hi != kUnbounded)
        sloc = qloc.exact ? StoredValueIndex::makeLocation(
                                src, qloc.lo + inv.shift,
                                static_cast<uint64_t>(qloc.hi - qloc.lo))
                          : WriteLocation{qloc.lo + inv.shift,
                                          qloc.hi + inv.shift, false};
      sub.push_back({src, sloc});
    }
  // A write of the callee's graph, related as cellValues relates it: an
  // exact view that rules it out wins.
  llvm::MapVector<unsigned, WriteRelation> relations;
  relateWrites(*index, sub, relations);
  if (llvm::any_of(relations,
                   [](const auto &e) { return !e.second.rels.empty(); }))
    return true;
  SetVector<const TUClone *> below;
  for (auto [node, qloc] : sub)
    for (const StoredValueIndex::Inverse &inv : index->getInverses(node))
      below.insert(inv.clone);
  return llvm::any_of(below, [&](const TUClone *next) {
    return calleeWrites(next, sub, depth + 1);
  });
}

std::optional<StoredValues>
StoredValueQuery::strongValues(Cell cell, uint64_t size, Type type,
                               ArrayRef<const TUClone *> context, Operation *at,
                               ArrayRef<const TUClone *> lift) {
  const PointsToAnalysisOptions &options = analysis.options;
  if (!options.strongUpdates || index->getGlobalUnknown())
    return std::nullopt;
  if (!visitsLeft)
    return std::nullopt;
  --visitsLeft;
  ++stats.numVisits;
  const Graph &graph = analysis.getGraph();

  // 1: the nodes of the read are clean.
  auto clean = [&](const Node *node) {
    constexpr NodeFlags kNotLocal =
        NodeFlags::External | NodeFlags::Incomplete | NodeFlags::UnknownObj |
        NodeFlags::HasUSM | NodeFlags::AsyncAccess | NodeFlags::Ghost |
        NodeFlags::Function | NodeFlags::Global | NodeFlags::Heap |
        NodeFlags::Summary | NodeFlags::MayBeShared | NodeFlags::MayOverlap;
    return !hasAny(node->getFlags(), kNotLocal) && !graph.isShared(node) &&
           !index->getCallerWriter(node) && !index->isCopiedShared(node);
  };
  SmallVector<std::pair<const Node *, WriteLocation>, 4> where;
  SmallVector<const Node *, 4> holders;
  auto gather = [&](Cell read) {
    where.clear();
    holders.clear();
    read = read.resolve();
    if (!index->locate(graph.descend(read), size, where))
      return false;
    // A read at several places of a node (G3): not one object part.
    for (size_t i = 0; i < where.size(); ++i)
      for (size_t j = 0; j < i; ++j)
        if (where[i].first == where[j].first)
          return false;
    bool truncated = false;
    collectHolders(read.getNode(), read.getOffset(),
                   read.getOffset() + static_cast<int64_t>(size), holders,
                   options.maxReadHolders, truncated);
    return !truncated;
  };
  if (!gather(cell))
    return std::nullopt;
  SmallVector<const TUClone *, 8> path(context.begin(), context.end());
  // The relation of each write over the nodes of the read (as in
  // cellValues).
  llvm::MapVector<unsigned, WriteRelation> best;
  for (;;) {
    if (!llvm::all_of(holders, clean) ||
        !llvm::all_of(where, [&](const auto &e) { return clean(e.first); }))
      return std::nullopt;
    relateWrites(*index, where, best);
    bool written = llvm::any_of(
        best, [](const auto &entry) { return !entry.second.rels.empty(); });
    // The copies of callees' graphs into the read's nodes whose callees
    // (or theirs, recursively) write the read: a copy whose graph has no
    // such write adds nothing (a callee that only reads the object).
    SetVector<const TUClone *> writers;
    for (auto [node, qloc] : where)
      for (const StoredValueIndex::Inverse &inv : index->getInverses(node))
        if (!writers.contains(inv.clone) &&
            calleeWrites(inv.clone, where, path.size()))
          writers.insert(inv.clone);
    if (writers.empty())
      break;
    // 3, the exception: the read's nodes are a copy of one callee's
    // graph and have no write of their own; read in that graph.
    if (written || writers.size() > 1 || path.size() >= options.maxContextDepth)
      return std::nullopt;
    const TUClone *clone = writers.front();
    // A lifted read is during the calls of `lift`: the object can only be a
    // local of the callee of the first one (any other callee's activation
    // is over).
    if (!lift.empty()) {
      if (clone != lift.front())
        return std::nullopt;
      lift = lift.drop_front();
    }
    const Node *read = where.front().first;
    WriteLocation qloc = where.front().second;
    const StoredValueIndex::Inverse *inv = nullptr;
    for (const StoredValueIndex::Inverse &other : index->getInverses(read)) {
      if (other.clone != clone)
        continue;
      if (inv)
        return std::nullopt;
      inv = &other;
    }
    if (!inv || !qloc.exact || qloc.hi == kUnbounded || read->isCollapsed() ||
        read->isArray() || inv->source->isCollapsed() || inv->source->isArray())
      return std::nullopt;
    // Every node of the read whose copy came from the callee must be among
    // the nodes of the read there.
    SmallVector<const Node *, 4> sources;
    for (auto [node, loc] : where)
      for (const StoredValueIndex::Inverse &other : index->getInverses(node))
        if (other.clone == clone)
          sources.push_back(other.source);
    const Node *source = inv->source;
    int64_t from = qloc.lo + inv->shift;
    if (!gather(Cell(const_cast<Node *>(source), from)))
      return std::nullopt;
    if (!llvm::all_of(sources, [&](const Node *src) {
          return llvm::any_of(where,
                              [&](const auto &e) { return e.first == src; });
        }))
      return std::nullopt;
    path.push_back(clone);
  }

  // The query point (2): `at`, or for a lifted read the call of the first
  // copy of `lift`, an op of that copy's caller (not one a copy inherited).
  bool lifted = !lift.empty();
  Operation *point = at;
  if (lifted) {
    const TUClone *outer = lift.front();
    point = outer->call;
    if (!point || !outer->caller ||
        point->getParentOfType<cir::FuncOp>() !=
            cast<cir::FuncOp>(outer->caller->func))
      return std::nullopt;
  }
  auto func = point->getParentOfType<cir::FuncOp>();
  if (!func || !allowsStrongUpdates(func))
    return std::nullopt;

  // 1: the one object, an alloca of F in no calling context.
  Value root;
  for (auto [node, loc] : where)
    for (const Origin &origin : node->getOrigins()) {
      if (node->getOrigins().size() != 1 ||
          origin.kind != Origin::Kind::Alloca || origin.ctx || !origin.site ||
          origin.site->getNumResults() != 1 ||
          origin.site->getParentOfType<cir::FuncOp>() != func)
        return std::nullopt;
      if (root && root != origin.site->getResult(0))
        return std::nullopt;
      root = origin.site->getResult(0);
    }
  if (!root)
    return std::nullopt;
  // 2: the read is of that object. A lifted read (readOut: a callee's read
  // during the call `at`) reaches it through memory: it is the object of
  // the running activation by 1 alone.
  if (!lifted && !readsThrough(point, root))
    return std::nullopt;

  // 3, 4: the writes that may overlap the read are ops of F; the stores of
  // exact location that cover it are the candidates for W.
  ArrayRef<WriteRecord> all = index->getWrites();
  llvm::DenseMap<Operation *, int> events; // the write id of W, or -1
  llvm::DenseMap<unsigned, std::pair<int, int64_t>> cover;
  for (auto &[id, relation] : best) {
    if (relation.rels.empty())
      continue;
    const WriteRecord &write = all[id];
    if (!write.op || write.op->getParentOfType<cir::FuncOp>() != func)
      return std::nullopt;
    bool must = false;
    if (write.kind == WriteRecord::Kind::Store)
      if (auto store = dyn_cast<cir::StoreOp>(write.op);
          store && addressRoot(store.getAddr()) == root)
        for (auto [node, qloc] : where) {
          if (must || node->isCollapsed() || node->isArray() || !qloc.exact)
            continue;
          // A write with several entries in the node may land at any of
          // them: not a must-write.
          if (llvm::count_if(index->getEntries(node),
                             [&](const auto &e) { return e.write == id; }) != 1)
            continue;
          for (const StoredValueIndex::Entry &entry : index->getEntries(node)) {
            if (entry.write != id || !entry.loc.exact)
              continue;
            int64_t delta = 0;
            int rel = compare(entry.loc, qloc, delta);
            if (rel == Equal || rel == Contains) {
              must = true;
              cover[id] = {rel, rel == Equal ? 0 : delta};
              break;
            }
          }
        }
    auto [it, inserted] =
        events.try_emplace(write.op, must ? static_cast<int>(id) : -1);
    if (!inserted)
      it->second = -1; // Two writes by one op.
  }

  // `at` itself writing the read (a call, or a lifted read in the middle
  // of a call whose model writes it): its own writes may come first.
  if (!isa<cir::LoadOp>(point) && events.contains(point))
    return std::nullopt;

  // 5: the backward walk from `at`.
  unsigned budget = options.maxStrongScanOps;
  bool ignore = options.testIgnoreStrongClobbers;
  // The lifetime markers of the alloca: on addresses derived from it (the
  // users of its result through the ops addressRoot looks through).
  llvm::SmallPtrSet<Operation *, 4> lifetimes;
  {
    SmallVector<Value, 8> work{root};
    llvm::SmallPtrSet<Operation *, 16> seen;
    while (!work.empty()) {
      Value v = work.pop_back_val();
      for (Operation *user : v.getUsers()) {
        if (!seen.insert(user).second)
          continue;
        if (isa<cir::LifetimeStartOp, cir::LifetimeEndOp>(user))
          lifetimes.insert(user);
        else if (isa<cir::GetMemberOp, cir::GetElementOp, cir::PtrStrideOp>(
                     user) ||
                 (isa<cir::CastOp>(user) &&
                  addressRoot(user->getResult(0)) == root))
          if (user->getNumResults() == 1 && user->getOperand(0) == v)
            work.push_back(user->getResult(0));
      }
    }
  }
  bool overScan = false;
  // Whether an op nested in \p top writes the read (W included: a nested W
  // does not dominate `at`), or is a lifetime marker of the alloca: one of
  // those ops has \p top as a proper ancestor.
  auto writesIn = [&](Operation *top) {
    auto inside = [&](Operation *op) {
      if (!budget) {
        overScan = true;
        return true;
      }
      --budget;
      return op != top && top->isAncestor(op);
    };
    return llvm::any_of(
               events,
               [&](const auto &entry) { return inside(entry.first); }) ||
           llvm::any_of(lifetimes, inside);
  };
  auto giveUp = [&]() -> std::optional<StoredValues> {
    if (overScan)
      ++stats.numStrongBudgetHits;
    return std::nullopt;
  };
  int found = -1;
  for (Operation *cur = point; found < 0;) {
    for (Operation *op = cur->getPrevNode(); op; op = op->getPrevNode()) {
      if (!budget) {
        overScan = true;
        return giveUp();
      }
      --budget;
      if (auto it = events.find(op); it != events.end()) {
        if (it->second >= 0) {
          found = it->second;
          break;
        }
        if (!ignore)
          return giveUp();
        continue;
      }
      if ((lifetimes.contains(op) || (op->getNumRegions() && writesIn(op))) &&
          (overScan || !ignore))
        return giveUp();
    }
    if (found >= 0)
      break;
    Region *region = cur->getParentRegion();
    Operation *parent = region ? region->getParentOp() : nullptr;
    if (!parent || !region->hasOneBlock() || isa<cir::FuncOp>(parent))
      return giveUp();
    if (!isa<cir::ScopeOp, cir::IfOp, cir::TernaryOp>(parent) &&
        (events.contains(parent) || writesIn(parent)) && (overScan || !ignore))
      return giveUp();
    cur = parent;
  }

  auto [rel, delta] = cover.lookup(static_cast<unsigned>(found));
  StoredValues out;
  applyWrite(all[found], rel, delta, size, type, path, out);
  // An incomplete strong answer is no better than the union.
  if (!out.complete)
    return std::nullopt;
  out.strong = true;
  ++stats.numStrong;
  return out;
}

StoredValues StoredValueQuery::cellValues(Cell cell, uint64_t size, Type type,
                                          ArrayRef<const TUClone *> context,
                                          Operation *at,
                                          ArrayRef<const TUClone *> lift) {
  Cell resolved = cell.resolve();
  const Node *base = resolved.getNode();
  if (at)
    if (std::optional<StoredValues> strong =
            strongValues(resolved, size, type, context, at, lift))
      return *strong;
  uint64_t chainHash = context.empty()
                           ? 0
                           : static_cast<uint64_t>(llvm::hash_combine_range(
                                 context.begin(), context.end())) |
                                 1;
  Key key{base, resolved.getOffset(), size,
          type ? type.getAsOpaquePointer() : static_cast<const void *>(nullptr),
          chainHash};
  if (auto it = memo.find(key); it != memo.end())
    return it->second;
  StoredValues out;
  if (inProgress.contains(key)) {
    out.setIncomplete(
        reason(Reason::Kind::MergedAt, nullptr, "a cycle of copies"));
    return out;
  }
  if (!visitsLeft) {
    overBudget = true;
    out.setIncomplete(
        reason(Reason::Kind::Budget, nullptr, "stored-value query visits"));
    return out;
  }
  --visitsLeft;
  ++stats.numVisits;
  inProgress.insert(key);
  bool budgetBefore = overBudget;

  if (const Reason *why = index->getGlobalUnknown())
    out.setIncomplete(why);

  // Unknown writers of the bytes read.
  SmallVector<const Node *, 4> holders;
  int64_t lo = resolved.getOffset();
  bool truncatedHolders = false;
  collectHolders(base, lo, lo + static_cast<int64_t>(size), holders,
                 analysis.options.maxReadHolders, truncatedHolders);
  // SOUNDNESS (budget maxReadHolders): an unknown writer or a write of a
  // holder left out would be missed.
  if (truncatedHolders) {
    ++stats.numHolderHits;
    out.setIncomplete(reason(Reason::Kind::Budget, nullptr,
                             "a read of too many subobjects "
                             "(-max-read-holders)"));
  }
  for (const Node *node : holders) {
    NodeFlags flags = node->getFlags();
    constexpr NodeFlags kUnknownWriter =
        NodeFlags::External | NodeFlags::Incomplete | NodeFlags::UnknownObj |
        NodeFlags::HasUSM | NodeFlags::AsyncAccess | NodeFlags::Ghost |
        NodeFlags::Function;
    if (hasAny(flags, kUnknownWriter)) {
      const Reason *why = node->getReason(NodeFlags::Incomplete);
      if (!why)
        why = node->getReason(NodeFlags::External);
      if (!why)
        why = reason(Reason::Kind::EscapedTo, nullptr,
                     "written by code we don't see (" +
                         toString(flags & kUnknownWriter) + ")");
      out.setIncomplete(why);
    }
    if (const Reason *why = index->getCallerWriter(node))
      out.setIncomplete(why);
    // MayBeShared (BU): the node may be a global's object, which every
    // function may write through the global; those writes are indexed on
    // the shared node, not on this one.
    if (node->hasFlags(NodeFlags::MayBeShared) &&
        !analysis.getGraph().isShared(node))
      out.setIncomplete(node->getReason(NodeFlags::MayBeShared));
    // MayOverlap (BU, G2): in a caller's context the node may be a part of
    // another node's object, whose writes are indexed on that node.
    if (node->hasFlags(NodeFlags::MayOverlap))
      out.setIncomplete(node->getReason(NodeFlags::MayOverlap));
    if (index->isCopiedShared(node))
      out.setIncomplete(reason(Reason::Kind::MergedAt, nullptr,
                               "shared node copied before it was shared"));
  }

  // The writes, also when an unknown writer already made the answer
  // incomplete: the constants seen still explain it.
  {
    // From the deepest node that holds the first byte read: a write to a
    // subobject is indexed at it and at its ancestors, not at the nodes
    // below the one it names.
    SmallVector<std::pair<const Node *, WriteLocation>, 4> where;
    // SOUNDNESS (budget maxAncestors): the writes indexed at a missing
    // ancestor would be missed.
    if (!index->locate(analysis.getGraph().descend(resolved), size, where)) {
      ++stats.numAncestorHits;
      out.setIncomplete(reason(Reason::Kind::Budget, nullptr,
                               "a read with too many ancestors "
                               "(-max-ancestors)"));
    }
    // The indeterminate-value rule off (file comment): the initial value of
    // a stack or heap object is a possible value. The object's origin is on
    // the node or on an ancestor (where), or on a subobject merged with
    // another object (holders).
    if (!analysis.options.assumeIndeterminateReadsUB) {
      auto fresh = [](const Node *node) {
        return llvm::any_of(node->getOrigins(), [](const Origin &origin) {
          return origin.kind == Origin::Kind::Alloca ||
                 origin.kind == Origin::Kind::HeapCall;
        });
      };
      if (llvm::any_of(where,
                       [&](const auto &entry) { return fresh(entry.first); }) ||
          llvm::any_of(holders, fresh))
        out.setIncomplete(reason(Reason::Kind::UnhandledOp, nullptr,
                                 "the initial value of a stack or heap "
                                 "object (indeterminate reads are not "
                                 "assumed UB)"));
    }
    SmallVector<const TUClone *, 8> chain(context.begin(), context.end());
    // One level: the writes of every node the read is in, then the copies
    // of callees' graphs below those nodes.
    std::function<void(ArrayRef<std::pair<const Node *, WriteLocation>>,
                       SmallVectorImpl<const TUClone *> &)>
        level = [&](ArrayRef<std::pair<const Node *, WriteLocation>> nodes,
                    SmallVectorImpl<const TUClone *> &path) {
          // The relation of each write over the nodes it shares with the
          // read (relateWrites). SOUNDNESS: where the read may be at several
          // places of the write (G3, seed 407679), the answer is the union
          // of what the write gives at each; past kMaxViews of them,
          // incomplete.
          constexpr size_t kMaxViews = 8;
          llvm::MapVector<unsigned, WriteRelation> relations;
          relateWrites(*index, nodes, relations);
          for (auto &[id, relation] : relations) {
            if (relation.rels.size() > kMaxViews) {
              out.setIncomplete(reason(Reason::Kind::Budget,
                                       index->getWrites()[id].op,
                                       "a write the read may be at too many "
                                       "places of"));
              continue;
            }
            for (auto [rel, delta] : relation.rels)
              applyWrite(index->getWrites()[id], rel, delta, size, type, path,
                         out);
          }
          if (path.size() >= analysis.options.maxContextDepth) {
            ++stats.numDepthHits;
            out.setIncomplete(reason(Reason::Kind::Budget, nullptr,
                                     "copies nested too deep"));
            return;
          }
          // The callees whose graphs were copied into these nodes.
          llvm::MapVector<
              const TUClone *,
              SmallVector<std::pair<const Node *, WriteLocation>, 2>>
              below;
          for (auto [node, qloc] : nodes)
            for (const StoredValueIndex::Inverse &inv :
                 index->getInverses(node)) {
              WriteLocation sloc{0, kUnbounded, false};
              const Node *src = inv.source;
              bool plain = !node->isCollapsed() && !node->isArray() &&
                           !src->isCollapsed() && !src->isArray();
              bool sameArray =
                  node->isArray() && !node->isCollapsed() &&
                  src->getArrayElemSize() == node->getArrayElemSize() &&
                  !src->isCollapsed();
              if ((plain || sameArray) && qloc.hi != kUnbounded) {
                if (qloc.exact) {
                  int64_t from = qloc.lo + inv.shift;
                  if (sameArray) {
                    int64_t elem =
                        static_cast<int64_t>(*src->getArrayElemSize());
                    from = ((from % elem) + elem) % elem;
                  }
                  sloc = StoredValueIndex::makeLocation(
                      src, from, static_cast<uint64_t>(qloc.hi - qloc.lo));
                } else if (plain) {
                  sloc = {qloc.lo + inv.shift, qloc.hi + inv.shift, false};
                }
              }
              below[inv.clone].push_back({src, sloc});
            }
          for (auto &[clone, sub] : below) {
            if (!visitsLeft) {
              overBudget = true;
              out.setIncomplete(reason(Reason::Kind::Budget, nullptr,
                                       "stored-value query visits"));
              return;
            }
            --visitsLeft;
            ++stats.numVisits;
            path.push_back(clone);
            level(sub, path);
            path.pop_back();
          }
        };
    level(where, chain);
  }

  inProgress.erase(key);
  if (context.empty() && (out.complete || overBudget == budgetBefore))
    memo[key] = out;
  return out;
}

void StoredValueQuery::applyWrite(const WriteRecord &write, int rel,
                                  int64_t delta, uint64_t size, Type type,
                                  ArrayRef<const TUClone *> chain,
                                  StoredValues &out) {
  const DataLayout &dataLayout = *analysis.dataLayout;
  auto partial = [&](StringRef detail) {
    out.setIncomplete(reason(Reason::Kind::UnhandledOp, write.op, detail));
  };
  // SOUNDNESS (file comment): a write whose offset in the node is not known
  // gives its value only if it has the size of the read.
  if (rel == Anywhere) {
    if (write.kind == WriteRecord::Kind::Unknown)
      return out.setIncomplete(reason(Reason::Kind::EscapedTo, write.op,
                                      "written with an unknown value"));
    if (write.size != size)
      return partial("may overlap the read at an unknown offset");
    if (write.kind == WriteRecord::Kind::Copy ||
        write.kind == WriteRecord::Kind::Memset)
      return partial("a byte copy at an unknown offset");
    rel = Equal;
    delta = 0;
  }
  if (rel == Partial)
    return partial("overlaps the read partially");
  switch (write.kind) {
  case WriteRecord::Kind::Unknown:
    return out.setIncomplete(reason(Reason::Kind::EscapedTo, write.op,
                                    "written with an unknown value"));
  case WriteRecord::Kind::Store: {
    // SOUNDNESS: the padding of a stored aggregate value is not written.
    if (isa<cir::RecordType, cir::ArrayType>(write.value.getType()) &&
        mayCoverPadding(write.value.getType(), rel == Equal ? 0 : delta, size,
                        dataLayout))
      return partial("the read covers padding of a stored aggregate value");
    if (rel == Equal)
      return resolveValue(write.value, size, type, chain, out);
    // The read is a part of a stored aggregate.
    Value value = write.value;
    if (auto load = value.getDefiningOp<cir::LoadOp>()) {
      Cell src = analysis.cellOf(load.getAddr());
      if (!src)
        return partial("a part of an aggregate loaded from nowhere");
      return readOut(Cell(src.getRawNode(), src.getRawOffset() + delta), size,
                     type, chain, out, load);
    }
    if (auto constant = value.getDefiningOp<cir::ConstantOp>()) {
      if (TypedAttr part = extractConstant(constant.getValue(), value.getType(),
                                           delta, size, type, dataLayout))
        return out.addConst(part);
      return partial("a part of a stored constant not understood");
    }
    return partial("a part of an aggregate value");
  }
  case WriteRecord::Kind::Copy: {
    if (!write.copySrc)
      return partial("a copy from no cell");
    return readOut(
        Cell(write.copySrc.getRawNode(), write.copySrc.getRawOffset() + delta),
        size, type, chain, out, write.op);
  }
  case WriteRecord::Kind::Memset: {
    if (!write.pattern)
      return partial("a memset of an unknown byte");
    if (TypedAttr value = makePattern(type, *write.pattern))
      return out.addConst(value);
    return partial("a memset read as a type it can't build");
  }
  case WriteRecord::Kind::GlobalInit: {
    auto global = cast<cir::GlobalOp>(write.op);
    if (TypedAttr value =
            extractConstant(write.init, global.getSymType(),
                            rel == Equal ? 0 : delta, size, type, dataLayout))
      return out.addConst(value);
    return partial("an initializer not understood");
  }
  }
}

size_t StoredValueQuery::mapOut(Cell cell, ArrayRef<const TUClone *> chain,
                                SmallVectorImpl<Cell> &out) const {
  SmallVector<Cell, 2> cells{cell.resolve()};
  size_t left = chain.size();
  for (; left; --left) {
    SmallVector<Cell, 2> next;
    if (!analysis.mapThroughClone(*chain[left - 1], cells, next,
                                  /*exact=*/true))
      break;
    cells = std::move(next);
  }
  out.append(cells.begin(), cells.end());
  return left;
}

void StoredValueQuery::readOut(Cell cell, uint64_t size, Type type,
                               ArrayRef<const TUClone *> chain,
                               StoredValues &out, Operation *at,
                               bool *allStrong) {
  // SOUNDNESS: a cell with no faithful image in a caller's graph is read in
  // the graph the walk got to, which is sound for all of its contexts
  // (PointsToTopDown.cpp, "Images"); the copies above it stay on the chain,
  // so the formals its writes store still name this context's operands. A
  // node no copy includes is one the callers cannot reach (it is not
  // reachable from the callee's formals, return or globals when copied), so
  // only the activation that made it writes it.
  //
  // The query point, for a strong update (file comment, "Strong updates"):
  // a cell mapped out to a caller's graph is read during the call of the
  // outermost copy it was mapped through; in that caller, the read happens
  // at that call.
  SmallVector<Cell, 2> cells;
  size_t left = mapOut(cell, chain, cells);
  ArrayRef<const TUClone *> lift = chain.drop_front(left);
  bool strong = !cells.empty();
  for (Cell c : cells) {
    StoredValues here =
        cellValues(c, size, type, chain.take_front(left), at, lift);
    strong &= here.strong;
    out.merge(here);
  }
  if (allStrong)
    *allStrong = strong;
}

void StoredValueQuery::resolveValue(Value value, uint64_t size, Type type,
                                    ArrayRef<const TUClone *> chain,
                                    StoredValues &out) {
  for (unsigned depth = 0; depth < 64; ++depth) {
    if (auto constant = value.getDefiningOp<cir::ConstantOp>()) {
      auto attr = dyn_cast<TypedAttr>(constant.getValue());
      if (TypedAttr c = attr ? reconcile(attr, type) : TypedAttr())
        return out.addConst(c);
      if (attr && isa<cir::ZeroAttr>(attr))
        if (TypedAttr c = makePattern(type ? type : attr.getType(), 0))
          return out.addConst(c);
      return out.addDef(value);
    }
    // An integral cast of a constant (CIRGen spells `long n = 1024` so).
    if (auto cast = value.getDefiningOp<cir::CastOp>()) {
      auto src = cast.getSrc().getDefiningOp<cir::ConstantOp>();
      auto srcTy = dyn_cast<cir::IntType>(cast.getSrc().getType());
      auto dstTy = dyn_cast<cir::IntType>(cast.getType());
      if (cast.getKind() == cir::CastKind::integral && src && srcTy && dstTy)
        if (auto attr = dyn_cast<cir::IntAttr>(src.getValue())) {
          APInt bits = srcTy.isSigned()
                           ? attr.getValue().sextOrTrunc(dstTy.getWidth())
                           : attr.getValue().zextOrTrunc(dstTy.getWidth());
          if (TypedAttr c =
                  reconcile(cir::IntAttr::get(cast.getType(), bits), type))
            return out.addConst(c);
        }
      return out.addDef(value);
    }
    // A formal of the function whose copy the walk came through: the actual
    // operand of the call of that copy.
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      auto func = dyn_cast<cir::FuncOp>(arg.getOwner()->getParentOp());
      if (!func || !arg.getOwner()->isEntryBlock() || chain.empty())
        return out.addDef(value);
      const TUClone *clone = chain.back();
      auto call = dyn_cast<CIRCallOpInterface>(clone->call);
      if (!call || clone->callee->func != func ||
          call->getParentOfType<cir::FuncOp>() != clone->caller->func ||
          arg.getArgNumber() >= call.getNumArgOperands())
        return out.addDef(value);
      // SOUNDNESS: a byval formal is the address of the callee's copy, not
      // the actual (LocalBuilder::build).
      if (func.getArgAttr(arg.getArgNumber(), "llvm.byval"))
        return out.addDef(value);
      value = call.getArgOperand(arg.getArgNumber());
      chain = chain.drop_back();
      continue;
    }
    // A spill slot: what the loaded cell holds, if that is known.
    if (auto load = value.getDefiningOp<cir::LoadOp>()) {
      Cell cell = analysis.cellOf(load.getAddr());
      if (cell) {
        StoredValues inner;
        readOut(cell, size, type ? type : value.getType(), chain, inner, load);
        if (inner.complete) {
          out.merge(inner);
          return;
        }
        if (!out.defWhy)
          out.defWhy = inner.why;
      }
      return out.addDef(value);
    }
    return out.addDef(value);
  }
  out.addDef(value);
}

//===----------------------------------------------------------------------===//
// PointsToAnalysis entry points
//===----------------------------------------------------------------------===//

StoredValueQuery &PointsToAnalysis::getStoredValueQuery() const {
  if (!storedValues)
    storedValues =
        std::make_unique<StoredValueQuery>(*this, options.maxQueryVisits);
  return *storedValues;
}

StoredValues
PointsToAnalysis::valuesAtInContexts(Value addr, std::optional<uint64_t> size,
                                     Type type, unsigned *numContexts,
                                     Operation *at, int64_t offset) const {
  if (numContexts)
    *numContexts = 0;
  StoredValues own;
  auto ptrTy = dyn_cast<cir::PointerType>(addr.getType());
  if (!type && ptrTy)
    type = ptrTy.getPointee();
  if (!size && type)
    if (std::optional<int64_t> bytes = getTypeSizeInBytes(type, *dataLayout);
        bytes && *bytes > 0)
      size = static_cast<uint64_t>(*bytes);
  Cell cell = cellOf(addr);
  if (cell && offset)
    cell = Cell(cell.getNode(), cell.getOffset() + offset);
  if (!cell || !size) {
    own.setIncomplete(
        notAnalysedWhy ? notAnalysedWhy
                       : graph->makeReason(Reason::Kind::UnhandledOp, nullptr,
                                           "no cell or no access size"));
    return own;
  }
  StoredValueQuery &query = getStoredValueQuery();
  own = query.valuesAt(cell, *size, at, type);
  if (own.complete && own.defs.empty())
    return own;
  // BU, top-down: the union over the calling contexts. SOUNDNESS: the
  // argument of PointsToTopDown.cpp. Every execution of a covered function
  // runs in a context that copied its graph, recursively up to a function
  // that is not covered; so the paths of copies from such a function down to
  // this one cover every execution, and the answer in each path's context is
  // sound for its executions (StoredValueQuery::valuesInContext reads the
  // cell as far out as it has faithful images, in graphs sound for all of
  // their contexts).
  const TUFunction *fn = getTopDownFunction(addr);
  if (!fn)
    return own;
  StoredValues all;
  all.strong = true;
  const Reason *defWhy = nullptr;
  unsigned count = 0;
  bool overBudget = false;
  SmallVector<const TUClone *, 8> path;
  std::function<void(const TUFunction &)> walk = [&](const TUFunction &fnHere) {
    if (overBudget || !all.complete)
      return;
    auto copies = coveredClones.find(&fnHere);
    bool tooDeep = path.size() >= options.maxContextDepth;
    queryStats.numDepthHits += tooDeep && copies != coveredClones.end();
    if (copies == coveredClones.end() || tooDeep) {
      if (++count > options.maxQueryContexts) {
        overBudget = true;
        return;
      }
      StoredValues here = query.valuesInContext(cell, *size, type, path, at);
      all.strong &= here.strong;
      for (TypedAttr c : here.consts)
        all.addConst(c);
      if (!here.defs.empty()) {
        // The values of a caller's context: not the query's to name.
        if (!defWhy)
          defWhy = graph->makeReason(
              Reason::Kind::UnknownCaller, here.defs.front().getDefiningOp(),
              "a value computed in a caller's context", here.defWhy);
        all.setIncomplete(defWhy);
      }
      if (!here.complete)
        all.setIncomplete(here.why);
      return;
    }
    for (const TUClone *clone : copies->second) {
      path.insert(path.begin(), clone);
      walk(*clone->caller);
      path.erase(path.begin());
    }
  };
  walk(*fn);
  ContextWalkResult result = overBudget
                                 ? ContextWalkResult::OverBudget
                                 : (all.complete ? ContextWalkResult::Done
                                                 : ContextWalkResult::Stopped);
  countContexts(result, count);
  // An incomplete union is no better than a complete own answer with defs.
  if (overBudget || (!all.complete && own.complete))
    return own;
  if (numContexts)
    *numContexts = count;
  return all;
}

std::optional<TypedAttr>
PointsToAnalysis::isConstantAt(Value addr, std::optional<uint64_t> size,
                               Type type, Operation *at, int64_t offset) const {
  StoredValues values =
      valuesAtInContexts(addr, size, type, nullptr, at, offset);
  if (!values.complete || !values.defs.empty() || values.consts.size() != 1)
    return std::nullopt;
  return values.consts.front();
}
