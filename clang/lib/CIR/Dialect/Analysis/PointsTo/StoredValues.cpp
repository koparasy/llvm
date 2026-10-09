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
//    writer.
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
// The answer is flow-insensitive (the union over the whole program), which is
// sound for every point: the `at` operation is reserved for strong updates
// (T24) and ignored.
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

#include <functional>

using namespace mlir;
using namespace cir;
using namespace cir::pointsto;

static constexpr uint64_t kUnknownSize = PointsToAnalysis::kUnknownSize;
static constexpr int64_t kUnbounded = WriteLocation::kUnbounded;
/// The deepest chain of copies a walk follows (a guard: the copies of the
/// bottom-up mode form no cycle).
static constexpr unsigned kMaxChain = 256;

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

void StoredValueIndex::locate(
    Cell cell, uint64_t size,
    SmallVectorImpl<std::pair<const Node *, WriteLocation>> &out,
    bool anywhere) const {
  constexpr size_t kMaxAncestors = 64;
  Cell at = cell.resolve();
  const Node *base = at.getNode();
  out.push_back({base, anywhere ? WriteLocation{0, kUnbounded, false}
                                : makeLocation(base, at.getOffset(), size)});
  const Graph &graph = analysis.getGraph();
  for (size_t i = 0; i < out.size() && out.size() < kMaxAncestors; ++i) {
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
      if (llvm::any_of(out, [&](const auto &entry) {
            return entry.first == link.parent;
          }))
        continue;
      out.push_back({link.parent, up});
    }
  }
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
  locate(write.dst, write.size, where, write.anywhere);
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
    auto global = dyn_cast_if_present<cir::GlobalOp>(
        SymbolTable::lookupNearestSymbolFrom(analysis.getModule(), sym));
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
  // `at` is reserved for flow sensitivity (T24): the answer is the
  // flow-insensitive one, which holds at every point.
  (void)at;
  ++stats.numQueries;
  visitsLeft = maxQueryVisits;
  overBudget = false;
  StoredValues out;
  if (!cell || size == kUnknownSize || size == 0) {
    out.setIncomplete(
        reason(Reason::Kind::UnhandledOp, at, "no cell or no access size"));
    return out;
  }
  out = cellValues(cell, size, type, {});
  if (overBudget)
    ++stats.numBudgetHits;
  return out;
}

StoredValues
StoredValueQuery::valuesInContext(Cell cell, uint64_t size, Type type,
                                  ArrayRef<const TUClone *> context) {
  ++stats.numQueries;
  visitsLeft = maxQueryVisits;
  overBudget = false;
  StoredValues out;
  readOut(cell, size, type, context, out);
  if (overBudget)
    ++stats.numBudgetHits;
  return out;
}

/// The nodes that hold bytes of the read [lo, hi) of \p node: it and the
/// subobjects overlapping the range (all of them on an array or collapsed
/// node).
static void collectHolders(const Node *node, int64_t lo, int64_t hi,
                           SmallVectorImpl<const Node *> &out) {
  if (out.size() > 256 || llvm::is_contained(out, node))
    return;
  out.push_back(node);
  bool all = node->isCollapsed() || node->isArray();
  for (const SubobjectLink &sub : node->getSubobjects()) {
    int64_t end = sub.offset + static_cast<int64_t>(sub.size);
    if (!all && (end <= lo || hi <= sub.offset))
      continue;
    Cell child = sub.child.resolve();
    int64_t shift = child.getOffset() - sub.offset;
    collectHolders(child.getNode(), lo + shift, hi + shift, out);
  }
}

StoredValues StoredValueQuery::cellValues(Cell cell, uint64_t size, Type type,
                                          ArrayRef<const TUClone *> context) {
  Cell resolved = cell.resolve();
  const Node *base = resolved.getNode();
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
  collectHolders(base, lo, lo + static_cast<int64_t>(size), holders);
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
    index->locate(analysis.getGraph().descend(resolved), size, where);
    SmallVector<const TUClone *, 8> chain(context.begin(), context.end());
    // One level: the writes of every node the read is in, then the copies
    // of callees' graphs below those nodes.
    std::function<void(ArrayRef<std::pair<const Node *, WriteLocation>>,
                       SmallVectorImpl<const TUClone *> &)>
        level = [&](ArrayRef<std::pair<const Node *, WriteLocation>> nodes,
                    SmallVectorImpl<const TUClone *> &path) {
          // The best relation of each write over the nodes it shares with
          // the read: an exact view that rules it out wins.
          llvm::DenseMap<unsigned, std::pair<int, int64_t>> best;
          SmallVector<unsigned, 8> order;
          for (auto [node, qloc] : nodes)
            for (const StoredValueIndex::Entry &entry :
                 index->getEntries(node)) {
              int64_t delta = 0;
              int rel = compare(entry.loc, qloc, delta);
              auto [it, inserted] =
                  best.try_emplace(entry.write, std::make_pair(rel, delta));
              if (inserted)
                order.push_back(entry.write);
              else if (rel < it->second.first)
                it->second = {rel, delta};
            }
          for (unsigned id : order) {
            auto [rel, delta] = best[id];
            if (rel == Disjoint)
              continue;
            applyWrite(index->getWrites()[id], rel, delta, size, type, path,
                       out);
          }
          if (path.size() >= kMaxChain) {
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
    if (rel == Equal)
      return resolveValue(write.value, size, type, chain, out);
    // The read is a part of a stored aggregate.
    Value value = write.value;
    if (auto load = value.getDefiningOp<cir::LoadOp>()) {
      Cell src = analysis.cellOf(load.getAddr());
      if (!src)
        return partial("a part of an aggregate loaded from nowhere");
      return readOut(Cell(src.getRawNode(), src.getRawOffset() + delta), size,
                     type, chain, out);
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
        size, type, chain, out);
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
                               StoredValues &out) {
  // SOUNDNESS: a cell with no faithful image in a caller's graph is read in
  // the graph the walk got to, which is sound for all of its contexts
  // (PointsToTopDown.cpp, "Images"); the copies above it stay on the chain,
  // so the formals its writes store still name this context's operands. A
  // node no copy includes is one the callers cannot reach (it is not
  // reachable from the callee's formals, return or globals when copied), so
  // only the activation that made it writes it.
  SmallVector<Cell, 2> cells;
  size_t left = mapOut(cell, chain, cells);
  for (Cell c : cells)
    out.merge(cellValues(c, size, type, chain.take_front(left)));
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
        readOut(cell, size, type ? type : value.getType(), chain, inner);
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
    storedValues = std::make_unique<StoredValueQuery>(*this);
  return *storedValues;
}

StoredValues PointsToAnalysis::valuesAtInContexts(Value addr,
                                                  std::optional<uint64_t> size,
                                                  Type type,
                                                  unsigned *numContexts) const {
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
  if (!cell || !size) {
    own.setIncomplete(
        notAnalysedWhy ? notAnalysedWhy
                       : graph->makeReason(Reason::Kind::UnhandledOp, nullptr,
                                           "no cell or no access size"));
    return own;
  }
  StoredValueQuery &query = getStoredValueQuery();
  own = query.valuesAt(cell, *size, nullptr, type);
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
  const Reason *defWhy = nullptr;
  unsigned count = 0;
  bool overBudget = false;
  SmallVector<const TUClone *, 8> path;
  std::function<void(const TUFunction &)> walk = [&](const TUFunction &at) {
    if (overBudget || !all.complete)
      return;
    auto copies = coveredClones.find(&at);
    if (copies == coveredClones.end() || path.size() >= 256) {
      if (++count > options.maxQueryContexts) {
        overBudget = true;
        return;
      }
      StoredValues here = query.valuesInContext(cell, *size, type, path);
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
                               Type type) const {
  StoredValues values = valuesAtInContexts(addr, size, type);
  if (!values.complete || !values.defs.empty() || values.consts.size() != 1)
    return std::nullopt;
  return values.consts.front();
}
