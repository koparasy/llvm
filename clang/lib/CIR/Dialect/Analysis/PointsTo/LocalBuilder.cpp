//===- LocalBuilder.cpp - Intraprocedural points-to graphs ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transfer rules of the local phase, per op:
//
//  - cir.alloca: a new Stack node, sized from the data layout; a dynamic
//    (VLA) alloca is an array of its element type.
//  - cir.load: Ref on the address node. A pointer result points to the
//    target of the slot loaded from. An aggregate result's slots are the
//    targets of the pointer slots inside the memory loaded from.
//  - cir.store: Mod on the address node. A pointer value is linked from the
//    slot; an aggregate value links each pointer slot it carries.
//  - cir.extract_member: the member's slot(s); cir.insert_member: the
//    record's slots with the member's replaced by the inserted value's.
//  - cir.freeze, cir.launder: the operand's cell or slots.
//  - Integer-only memory accesses (bitfields, atomic flags, memset,
//    clear_padding, masked vector loads, strlen): Mod and/or Ref only.
//  - cir.copy: a typed load of the pointee plus a store of it (the pointer
//    slots of the type share their targets on both sides), and a byte-range
//    copy record. cir.libc.memcpy/memmove of a constant length: a byte-range
//    copy record (an unknown length: a range without an end). After the
//    walk, closeMemTransfers unifies the targets at every offset either side
//    of a record has a link at. A cir.copy of a type of unknown size
//    collapses both nodes and unifies their single links.
//  - cir.libc.memchr: the source's node at an unknown offset (collapse).
//  - cir.vtable.get_vptr: the object's cell (the vptr is at offset 0).
//  - cir.vtable.address_point: the cell of the vtable global's node at the
//    address point's byte offset (GEP [0, index, offset] on the vtable's
//    record type); cir.vtt.address_point: entry `offset` of the VTT global
//    or of the sub-VTT operand. Loads of their slots read the links of the
//    #cir.vtable / VTT initializer (GlobalNodes).
//  - cir.vtable.get_virtual_fn_addr: the vptr's cell plus index pointers;
//    cir.vtable.get_type_info: the vptr's cell minus one pointer.
//  - cir.get_method of a constant #cir.method<@f>: the callee is @f's
//    Function node and the adjusted `this` the object's cell. Any other
//    method pointer is pessimistic.
//  - A load, store or copy through a type with arrays too long to scan for
//    pointer slots smashes the node (an array of the element size, or a
//    collapse) so that the scanned slots stand for the missing ones.
//  - Address arithmetic (get_member, get_element, ptr_stride, base/derived
//    class addresses, complex real/imag pointers, bitcast, address_space and
//    array_to_ptrdecay casts): the operand's cell plus the constant byte
//    offset decomposePointer computes for the one step. A dynamic
//    ptr_stride or get_element index, or a non-zero constant ptr_stride,
//    makes the node an array of the element size; any other unknown offset
//    collapses it.
//  - cir.ptr_mask, cir.get_runtime_member, cir.dyn_cast: the operand's node at
//    an unknown offset, so the node collapses.
//  - int_to_ptr casts give a fresh UnknownObj node, External and Incomplete;
//    ptr_to_int casts make the operand's node External.
//  - cir.select, region results (through RegionBranchTerminatorOpInterface),
//    region entry arguments, block arguments (through BranchOpInterface) and
//    cir.return (into the single return cell): unification, slot by slot
//    for aggregates.
//  - Atomics: a load of the old value plus a store of the new one.
//  - Calls: recorded, with a fresh node for a pointer result and fresh slot
//    nodes for an aggregate result. Then, for a direct call: a library model
//    (PointsToModels.h) if the registry has one; nothing more for a callee
//    with a body (the interprocedural phase connects it); the unknown-call
//    rule for any other declaration. Indirect calls are decided after the
//    walk: resolved (recorded only) if the callee pointer points only to
//    functions, the unknown-call rule otherwise; or, under
//    setDeferIndirectCalls, left to the interprocedural phase.
//  - The unknown-call rule: every node reachable from a pointer argument
//    (or a pointer slot of an aggregate argument) becomes External, with Mod
//    and Ref and Reason{EscapedTo, @callee}; the result and the result slots
//    point to UnknownObj nodes, External and Incomplete; the site is marked
//    clobbersGlobals. Effects on reachable nodes are applied after the walk,
//    so that links added after the call in walk order are covered too.
//  - cir.array.ctor / cir.array.dtor: the array's node becomes an array of
//    the element size and the element pointer of each region is its cell,
//    so the constructor or destructor call in the region is recorded with
//    `this` = the array. cir.delete_array: synthetic call sites for the
//    element destructor and for operator delete, with the array's cell.
//  - cir.std.find: the result points into the node of `first`, collapsed
//    (the same for each pointer slot of an iterator record).
//  - cir.get_global: one Global (or Function) node per symbol, its
//    initializer linked (GlobalNodes).
//  - cir.const: a null pointer has no cell; #cir.global_view and
//    #cir.global_offset point into the symbol's node at their byte offset; a
//    non-null #cir.ptr<N> points to an unknown object; an aggregate constant
//    gets one slot per pointer it holds (null for zero/undef/poison). A
//    symbol address held as an integer makes its node External.
//  - Entry block arguments: a fresh Incomplete node each (each slot, for an
//    aggregate).
//
// NoPointerSemantics ops are skipped. Pessimistic ops, unclassified ops and
// Handled ops whose rule does not cover them go to visitUnhandled. A pointer
// or aggregate result that is still undefined after its op was visited gets
// the pessimistic default too.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/PointsTo/LocalBuilder.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "clang/CIR/Dialect/Analysis/CIRPointerDecomposition.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/OpClassification.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToModels.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToVerifier.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

#define DEBUG_TYPE "cir-points-to"

using namespace mlir;
using namespace cir;
using namespace cir::pointsto;

/// Element counts beyond this are not scanned for pointer slots in an
/// aggregate SSA value.
static constexpr uint64_t kMaxScannedArrayElems = 256;

bool cir::pointsto::isPointerValueType(Type type) {
  return isa<cir::PointerType, cir::VPtrType>(type);
}

static bool isPointerType(Type type) { return isPointerValueType(type); }

bool cir::pointsto::isNullPointerConstant(Value value) {
  auto constOp = dyn_cast_if_present<cir::ConstantOp>(value.getDefiningOp());
  if (!constOp)
    return false;
  if (!isPointerType(value.getType()))
    return false;
  if (isa<cir::ZeroAttr>(constOp.getValue()))
    return true;
  auto ptrAttr = dyn_cast<cir::ConstPtrAttr>(constOp.getValue());
  return ptrAttr && ptrAttr.isNullValue();
}

static bool isNullPointer(Value value) { return isNullPointerConstant(value); }

/// The cell \p delta bytes after \p cell.
static Cell shift(Cell cell, int64_t delta) {
  return Cell(cell.getRawNode(), cell.getRawOffset() + delta);
}

//===----------------------------------------------------------------------===//
// FunctionGraph
//===----------------------------------------------------------------------===//

StringRef cir::pointsto::getCallKindName(CallKind kind) {
  switch (kind) {
  case CallKind::Defined:
    return "defined";
  case CallKind::Modelled:
    return "modelled";
  case CallKind::UnknownDecl:
    return "unknown-decl";
  case CallKind::IndirectResolved:
    return "indirect-resolved";
  case CallKind::IndirectUnknown:
    return "indirect-unknown";
  case CallKind::IndirectDeferred:
    return "indirect-deferred";
  }
  llvm_unreachable("unknown call kind");
}

static void printOptCell(raw_ostream &os, std::optional<Cell> cell) {
  Graph::printCell(os, cell ? *cell : Cell());
}

static void printSlots(raw_ostream &os, const AggregateSlots &slots) {
  os << "{";
  for (auto [i, slot] : llvm::enumerate(slots)) {
    os << (i ? ", " : "") << slot.first << ": ";
    Graph::printCell(os, slot.second);
  }
  os << "}";
}

void FunctionGraph::print(raw_ostream &os) const {
  graph.print(os);
  for (auto [i, formal] : llvm::enumerate(formals)) {
    if (formal) {
      os << "formal #" << i << " -> ";
      Graph::printCell(os, *formal);
      os << "\n";
      continue;
    }
    if (const AggregateSlots *slots =
            lookupSlots(const_cast<cir::FuncOp &>(func).getArgument(i))) {
      os << "formal #" << i << " -> ";
      printSlots(os, *slots);
      os << "\n";
    }
  }
  if (returnCell) {
    os << "return -> ";
    Graph::printCell(os, *returnCell);
    os << "\n";
  }
  if (returnSlots) {
    os << "return -> ";
    printSlots(os, *returnSlots);
    os << "\n";
  }
  for (const CallSiteRecord &site : callSites) {
    os << "call " << site.call->getName();
    if (site.callee)
      os << " " << site.callee;
    if (site.synthetic)
      os << " synthetic";
    if (site.calleePtr) {
      os << " callee=";
      Graph::printCell(os, *site.calleePtr);
    }
    os << " args={";
    bool first = true;
    auto callOp = dyn_cast<CIRCallOpInterface>(site.call);
    for (auto [i, arg] : llvm::enumerate(site.args)) {
      if (!site.synthetic && callOp &&
          !isPointerType(callOp.getArgOperand(i).getType()))
        continue;
      os << (first ? "" : ", ") << i << ": ";
      printOptCell(os, arg);
      first = false;
    }
    os << "} result=";
    printOptCell(os, site.result);
    if (!site.aggregateArgs.empty()) {
      os << " slots={";
      for (auto [i, entry] : llvm::enumerate(site.aggregateArgs)) {
        os << (i ? ", " : "") << entry.first << ": ";
        printSlots(os, entry.second);
      }
      os << "}";
    }
    if (site.resultSlots) {
      os << " result.slots=";
      printSlots(os, *site.resultSlots);
    }
    os << " kind=" << getCallKindName(site.kind);
    if (!site.modelled.empty())
      os << " model=" << site.modelled;
    if (site.clobbersGlobals)
      os << " clobbers-globals";
    os << "\n";
  }
}

//===----------------------------------------------------------------------===//
// LocalBuilder
//===----------------------------------------------------------------------===//

mlir::DataLayout LocalBuilder::getDataLayoutFor(cir::FuncOp func) {
  return mlir::DataLayout(func->getParentOfType<mlir::ModuleOp>());
}

LocalBuilder::LocalBuilder(cir::FuncOp func, SymbolTableCollection *symbols,
                           ReasonTable *reasons,
                           const PointsToModelRegistry *models)
    : func(func), dataLayout(getDataLayoutFor(func)), symbols(symbols),
      models(models) {
  if (!symbols) {
    ownedSymbols = std::make_unique<SymbolTableCollection>();
    this->symbols = ownedSymbols.get();
  }
  auto voidPtrTy = cir::PointerType::get(cir::VoidType::get(func.getContext()));
  unsigned pointerSize = dataLayout.getTypeSize(voidPtrTy).getFixedValue();
  fg = std::make_unique<FunctionGraph>(func, pointerSize, reasons);
  globals = std::make_unique<GlobalNodes>(fg->getGraph(), dataLayout,
                                          *this->symbols, func);
}

LocalBuilder::~LocalBuilder() = default;

std::optional<Cell> LocalBuilder::cellFor(Value value) {
  if (!value || !isPointerType(value.getType()) || isNullPointer(value))
    return std::nullopt;
  auto [it, inserted] = fg->scalarMap.try_emplace(value, Cell());
  if (inserted)
    it->second = Cell(fg->getGraph().makeNode());
  return it->second;
}

void LocalBuilder::define(Value value, Cell cell) {
  defined.insert(value);
  auto [it, inserted] = fg->scalarMap.try_emplace(value, cell);
  if (!inserted)
    fg->getGraph().unify(it->second, cell);
}

std::optional<AggregateSlots> LocalBuilder::slotsFor(Value value) {
  if (!value || isPointerType(value.getType()))
    return std::nullopt;
  auto it = fg->aggregateMap.find(value);
  if (it != fg->aggregateMap.end())
    return it->second;
  SmallVector<int64_t, 2> offsets = getPointerOffsets(value.getType());
  if (offsets.empty())
    return std::nullopt;
  AggregateSlots slots;
  for (int64_t off : offsets)
    slots.emplace_back(off, Cell(fg->getGraph().makeNode()));
  fg->aggregateMap.try_emplace(value, slots);
  return slots;
}

void LocalBuilder::defineSlots(Value value, const AggregateSlots &slots) {
  defined.insert(value);
  auto [it, inserted] = fg->aggregateMap.try_emplace(value, slots);
  if (inserted)
    return;
  // A placeholder made by a use seen first: merge slot by slot.
  Graph &graph = fg->getGraph();
  for (auto [have, give] : llvm::zip(it->second, slots)) {
    if (!give.second)
      continue;
    if (!have.second)
      have.second = give.second;
    else
      graph.unify(have.second, give.second);
  }
}

void LocalBuilder::defineUnknown(Value value, const Reason *why) {
  Graph &graph = fg->getGraph();
  auto fresh = [&] {
    Cell cell(graph.makeNode(NodeFlags::UnknownObj));
    graph.addFlags(cell, NodeFlags::External | NodeFlags::Incomplete, why);
    return cell;
  };
  if (isPointerType(value.getType())) {
    define(value, fresh());
    return;
  }
  SmallVector<int64_t, 2> offsets = getPointerOffsets(value.getType());
  if (offsets.empty())
    return;
  AggregateSlots slots;
  for (int64_t off : offsets)
    slots.emplace_back(off, fresh());
  defineSlots(value, slots);
}

void LocalBuilder::defineCopy(Value result, Value src) {
  if (isPointerType(result.getType())) {
    // A copy of null points to nothing, like a merge of nulls.
    std::optional<Cell> cell = cellFor(src);
    define(result, cell ? *cell : Cell(fg->getGraph().makeNode()));
    return;
  }
  if (std::optional<AggregateSlots> slots = slotsFor(src))
    defineSlots(result, *slots);
}

void LocalBuilder::definePlaceholder(Value value) {
  if (isPointerType(value.getType())) {
    if (isNullPointer(value))
      return;
    if (!cellFor(value))
      return;
  } else if (!slotsFor(value)) {
    return;
  }
  defined.insert(value);
}

void LocalBuilder::unifyValues(Value a, Value b, Operation *at) {
  Graph &graph = fg->getGraph();
  if (isPointerType(a.getType())) {
    std::optional<Cell> ca = cellFor(a);
    std::optional<Cell> cb = cellFor(b);
    if (!ca || !cb)
      return;
    graph.unify(*ca, *cb, graph.makeReason(Reason::Kind::MergedAt, at));
    return;
  }
  std::optional<AggregateSlots> sa = slotsFor(a);
  std::optional<AggregateSlots> sb = slotsFor(b);
  if (!sa || !sb)
    return;
  const Reason *why = nullptr;
  for (auto [x, y] : llvm::zip(*sa, *sb)) {
    if (!x.second || !y.second)
      continue;
    if (!why)
      why = graph.makeReason(Reason::Kind::MergedAt, at);
    graph.unify(x.second, y.second, why);
  }
}

static void collectPointerOffsets(Type type, int64_t base,
                                  const mlir::DataLayout &dataLayout,
                                  SmallVectorImpl<int64_t> &out,
                                  SlotTruncation &truncation) {
  if (isPointerType(type)) {
    out.push_back(base);
    return;
  }
  if (auto recordTy = dyn_cast<cir::RecordType>(type)) {
    if (recordTy.isIncomplete())
      return;
    for (unsigned i = 0, e = recordTy.getNumElements(); i != e; ++i)
      collectPointerOffsets(recordTy.getElementType(i),
                            base + recordTy.getElementOffset(dataLayout, i),
                            dataLayout, out, truncation);
    return;
  }
  if (auto arrayTy = dyn_cast<cir::ArrayType>(type)) {
    SmallVector<int64_t> elemOffsets;
    SlotTruncation elemTruncation;
    collectPointerOffsets(arrayTy.getElementType(), 0, dataLayout, elemOffsets,
                          elemTruncation);
    std::optional<int64_t> elemSize =
        getTypeSizeInBytes(arrayTy.getElementType(), dataLayout);
    if (elemOffsets.empty() || !elemSize)
      return;
    uint64_t count =
        std::min<uint64_t>(arrayTy.getSize(), kMaxScannedArrayElems);
    for (uint64_t k = 0; k != count; ++k)
      for (int64_t off : elemOffsets)
        out.push_back(base + k * *elemSize + off);

    // Record what was left out: this array if it is too long, and whatever
    // the element type left out, once per scanned element.
    auto record = [&](int64_t at, std::optional<uint64_t> elem) {
      std::optional<uint64_t> smash = at == 0 ? elem : std::nullopt;
      if (!truncation.truncated) {
        truncation.truncated = true;
        truncation.arrayElemSize = smash;
      } else if (truncation.arrayElemSize != smash) {
        truncation.arrayElemSize = std::nullopt;
      }
    };
    if (arrayTy.getSize() > kMaxScannedArrayElems)
      record(base, *elemSize);
    if (elemTruncation.truncated)
      for (uint64_t k = 0; k != count; ++k)
        record(base + k * *elemSize, elemTruncation.arrayElemSize);
  }
}

SmallVector<int64_t, 2> cir::pointsto::getPointerSlotOffsets(
    Type type, const mlir::DataLayout &dataLayout, SlotTruncation *truncation) {
  SmallVector<int64_t, 2> offsets;
  SlotTruncation local;
  SlotTruncation &out = truncation ? *truncation : local;
  out = SlotTruncation();
  if (isPointerType(type))
    return offsets;
  collectPointerOffsets(type, 0, dataLayout, offsets, out);
  llvm::sort(offsets);
  offsets.erase(llvm::unique(offsets), offsets.end());
  return offsets;
}

LocalBuilder::PointerLayout LocalBuilder::getPointerLayout(Type type) {
  auto [it, inserted] = pointerLayouts.try_emplace(type);
  if (inserted)
    it->second.offsets =
        getPointerSlotOffsets(type, dataLayout, &it->second.truncation);
  return it->second;
}

SmallVector<int64_t, 2> LocalBuilder::getPointerOffsets(Type type) {
  return getPointerLayout(type).offsets;
}

bool LocalBuilder::hasPointers(Type type) {
  return isPointerType(type) || !getPointerOffsets(type).empty();
}

std::unique_ptr<FunctionGraph> LocalBuilder::build() {
  assert(fg && "build() called twice");
  Graph &graph = fg->getGraph();

  // Formals: what a caller passes in is not known yet, unless every caller
  // is bound to the function later (setFormalsComplete).
  Block &entry = func.getBody().front();
  for (BlockArgument arg : entry.getArguments()) {
    if (!hasPointers(arg.getType())) {
      fg->formals.push_back(std::nullopt);
      continue;
    }
    const Reason *why =
        formalsComplete
            ? nullptr
            : graph.makeReason(Reason::Kind::UnknownCaller, func,
                               "formal #" + std::to_string(arg.getArgNumber()));
    auto fresh = [&] {
      Cell cell(graph.makeNode());
      if (why)
        graph.addFlags(cell, NodeFlags::Incomplete, why);
      return cell;
    };
    if (!isPointerType(arg.getType())) {
      AggregateSlots slots;
      for (int64_t off : getPointerOffsets(arg.getType()))
        slots.emplace_back(off, fresh());
      defineSlots(arg, slots);
      fg->formals.push_back(std::nullopt);
      continue;
    }
    Cell cell = fresh();
    define(arg, cell);
    fg->formals.push_back(cell);
  }

  func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op != func.getOperation())
      visit(op);
  });
  defineBlockArguments();
  closeMemTransfers(graph, fg->memTransfers);
  // A child at offset 0 used past its extent was a pointer to its parent
  // (pointer-interconvertibility): fold it back, then relate the copies
  // again, as the merge may have added links in their ranges.
  if (subobjects)
    while (graph.absorbOverflowingSubobjects(graph.makeReason(
        Reason::Kind::MergedAt, func, "member used as its enclosing object")))
      closeMemTransfers(graph, fg->memTransfers);
  applyDeferredEffects();
  if (deferIndirectCalls) {
    for (size_t index : indirectCalls)
      fg->callSites[index].kind = CallKind::IndirectDeferred;
    indirectCalls.clear();
  } else {
    resolveIndirectCalls();
  }
  for (auto [sym, node] : globals->getSymbolNodes())
    fg->symbolCells.emplace_back(sym, Cell(node));

  if (isPointsToVerifyEnabled())
    verifyErrors = failed(verifyFunctionGraph(*fg));
  return std::move(fg);
}

void LocalBuilder::defineBlockArguments() {
  Graph &graph = fg->getGraph();
  Block *funcEntry = &func.getBody().front();
  func.walk([&](Block *block) {
    if (block == funcEntry)
      return;
    // A block reached through a terminator that is not a BranchOpInterface
    // gets its arguments from an edge nobody modelled.
    Operation *unknownPred = nullptr;
    if (!block->isEntryBlock())
      for (Block *pred : block->getPredecessors())
        if (!isa<BranchOpInterface>(pred->getTerminator()))
          unknownPred = pred->getTerminator();
    const Reason *why = nullptr;
    for (BlockArgument arg : block->getArguments()) {
      if (!hasPointers(arg.getType()) || defined.contains(arg))
        continue;
      if (!unknownPred) {
        // Defined by the edges into the block, or dead.
        definePlaceholder(arg);
        continue;
      }
      if (!why)
        why = graph.makeReason(Reason::Kind::UnhandledOp, unknownPred,
                               "block argument from an unmodelled edge");
      defineUnknown(arg, why);
    }
  });
}

void LocalBuilder::visit(Operation *op) {
  StringRef name = op->getName().getStringRef();
  std::optional<OpClass> cls = lookupOpClass(name);
  if (!cls) {
    ++unclassifiedOps[name];
    LDBG() << "op not in the points-to classification table, treated as "
              "Pessimistic: "
           << name;
  }

  // Control flow is handled through the interfaces, before and independently
  // of the op-specific rules (a cir.try_call is both a branch and a call).
  bool isControlFlow = false;
  if (isa<RegionBranchTerminatorOpInterface>(op)) {
    visitRegionTerminator(op);
    isControlFlow = true;
  }
  if (isa<BranchOpInterface>(op)) {
    visitBranch(op);
    isControlFlow = true;
  }
  // Region results are defined by the terminators that yield them.
  bool isRegionBranch = isa<RegionBranchOpInterface>(op);
  if (isRegionBranch) {
    visitRegionEntry(op);
    isControlFlow = true;
  } else if (op->getNumRegions() != 0) {
    // Nobody models what such an op passes into its regions.
    const Reason *why = nullptr;
    for (Region &region : op->getRegions()) {
      if (region.empty())
        continue;
      for (BlockArgument arg : region.front().getArguments()) {
        if (!hasPointers(arg.getType()))
          continue;
        if (isa<cir::ArrayCtor, cir::ArrayDtor>(op))
          continue; // The element pointer: visitArrayInit.
        if (!why)
          why = fg->getGraph().makeReason(Reason::Kind::UnhandledOp, op,
                                          "region argument");
        defineUnknown(arg, why);
      }
    }
  }

  switch (cls.value_or(OpClass::Pessimistic)) {
  case OpClass::NoPointerSemantics:
    break;
  case OpClass::Pessimistic:
    visitUnhandled(op, cls ? StringRef() : StringRef("unclassified op"));
    break;
  case OpClass::Handled:
    llvm::TypeSwitch<Operation *>(op)
        .Case<cir::AllocaOp>([&](auto op) { visitAlloca(op); })
        .Case<cir::LoadOp>([&](auto op) { visitLoad(op); })
        .Case<cir::StoreOp>([&](auto op) { visitStore(op); })
        .Case<cir::GetMemberOp, cir::GetElementOp, cir::PtrStrideOp,
              cir::BaseClassAddrOp, cir::DerivedClassAddrOp,
              cir::ComplexRealPtrOp, cir::ComplexImagPtrOp>(
            [&](Operation *op) { visitAddressStep(op); })
        .Case<cir::CastOp>([&](auto op) { visitCast(op); })
        .Case<cir::PtrMaskOp>(
            [&](cir::PtrMaskOp op) { visitCollapse(op, op.getPtr()); })
        .Case<cir::GetRuntimeMemberOp>([&](cir::GetRuntimeMemberOp op) {
          visitCollapse(op, op.getAddr());
        })
        .Case<cir::DynamicCastOp>(
            [&](cir::DynamicCastOp op) { visitCollapse(op, op.getSrc()); })
        .Case<cir::SelectOp>([&](cir::SelectOp op) {
          definePlaceholder(op.getResult());
          unifyValues(op.getResult(), op.getTrueValue(), op);
          unifyValues(op.getResult(), op.getFalseValue(), op);
        })
        .Case<cir::FreezeOp>([&](cir::FreezeOp op) {
          defineCopy(op.getResult(), op.getInput());
        })
        .Case<cir::LaunderOp>(
            [&](cir::LaunderOp op) { defineCopy(op.getResult(), op.getArg()); })
        .Case<cir::ExtractMemberOp>([&](auto op) { visitExtractMember(op); })
        .Case<cir::InsertMemberOp>([&](auto op) { visitInsertMember(op); })
        .Case<cir::AtomicFetchOp>([&](cir::AtomicFetchOp op) {
          visitAtomic(op, op.getPtr(), op.getVal(), op.getResult());
        })
        .Case<cir::AtomicXchgOp>([&](cir::AtomicXchgOp op) {
          visitAtomic(op, op.getPtr(), op.getVal(), op.getResult());
        })
        .Case<cir::AtomicCmpXchgOp>([&](cir::AtomicCmpXchgOp op) {
          visitAtomic(op, op.getPtr(), op.getDesired(), op.getOld());
        })
        .Case<cir::AtomicTestAndSetOp>([&](cir::AtomicTestAndSetOp op) {
          visitModRef(op.getPtr(), op.getPtr().getType().getPointee(),
                      /*mod=*/true, /*ref=*/true);
        })
        .Case<cir::AtomicClearOp>([&](cir::AtomicClearOp op) {
          visitModRef(op.getPtr(), op.getPtr().getType().getPointee(),
                      /*mod=*/true, /*ref=*/true);
        })
        .Case<cir::GetBitfieldOp>([&](cir::GetBitfieldOp op) {
          visitModRef(op.getAddr(), {}, /*mod=*/false, /*ref=*/true);
        })
        .Case<cir::SetBitfieldOp>([&](cir::SetBitfieldOp op) {
          visitModRef(op.getAddr(), {}, /*mod=*/true, /*ref=*/false);
        })
        .Case<cir::VecMaskedLoadOp>([&](cir::VecMaskedLoadOp op) {
          visitModRef(op.getAddr(), op.getType(), /*mod=*/false,
                      /*ref=*/true);
        })
        .Case<cir::MemSetOp>([&](cir::MemSetOp op) {
          // A byte pattern makes no pointer (zero writes null), and a
          // flow-insensitive graph never removes a link.
          visitModRef(op.getDst(), {}, /*mod=*/true, /*ref=*/false);
          std::optional<Cell> dst = cellFor(op.getDst());
          std::optional<int64_t> length = getConstantIndex(op.getLen());
          if (dst && length && *length > 0)
            fg->getGraph().growSize(dst->getNode(), dst->getOffset() + *length);
        })
        .Case<cir::CopyOp>([&](auto op) { visitCopy(op); })
        .Case<cir::MemCpyOp>([&](cir::MemCpyOp op) {
          visitMemTransfer(op, op.getDst(), op.getSrc(), op.getLen());
        })
        .Case<cir::MemMoveOp>([&](cir::MemMoveOp op) {
          visitMemTransfer(op, op.getDst(), op.getSrc(), op.getLen());
        })
        .Case<cir::MemChrOp>([&](auto op) { visitMemChr(op); })
        .Case<cir::StdFindOp>([&](auto op) { visitStdFind(op); })
        .Case<cir::ArrayCtor>([&](cir::ArrayCtor op) {
          visitArrayInit(op, op.getAddr(), op.getNumElements());
        })
        .Case<cir::ArrayDtor>([&](cir::ArrayDtor op) {
          visitArrayInit(op, op.getAddr(), op.getNumElements());
        })
        .Case<cir::DeleteArrayOp>([&](auto op) { visitDeleteArray(op); })
        .Case<cir::StrLenOp>([&](cir::StrLenOp op) {
          visitModRef(op.getString(), {}, /*mod=*/false, /*ref=*/true);
        })
        .Case<cir::VTableGetVPtrOp>([&](cir::VTableGetVPtrOp op) {
          // The vptr is at offset 0 of the object: like a bitcast.
          defineCopy(op.getResult(), op.getSrc());
        })
        .Case<cir::VTableAddrPointOp>(
            [&](cir::VTableAddrPointOp op) { visitVTableAddrPoint(op); })
        .Case<cir::VTTAddrPointOp>(
            [&](cir::VTTAddrPointOp op) { visitVTTAddrPoint(op); })
        .Case<cir::VTableGetVirtualFnAddrOp>(
            [&](cir::VTableGetVirtualFnAddrOp op) {
              visitVPtrStep(op, op.getVptr(),
                            static_cast<int64_t>(op.getIndex()) *
                                fg->getGraph().getPointerSize());
            })
        .Case<cir::VTableGetTypeInfoOp>([&](cir::VTableGetTypeInfoOp op) {
          // Itanium C++ ABI 2.5.2: the RTTI pointer is the slot just before
          // the address point (lowered as a ptr_stride of -1 pointer).
          visitVPtrStep(op, op.getVptr(),
                        -static_cast<int64_t>(fg->getGraph().getPointerSize()));
        })
        .Case<cir::GetMethodOp>([&](auto op) { visitGetMethod(op); })
        .Case<cir::LocalInitOp>([](cir::LocalInitOp) {
          // The walk visits the ctor and dtor regions as part of the
          // function; the op itself moves no pointer.
        })
        .Case<cir::ClearPaddingOp>([&](cir::ClearPaddingOp op) {
          visitModRef(op.getArg(), {}, /*mod=*/true, /*ref=*/false);
        })
        .Case<cir::LLVMIntrinsicCallOp>([&](cir::LLVMIntrinsicCallOp op) {
          if (!isPointerFreeLLVMIntrinsic(op.getIntrinsicName()))
            visitUnhandled(op, "intrinsic not on the allow-list");
        })
        .Case<CIRCallOpInterface>([&](auto op) { visitCall(op); })
        .Case<cir::GetGlobalOp>([&](auto op) { visitGetGlobal(op); })
        .Case<cir::ReturnOp>([&](auto op) { visitReturn(op); })
        .Case<cir::ConstantOp>([&](auto op) { visitConstant(op); })
        .Default([&](Operation *op) {
          if (isControlFlow)
            return;
          LDBG() << "op classified Handled has no transfer rule: "
                 << op->getName();
          visitUnhandled(op, "no transfer rule");
        });
    break;
  }

  if (!isRegionBranch)
    defineMissingResults(op);
}

void LocalBuilder::defineMissingResults(Operation *op) {
  const Reason *why = nullptr;
  for (OpResult result : op->getResults()) {
    if (defined.contains(result) || isNullPointer(result) ||
        !hasPointers(result.getType()))
      continue;
    LDBG() << "result #" << result.getResultNumber() << " of " << op->getName()
           << " has no rule, treated pessimistically";
    if (!why)
      why = fg->getGraph().makeReason(Reason::Kind::UnhandledOp, op,
                                      "result without a rule");
    defineUnknown(result, why);
  }
}

void LocalBuilder::visitUnhandled(Operation *op, StringRef detail) {
  StringRef name = op->getName().getStringRef();
  ++unhandledOps[name];
  Graph &graph = fg->getGraph();
  const Reason *why = nullptr;
  auto reason = [&] {
    if (!why)
      why = graph.makeReason(Reason::Kind::UnhandledOp, op, detail);
    return why;
  };
  constexpr NodeFlags kEscaped = NodeFlags::Collapsed | NodeFlags::External;
  bool touched = false;
  // The op may read, write, publish or offset anything its pointers reach.
  for (Value operand : op->getOperands()) {
    if (isPointerType(operand.getType())) {
      if (std::optional<Cell> cell = cellFor(operand)) {
        graph.addFlags(*cell, kEscaped, reason());
        touched = true;
      }
      continue;
    }
    if (std::optional<AggregateSlots> slots = slotsFor(operand))
      for (auto &[off, cell] : *slots) {
        if (!cell)
          continue;
        graph.addFlags(cell, kEscaped, reason());
        touched = true;
      }
  }
  // And its pointer results may point anywhere.
  for (Value result : op->getResults()) {
    if (!hasPointers(result.getType()) || isNullPointer(result))
      continue;
    defineUnknown(result, reason());
    touched = true;
  }
  if (touched || !isMemoryEffectFree(op))
    ++pessimisticOps[name];
}

void LocalBuilder::visitModRef(Value addrValue, Type accessType, bool mod,
                               bool ref) {
  std::optional<Cell> addr = cellFor(addrValue);
  if (!addr)
    return;
  Graph &graph = fg->getGraph();
  NodeFlags flags = NodeFlags::None;
  if (mod)
    flags |= NodeFlags::Mod;
  if (ref)
    flags |= NodeFlags::Ref;
  graph.addFlags(*addr, flags);
  if (accessType)
    if (std::optional<int64_t> size =
            getTypeSizeInBytes(accessType, dataLayout))
      graph.growSize(addr->getNode(), addr->getOffset() + *size);
}

void LocalBuilder::visitAlloca(cir::AllocaOp op) {
  Graph &graph = fg->getGraph();
  Node *node =
      graph.makeNode(Origin{Origin::Kind::Alloca, op}, NodeFlags::Stack);
  std::optional<int64_t> size =
      getTypeSizeInBytes(op.getAllocaType(), dataLayout);
  if (size)
    graph.growSize(node, *size);
  if (op.getDynAllocSize())
    graph.markArray(node, size.value_or(0),
                    graph.makeReason(Reason::Kind::CollapsedAt, op,
                                     "variable-length alloca"));
  define(op.getResult(), Cell(node));
}

void LocalBuilder::visitLoad(cir::LoadOp op) {
  std::optional<Cell> addr = cellFor(op.getAddr());
  if (!addr)
    return; // Loading through null is UB.
  Graph &graph = fg->getGraph();
  graph.addFlags(*addr, NodeFlags::Ref);
  Type type = op.getType();
  if (std::optional<int64_t> size = getTypeSizeInBytes(type, dataLayout))
    graph.growSize(addr->getNode(), addr->getOffset() + *size);

  // A new link can collapse the node (a misaligned slot, a byte array).
  const Reason *why = nullptr;
  auto reason = [&] {
    if (!why)
      why = graph.makeReason(Reason::Kind::CollapsedAt, op);
    return why;
  };
  if (isPointerType(type)) {
    define(op.getResult(), graph.getLink(*addr, reason()));
    return;
  }
  SmallVector<int64_t, 2> offsets = getPointerOffsets(type);
  if (offsets.empty())
    return;
  smashIfTruncated(*addr, type, op);
  AggregateSlots slots;
  for (int64_t off : offsets)
    slots.emplace_back(off, graph.getLink(shift(*addr, off), reason()));
  defineSlots(op.getResult(), slots);
}

void LocalBuilder::visitStore(cir::StoreOp op) {
  std::optional<Cell> addr = cellFor(op.getAddr());
  if (!addr)
    return; // Storing through null is UB.
  Graph &graph = fg->getGraph();
  graph.addFlags(*addr, NodeFlags::Mod);
  Value value = op.getValue();
  if (std::optional<int64_t> size =
          getTypeSizeInBytes(value.getType(), dataLayout))
    graph.growSize(addr->getNode(), addr->getOffset() + *size);

  const Reason *why = nullptr;
  auto reason = [&] {
    if (!why)
      why = graph.makeReason(Reason::Kind::CollapsedAt, op);
    return why;
  };
  if (isPointerType(value.getType())) {
    if (std::optional<Cell> target = cellFor(value))
      graph.setLink(*addr, *target, reason());
    return;
  }
  std::optional<AggregateSlots> slots = slotsFor(value);
  if (!slots)
    return;
  smashIfTruncated(*addr, value.getType(), op);
  for (auto [off, target] : *slots)
    if (target)
      graph.setLink(shift(*addr, off), target, reason());
}

/// The slot at \p offset in \p slots: std::nullopt if there is none, an
/// empty Cell if it holds null.
static std::optional<Cell> findSlot(const AggregateSlots &slots,
                                    int64_t offset) {
  auto it = llvm::lower_bound(slots, offset, [](const auto &slot, int64_t o) {
    return slot.first < o;
  });
  if (it == slots.end() || it->first != offset)
    return std::nullopt;
  return it->second;
}

void LocalBuilder::visitExtractMember(cir::ExtractMemberOp op) {
  Value result = op.getResult();
  Type resultType = result.getType();
  if (!hasPointers(resultType))
    return;
  auto recordTy = cast<cir::RecordType>(op.getRecord().getType());
  int64_t base = recordTy.getElementOffset(dataLayout, op.getIndex());
  std::optional<AggregateSlots> src = slotsFor(op.getRecord());
  if (!src)
    return; // Left to defineMissingResults.
  if (isPointerType(resultType)) {
    std::optional<Cell> slot = findSlot(*src, base);
    if (!slot)
      return; // Not scanned: left to defineMissingResults.
    define(result, *slot ? *slot : Cell(fg->getGraph().makeNode()));
    return;
  }
  AggregateSlots slots;
  for (int64_t off : getPointerOffsets(resultType)) {
    std::optional<Cell> slot = findSlot(*src, base + off);
    if (!slot)
      return;
    slots.emplace_back(off, *slot);
  }
  defineSlots(result, slots);
}

void LocalBuilder::visitInsertMember(cir::InsertMemberOp op) {
  Value record = op.getRecord();
  if (!hasPointers(record.getType()))
    return;
  auto recordTy = cast<cir::RecordType>(record.getType());
  unsigned index = op.getIndex();
  int64_t base = recordTy.getElementOffset(dataLayout, index);
  std::optional<int64_t> memberSize =
      getTypeSizeInBytes(recordTy.getElementType(index), dataLayout);
  std::optional<AggregateSlots> src = slotsFor(record);
  if (!memberSize || !src)
    return; // Left to defineMissingResults.

  Value value = op.getValue();
  bool isPointerValue = isPointerType(value.getType());
  std::optional<Cell> valueCell =
      isPointerValue ? cellFor(value) : std::nullopt;
  std::optional<AggregateSlots> valueSlots =
      isPointerValue ? std::nullopt : slotsFor(value);
  AggregateSlots slots;
  for (auto [off, cell] : *src) {
    if (off < base || off >= base + *memberSize) {
      slots.emplace_back(off, cell);
      continue;
    }
    // Inside the member: what the inserted value holds there; a slot it has
    // no pointer at holds no valid pointer any more.
    Cell from;
    if (isPointerValue) {
      if (off == base && valueCell)
        from = *valueCell;
    } else if (valueSlots) {
      if (std::optional<Cell> slot = findSlot(*valueSlots, off - base))
        from = *slot;
    }
    slots.emplace_back(off, from);
  }
  defineSlots(op.getResult(), slots);
}

Cell LocalBuilder::constantPointerCell(const ConstantPointer &pointer,
                                       Operation *op) {
  Graph &graph = fg->getGraph();
  switch (pointer.kind) {
  case ConstantPointer::Kind::Symbol:
    return globals->getCell(pointer, op);
  case ConstantPointer::Kind::Absolute:
    return globals->makeUnknown(
        graph.makeReason(Reason::Kind::EscapedTo, op, "absolute address"));
  case ConstantPointer::Kind::Unknown:
    break;
  }
  return globals->makeUnknown(graph.makeReason(Reason::Kind::UnhandledOp, op,
                                               "constant not understood"));
}

std::optional<AggregateSlots>
LocalBuilder::constantSlots(Attribute value, Type type, Operation *op) {
  Graph &graph = fg->getGraph();
  PointerLayout layout = getPointerLayout(type);
  AggregateSlots slots;
  for (int64_t off : layout.offsets)
    slots.emplace_back(off, Cell());
  bool understood = true;
  // With a truncated layout some pointers have no slot: every slot then
  // stands for all of them.
  bool truncated = layout.truncation.truncated;
  Cell merged;
  walkConstantPointers(
      value, 0, dataLayout,
      [&](FlatSymbolRefAttr sym) { return globals->getSymbolType(sym); },
      [&](int64_t off, const ConstantPointer &pointer) {
        if (pointer.asInteger) {
          graph.addFlags(globals->getCell(pointer, op), NodeFlags::External,
                         graph.makeReason(Reason::Kind::EscapedTo, op,
                                          "address escaped as an integer"));
          return;
        }
        if (pointer.kind == ConstantPointer::Kind::Unknown) {
          understood = false;
          return;
        }
        Cell target = constantPointerCell(pointer, op);
        if (truncated) {
          if (!merged)
            merged = target;
          else
            graph.unify(merged, target,
                        graph.makeReason(Reason::Kind::MergedAt, op));
          return;
        }
        auto it =
            llvm::lower_bound(slots, off, [](const auto &slot, int64_t o) {
              return slot.first < o;
            });
        if (it == slots.end() || it->first != off) {
          // A pointer where the type has none: the attribute and the type
          // disagree.
          understood = false;
          return;
        }
        if (!it->second)
          it->second = target;
        else
          graph.unify(it->second, target,
                      graph.makeReason(Reason::Kind::MergedAt, op));
      });
  if (!understood)
    return std::nullopt;
  if (truncated && merged)
    for (auto &slot : slots)
      slot.second = merged;
  return slots;
}

void LocalBuilder::visitConstant(cir::ConstantOp op) {
  Type type = op.getType();
  Attribute value = op.getValue();
  Graph &graph = fg->getGraph();
  auto symbolType = [&](FlatSymbolRefAttr sym) {
    return globals->getSymbolType(sym);
  };
  if (!hasPointers(type)) {
    // A symbol address held as an integer escapes, like a ptr_to_int cast.
    walkConstantPointers(
        value, 0, dataLayout, symbolType,
        [&](int64_t, const ConstantPointer &pointer) {
          if (pointer.kind == ConstantPointer::Kind::Symbol)
            graph.addFlags(globals->getCell(pointer, op), NodeFlags::External,
                           graph.makeReason(Reason::Kind::EscapedTo, op,
                                            "address escaped as an integer"));
        });
    return;
  }
  if (isNullPointer(op.getResult()))
    return; // A null pointer constant has no cell (see cellFor).

  if (isPointerType(type)) {
    // &g + offset, an absolute address, or (undef, poison) nothing.
    std::optional<Cell> cell;
    bool understood = true;
    walkConstantPointers(value, 0, dataLayout, symbolType,
                         [&](int64_t off, const ConstantPointer &pointer) {
                           if (off != 0 || pointer.asInteger || cell) {
                             understood = false;
                             return;
                           }
                           cell = constantPointerCell(pointer, op);
                         });
    if (!understood) {
      visitUnhandled(op, "pointer constant not understood");
      return;
    }
    define(op.getResult(), cell ? *cell : Cell(graph.makeNode()));
    return;
  }

  std::optional<AggregateSlots> slots = constantSlots(value, type, op);
  if (!slots) {
    visitUnhandled(op, "aggregate constant not understood");
    return;
  }
  defineSlots(op.getResult(), *slots);
}

void LocalBuilder::visitAddressStep(Operation *op) {
  Value result = op->getResult(0);
  PointerDecomposition step = decomposePointer(result, dataLayout,
                                               /*maxDepth=*/1);
  assert(step.base != result && "expected a one-step address computation");
  std::optional<Cell> base = cellFor(step.base);
  Graph &graph = fg->getGraph();
  if (!base) {
    // An address computed from null points to nothing.
    define(result, Cell(graph.makeNode()));
    return;
  }
  std::optional<int64_t> elemSize;
  auto strideOp = dyn_cast<cir::PtrStrideOp>(op);
  if (strideOp)
    elemSize = getTypeSizeInBytes(strideOp.getElementType(), dataLayout);
  else if (auto elementOp = dyn_cast<cir::GetElementOp>(op))
    elemSize = getTypeSizeInBytes(elementOp.getElementType(), dataLayout);

  if (subobjects && (strideOp || isa<cir::DerivedClassAddrOp>(op)))
    collapseIfLeavingSubobject(op, *base, step.offset, elemSize);

  if (step.offset) {
    // A member of class, union or array type is its own child node.
    if (subobjects)
      if (auto member = dyn_cast<cir::GetMemberOp>(op))
        if (std::optional<uint64_t> size = getSubobjectSize(member)) {
          define(result,
                 graph.getSubobject(shift(*base, *step.offset), *size,
                                    graph.makeReason(Reason::Kind::MergedAt, op,
                                                     "subobject")));
          return;
        }
    // Pointer arithmetic is only defined within an array, so a non-zero
    // constant stride is evidence that the node is an array of the element
    // type (the classic DSA rule). Without it, `p = p + 1` in a loop unifies
    // the slot with itself at a shift and collapses the node.
    if (strideOp && *step.offset != 0 && elemSize && *elemSize > 0)
      graph.markArray(base->getNode(), *elemSize,
                      graph.makeReason(Reason::Kind::CollapsedAt, op,
                                       "pointer arithmetic"));
    define(result, shift(*base, *step.offset));
    return;
  }

  // The offset is not a compile-time constant. A dynamic index steps over
  // elements of a known size: smash the node into an array of them. Any
  // other unknown offset leaves no field to name.
  if (elemSize && *elemSize > 0)
    graph.markArray(
        base->getNode(), *elemSize,
        graph.makeReason(Reason::Kind::CollapsedAt, op, "dynamic index"));
  else
    graph.collapse(base->getNode(), graph.makeReason(Reason::Kind::CollapsedAt,
                                                     op, "unknown offset"));
  define(result, *base);
}

std::optional<uint64_t> LocalBuilder::getSubobjectSize(cir::GetMemberOp op) {
  // The members of a union are one region; they get no children.
  auto recordTy = dyn_cast<cir::RecordType>(op.getAddrTy().getPointee());
  if (!recordTy || recordTy.isUnion() || recordTy.isIncomplete())
    return std::nullopt;
  uint64_t index = op.getIndex();
  if (index >= recordTy.getNumElements())
    return std::nullopt;
  // An empty member holds no byte (and [[no_unique_address]] may overlap it
  // with another member).
  ArrayRef<cir::RecordMemberKind> kinds = recordTy.getMemberKinds();
  if (index < kinds.size() && kinds[index] != cir::RecordMemberKind::Data)
    return std::nullopt;
  Type member = recordTy.getElementType(index);
  if (auto memberRecord = dyn_cast<cir::RecordType>(member)) {
    if (memberRecord.isIncomplete())
      return std::nullopt;
  } else if (!isa<cir::ArrayType>(member)) {
    return std::nullopt; // A scalar or a pointer: a cell of the parent.
  }
  std::optional<int64_t> size = getTypeSizeInBytes(member, dataLayout);
  if (!size || *size <= 0)
    return std::nullopt;
  return static_cast<uint64_t>(*size);
}

void LocalBuilder::collapseIfLeavingSubobject(Operation *op, Cell base,
                                              std::optional<int64_t> delta,
                                              std::optional<int64_t> elemSize) {
  Graph &graph = fg->getGraph();
  Cell from = base.resolve();
  SmallVector<ParentLink, 1> parents = graph.getParents(from.getNode());
  const Reason *why = nullptr;
  for (const ParentLink &parent : parents) {
    int64_t start = parent.childOffset;
    int64_t end = start + static_cast<int64_t>(parent.size);
    bool leaves;
    if (delta) {
      // One past the end is still a pointer into the subobject.
      int64_t to = from.getOffset() + *delta;
      leaves = to < start || to > end;
    } else {
      // A dynamic index over elements larger than the subobject.
      leaves = !elemSize || *elemSize > end - start;
    }
    if (!leaves)
      continue;
    // SOUNDNESS: the program computes a pointer outside the member from a
    // pointer to it (char* arithmetic, a cast to a derived class), which only
    // a parent's node can describe: the parent and all its children become
    // one field-insensitive node.
    if (!why)
      why = graph.makeReason(Reason::Kind::CollapsedAt, op,
                             "pointer arithmetic leaves a subobject");
    graph.collapse(parent.parent, why);
  }
}

/// The byte offset of the address point \p point inside a vtable (group) of
/// type \p type, following the lowering: a GEP [0, index, offset] on the
/// vtable's record type. std::nullopt if the type is not such a record.
static std::optional<int64_t>
getAddressPointOffset(Type type, cir::AddressPointAttr point,
                      const DataLayout &dataLayout) {
  auto recordTy = dyn_cast_if_present<cir::RecordType>(type);
  if (!recordTy || recordTy.isIncomplete() || point.getIndex() < 0 ||
      static_cast<size_t>(point.getIndex()) >= recordTy.getNumElements())
    return std::nullopt;
  unsigned index = point.getIndex();
  auto arrayTy = dyn_cast<cir::ArrayType>(recordTy.getElementType(index));
  if (!arrayTy || point.getOffset() < 0 ||
      static_cast<uint64_t>(point.getOffset()) > arrayTy.getSize())
    return std::nullopt;
  std::optional<int64_t> elemSize =
      getTypeSizeInBytes(arrayTy.getElementType(), dataLayout);
  if (!elemSize)
    return std::nullopt;
  return static_cast<int64_t>(recordTy.getElementOffset(dataLayout, index)) +
         point.getOffset() * *elemSize;
}

void LocalBuilder::visitVTableAddrPoint(cir::VTableAddrPointOp op) {
  // The address point is a cell of the vtable global's node, whose slots
  // GlobalNodes linked to the Function nodes of the #cir.vtable initializer.
  Cell vtable(globals->getNode(op.getNameAttr()));
  std::optional<int64_t> offset =
      getAddressPointOffset(globals->getSymbolType(op.getNameAttr()),
                            op.getAddressPoint(), dataLayout);
  if (!offset) {
    Graph &graph = fg->getGraph();
    graph.collapse(vtable.getNode(),
                   graph.makeReason(Reason::Kind::CollapsedAt, op,
                                    "vtable address point not understood"));
    define(op.getAddr(), vtable);
    return;
  }
  define(op.getAddr(), shift(vtable, *offset));
}

void LocalBuilder::visitVTTAddrPoint(cir::VTTAddrPointOp op) {
  // The VTT is an array of vtable address points: entry `offset` of the VTT
  // global, or of the sub-VTT the operand points to.
  Graph &graph = fg->getGraph();
  int64_t step = static_cast<int64_t>(op.getOffset()) * graph.getPointerSize();
  if (Value addr = op.getSymAddr()) {
    if (!isPointerType(addr.getType())) {
      visitUnhandled(op, "VTT address that is not a pointer");
      return;
    }
    std::optional<Cell> base = cellFor(addr);
    define(op.getAddr(), base ? shift(*base, step) : Cell(graph.makeNode()));
    return;
  }
  FlatSymbolRefAttr name = op.getNameAttr();
  if (!name) {
    visitUnhandled(op, "VTT address point without a VTT");
    return;
  }
  Cell vtt(globals->getNode(name));
  // The lowering is a GEP [0, offset] on the VTT's type: an array of
  // pointers. Any other type: step by its element size if it is an array,
  // else smash the node.
  std::optional<int64_t> elemSize;
  if (auto arrayTy =
          dyn_cast_if_present<cir::ArrayType>(globals->getSymbolType(name)))
    elemSize = getTypeSizeInBytes(arrayTy.getElementType(), dataLayout);
  if (!elemSize) {
    graph.collapse(vtt.getNode(),
                   graph.makeReason(Reason::Kind::CollapsedAt, op,
                                    "VTT of an unexpected type"));
    define(op.getAddr(), vtt);
    return;
  }
  define(op.getAddr(), shift(vtt, op.getOffset() * *elemSize));
}

void LocalBuilder::visitVPtrStep(Operation *op, Value vptr, int64_t offset) {
  // A vptr points to an address point inside a vtable; a slot of it is a
  // constant number of pointers away. A load from the result reads the
  // slot's link: the Function node of the virtual function, or the RTTI.
  std::optional<Cell> base = cellFor(vptr);
  if (!base) {
    // A null vptr points to nothing (using it is UB).
    define(op->getResult(0), Cell(fg->getGraph().makeNode()));
    return;
  }
  define(op->getResult(0), shift(*base, offset));
}

void LocalBuilder::visitGetMethod(cir::GetMethodOp op) {
  // Itanium C++ ABI 2.3: a pointer to a non-virtual member function holds
  // the function's address and a this-adjustment, which is zero for a
  // constant naming a member of the class itself (conversions are separate
  // cir.base_method / cir.derived_method ops). Anything else (a virtual
  // member, a method pointer read from memory or converted) stays
  // pessimistic.
  auto constant = op.getMethod().getDefiningOp<cir::ConstantOp>();
  auto method =
      constant ? dyn_cast<cir::MethodAttr>(constant.getValue()) : nullptr;
  if (!method || !method.getSymbol()) {
    visitUnhandled(op, "method pointer not a constant non-virtual member");
    return;
  }
  define(op.getCallee(), Cell(globals->getNode(*method.getSymbol())));
  if (std::optional<Cell> object = cellFor(op.getObject()))
    define(op.getAdjustedThis(), *object);
}

void LocalBuilder::visitCast(cir::CastOp op) {
  Graph &graph = fg->getGraph();
  switch (op.getKind()) {
  case cir::CastKind::bitcast:
  case cir::CastKind::address_space:
  case cir::CastKind::array_to_ptrdecay:
    // The address doesn't change (array_to_ptrdecay points to element 0).
    // Handled here rather than by decomposePointer, which only looks through
    // casts from a cir.ptr, not from a !cir.vptr.
    if (isPointerType(op.getSrc().getType()) && isPointerType(op.getType())) {
      defineCopy(op.getResult(), op.getSrc());
      return;
    }
    break;
  case cir::CastKind::int_to_ptr: {
    // The integer may hold any address, including ones of objects we never
    // see.
    Cell cell(graph.makeNode(NodeFlags::UnknownObj));
    graph.addFlags(cell, NodeFlags::External | NodeFlags::Incomplete,
                   graph.makeReason(Reason::Kind::EscapedTo, op,
                                    "pointer made from an integer"));
    define(op.getResult(), cell);
    return;
  }
  case cir::CastKind::ptr_to_int:
    if (std::optional<Cell> src = cellFor(op.getSrc()))
      graph.addFlags(*src, NodeFlags::External,
                     graph.makeReason(Reason::Kind::EscapedTo, op,
                                      "address escaped as an integer"));
    return;
  case cir::CastKind::ptr_to_bool:
    return; // No pointer semantics.
  default:
    break;
  }
  if (hasPointers(op.getSrc().getType()) || hasPointers(op.getType()))
    visitUnhandled(op, "cast kind");
}

void LocalBuilder::visitCollapse(Operation *op, Value src) {
  std::optional<Cell> base = cellFor(src);
  Graph &graph = fg->getGraph();
  if (!base) {
    define(op->getResult(0), Cell(graph.makeNode()));
    return;
  }
  graph.collapse(base->getNode(), graph.makeReason(Reason::Kind::CollapsedAt,
                                                   op, "unknown offset"));
  define(op->getResult(0), *base);
}

void LocalBuilder::visitAtomic(Operation *op, Value addrValue, Value stored,
                               Value loaded) {
  std::optional<Cell> addr = cellFor(addrValue);
  if (!addr)
    return;
  Graph &graph = fg->getGraph();
  graph.addFlags(*addr, NodeFlags::Mod | NodeFlags::Ref);
  if (std::optional<int64_t> size =
          getTypeSizeInBytes(stored.getType(), dataLayout))
    graph.growSize(addr->getNode(), addr->getOffset() + *size);
  if (!isPointerType(stored.getType()))
    return;
  if (std::optional<Cell> target = cellFor(stored))
    graph.setLink(*addr, *target,
                  graph.makeReason(Reason::Kind::CollapsedAt, op));
  define(loaded, graph.getLink(*addr));
}

void LocalBuilder::visitCall(CIRCallOpInterface op) {
  CallSiteRecord site;
  site.call = op;
  if (op.isIndirect()) {
    site.calleePtr = cellFor(op.getIndirectCall());
    // A virtual call: the callee is a slot of the vtable an object's vptr
    // points to.
    auto slot = op.getIndirectCall().getDefiningOp<cir::LoadOp>();
    auto fnAddr =
        slot ? slot.getAddr().getDefiningOp<cir::VTableGetVirtualFnAddrOp>()
             : cir::VTableGetVirtualFnAddrOp();
    auto vptr =
        fnAddr ? fnAddr.getVptr().getDefiningOp<cir::LoadOp>() : cir::LoadOp();
    auto getVPtr = vptr ? vptr.getAddr().getDefiningOp<cir::VTableGetVPtrOp>()
                        : cir::VTableGetVPtrOp();
    if (getVPtr)
      if (std::optional<Cell> holder = cellFor(getVPtr.getSrc())) {
        site.vptrHolder = holder;
        site.vtableSlot = static_cast<int64_t>(fnAddr.getIndex()) *
                          fg->getGraph().getPointerSize();
      }
  } else
    site.callee = op->getAttrOfType<FlatSymbolRefAttr>("callee");
  for (unsigned i = 0, e = op.getNumArgOperands(); i != e; ++i) {
    Value arg = op.getArgOperand(i);
    site.args.push_back(cellFor(arg));
    if (std::optional<AggregateSlots> slots = slotsFor(arg))
      site.aggregateArgs.emplace_back(i, std::move(*slots));
  }
  Graph &graph = fg->getGraph();
  for (Value result : op->getResults()) {
    if (isPointerType(result.getType())) {
      Cell cell(graph.makeNode());
      define(result, cell);
      site.result = cell;
      continue;
    }
    SmallVector<int64_t, 2> offsets = getPointerOffsets(result.getType());
    if (offsets.empty())
      continue;
    AggregateSlots slots;
    for (int64_t off : offsets)
      slots.emplace_back(off, Cell(graph.makeNode()));
    defineSlots(result, slots);
    site.resultSlots = std::move(slots);
  }
  fg->callSites.push_back(std::move(site));
  size_t index = fg->callSites.size() - 1;
  if (op.isIndirect()) {
    indirectCalls.push_back(index);
    return;
  }
  dispatchCall(index, op);
}

void LocalBuilder::dispatchCall(size_t index, CIRCallOpInterface call) {
  CallSiteRecord &site = fg->callSites[index];
  Graph &graph = fg->getGraph();
  cir::FuncOp callee;
  if (site.callee)
    callee =
        symbols->lookupNearestSymbolFrom<cir::FuncOp>(site.call, site.callee);
  StringRef name = site.callee ? site.callee.getValue() : StringRef();
  if (models)
    if (const PointsToModel *model = models->lookup(call, callee, name)) {
      ModelContext ctx(*fg, dataLayout, deferredEffects);
      SmallVector<CallSiteRecord, 0> extraCalls;
      ctx.setLaunchSink(&fg->launches);
      ctx.setCallSink(&extraCalls);
      model->apply(site, ctx);
      site.kind = CallKind::Modelled;
      site.modelled = model->name();
      // The calls the model makes: indirect, so decided after the walk (or
      // by the interprocedural phase), like the ones the body spells.
      for (CallSiteRecord &extra : extraCalls) {
        fg->callSites.push_back(std::move(extra));
        indirectCalls.push_back(fg->callSites.size() - 1);
      }
      return;
    }
  if (callee && !callee.isDeclaration()) {
    site.kind = CallKind::Defined; // T08 connects the body.
    return;
  }
  site.kind = CallKind::UnknownDecl;
  applyUnknownCall(site, graph.makeReason(Reason::Kind::EscapedTo, site.call,
                                          ("@" + name).str()));
}

void LocalBuilder::applyUnknownCall(CallSiteRecord &site, const Reason *why) {
  // SOUNDNESS: code we don't see may read, write, keep or publish anything
  // reachable from what it is given, and return any pointer.
  ModelContext::forEachPointerArg(site, [&](Cell cell) {
    deferredEffects.push_back(
        {DeferredEffect::Kind::Reachable, cell,
         NodeFlags::External | NodeFlags::Mod | NodeFlags::Ref, why});
  });
  ModelContext ctx(*fg, dataLayout, deferredEffects);
  ctx.setResultUnknown(site, NodeFlags::External | NodeFlags::Incomplete, why);
  // And it may read or write any global another TU can name; the globals
  // graph (T08) applies this.
  site.clobbersGlobals = true;
}

/// Whether the callee pointer of an indirect call points only to functions
/// known by symbol, so that the interprocedural phase can bind the targets.
static bool isResolvedCallee(std::optional<Cell> calleePtr) {
  if (!calleePtr)
    return false;
  const Node *node = calleePtr->getNode();
  if (!node || node->getOrigins().empty())
    return false;
  if (hasAny(node->getFlags(), NodeFlags::UnknownObj | NodeFlags::Incomplete |
                                   NodeFlags::External))
    return false;
  return llvm::all_of(node->getOrigins(), [](const Origin &origin) {
    return origin.kind == Origin::Kind::FuncSym;
  });
}

void LocalBuilder::resolveIndirectCalls() {
  // An unresolved call makes what its arguments reach External, which may
  // in turn make another callee pointer External: iterate until no call
  // changes. Flags only grow, so this ends. A resolved call gets, now, the
  // effect of each target that is a declaration (its model or the
  // unknown-call rule); its defined targets are left to the interprocedural
  // phase.
  SmallVector<size_t, 0> pending = std::move(indirectCalls);
  indirectCalls.clear();
  llvm::DenseSet<size_t> applied;
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::erase_if(pending, [&](size_t index) {
      CallSiteRecord &site = fg->callSites[index];
      if (isResolvedCallee(site.calleePtr)) {
        if (applied.insert(index).second) {
          applyResolvedTargets(site);
          changed = true;
        }
        return false;
      }
      site.kind = CallKind::IndirectUnknown;
      applyUnknownCall(
          site, fg->getGraph().makeReason(Reason::Kind::IndirectUnresolved,
                                          site.call, "callee not known"));
      changed = true;
      return true;
    });
    applyDeferredEffects();
  }
  for (size_t index : pending)
    fg->callSites[index].kind = CallKind::IndirectResolved;
}

void LocalBuilder::applyResolvedTargets(CallSiteRecord &site) {
  auto call = cast<CIRCallOpInterface>(site.call);
  SmallVector<FlatSymbolRefAttr, 2> targets;
  for (const Origin &origin : site.calleePtr->getNode()->getOrigins())
    if (auto sym = dyn_cast_if_present<FlatSymbolRefAttr>(origin.sym))
      targets.push_back(sym);
  for (FlatSymbolRefAttr target : targets) {
    auto callee = symbols->lookupNearestSymbolFrom<cir::FuncOp>(func, target);
    if (models)
      if (const PointsToModel *model =
              models->lookup(call, callee, target.getValue())) {
        // The model reads the callee from the site: give it the target. The
        // cells are shared, so the copy's effects are the site's.
        CallSiteRecord targetSite = site;
        targetSite.callee = target;
        ModelContext ctx(*fg, dataLayout, deferredEffects);
        model->apply(targetSite, ctx);
        site.clobbersGlobals |= targetSite.clobbersGlobals;
        site.modelled = model->name();
        continue;
      }
    if (callee && !callee.isDeclaration())
      continue; // T08 binds the body.
    applyUnknownCall(
        site, fg->getGraph().makeReason(Reason::Kind::EscapedTo, site.call,
                                        ("@" + target.getValue()).str()));
  }
}

void LocalBuilder::applyDeferredEffects() {
  Graph &graph = fg->getGraph();
  SmallVector<DeferredEffect, 0> effects = std::move(deferredEffects);
  deferredEffects.clear();
  for (const DeferredEffect &effect : effects) {
    switch (effect.kind) {
    case DeferredEffect::Kind::Reachable:
      graph.markReachable(effect.cell, effect.flags, effect.why);
      break;
    }
  }
}

void LocalBuilder::visitGetGlobal(cir::GetGlobalOp op) {
  define(op.getAddr(), Cell(globals->getNode(op.getNameAttr())));
}

void LocalBuilder::smashIfTruncated(Cell addr, Type type, Operation *op) {
  PointerLayout layout = getPointerLayout(type);
  if (!layout.truncation.truncated)
    return;
  Graph &graph = fg->getGraph();
  const Reason *why =
      graph.makeReason(Reason::Kind::CollapsedAt, op, "array too long to scan");
  if (layout.truncation.arrayElemSize)
    graph.markArray(addr.getNode(), *layout.truncation.arrayElemSize, why);
  else
    graph.collapse(addr.getNode(), why);
}

void LocalBuilder::copyFieldInsensitive(Cell dst, Cell src, Operation *op,
                                        StringRef detail) {
  Graph &graph = fg->getGraph();
  const Reason *why = graph.makeReason(Reason::Kind::CollapsedAt, op, detail);
  graph.collapse(dst.getNode(), why);
  graph.collapse(src.getNode(), why);
  graph.unify(graph.getLink(dst, why), graph.getLink(src, why), why);
}

void LocalBuilder::visitCopy(cir::CopyOp op) {
  std::optional<Cell> dst = cellFor(op.getDst());
  std::optional<Cell> src = cellFor(op.getSrc());
  if (!dst || !src)
    return; // Copying from or to null is UB.
  Graph &graph = fg->getGraph();
  graph.addFlags(*dst, NodeFlags::Mod);
  graph.addFlags(*src, NodeFlags::Ref);
  Type type = op.getType().getPointee();
  if (!getTypeSizeInBytes(type, dataLayout)) {
    copyFieldInsensitive(*dst, *src, op, "copy of a type of unknown size");
    return;
  }
  uint64_t length = op.getCopySizeInBytes(dataLayout);
  graph.growSize(dst->getNode(), dst->getOffset() + length);
  graph.growSize(src->getNode(), src->getOffset() + length);

  // A typed load of the pointee from src plus a store of it to dst: the
  // pointer slots of the type link to the same targets on both sides. The
  // links exist from now on, so pointers stored into either side later in
  // the walk are seen by both.
  smashIfTruncated(*dst, type, op);
  smashIfTruncated(*src, type, op);
  const Reason *why = nullptr;
  for (int64_t off : getPointerOffsets(type)) {
    if (off >= static_cast<int64_t>(length))
      break; // Tail padding that skip_tail_padding leaves alone.
    if (!why)
      why = graph.makeReason(Reason::Kind::CollapsedAt, op);
    graph.setLink(shift(*dst, off), graph.getLink(shift(*src, off)), why);
  }
  // Pointers stored where the type has none (a byte buffer holding an object
  // constructed in place, a union member the type does not show) are copied
  // too: by the byte-range closure after the walk.
  fg->memTransfers.push_back({op, *dst, *src, length});
}

void cir::pointsto::recordMemTransfer(FunctionGraph &fg, Operation *op,
                                      Cell dst, Cell src,
                                      std::optional<int64_t> length) {
  recordMemTransfer(fg.getGraph(), fg.memTransfers, op, dst, src, length);
}

void cir::pointsto::recordMemTransfer(
    Graph &graph, SmallVectorImpl<MemTransferRecord> &transfers, Operation *op,
    Cell dst, Cell src, std::optional<int64_t> length, bool ghosts) {
  graph.addFlags(dst, NodeFlags::Mod);
  graph.addFlags(src, NodeFlags::Ref);
  // Like cir.copy over the bytes [0, length), without a type: done by the
  // closure after the walk, once every link of the function is known. An
  // unknown length is a range without an end: the closure relates every
  // offset at or after the two cells, which over-approximates any length.
  if (length && *length <= 0)
    return;
  if (length) {
    graph.growSize(dst.getNode(), dst.getOffset() + *length);
    graph.growSize(src.getNode(), src.getOffset() + *length);
  }
  transfers.push_back(
      {op, dst, src,
       length ? static_cast<uint64_t>(*length) : MemTransferRecord::kUnknown,
       ghosts});
}

void LocalBuilder::visitMemTransfer(Operation *op, Value dstValue,
                                    Value srcValue, Value lengthValue) {
  std::optional<Cell> dst = cellFor(dstValue);
  std::optional<Cell> src = cellFor(srcValue);
  if (!dst || !src)
    return; // Copying from or to null is UB.
  recordMemTransfer(*fg, op, *dst, *src, getConstantIndex(lengthValue));
}

void LocalBuilder::visitMemChr(cir::MemChrOp op) {
  Graph &graph = fg->getGraph();
  std::optional<Cell> src = cellFor(op.getSrc());
  if (!src) {
    define(op.getResult(), Cell(graph.makeNode()));
    return;
  }
  graph.addFlags(*src, NodeFlags::Ref);
  // The result points into the searched range at an unknown offset.
  graph.collapse(src->getNode(),
                 graph.makeReason(Reason::Kind::CollapsedAt, op,
                                  "pointer into the searched range"));
  define(op.getResult(), *src);
}

void LocalBuilder::visitStdFind(cir::StdFindOp op) {
  // C++20 [alg.find]: the result is the first iterator in [first, last) that
  // matches, or last. Either way it points into the range, at an offset we
  // don't know: the node of `first`, collapsed, which `last` points into too.
  Graph &graph = fg->getGraph();
  const Reason *why = graph.makeReason(Reason::Kind::CollapsedAt, op,
                                       "iterator into the searched range");
  if (isPointerType(op.getPattern().getType()))
    if (std::optional<Cell> pattern = cellFor(op.getPattern()))
      graph.addFlags(*pattern, NodeFlags::Ref);
  auto relate = [&](Cell first, Cell last, Cell result) {
    graph.addFlags(first, NodeFlags::Ref);
    graph.collapse(first.getNode(), why);
    if (last)
      graph.unify(first, last, why);
    graph.unify(first, result, why);
  };
  Value result = op.getResult();
  if (isPointerType(result.getType())) {
    std::optional<Cell> first = cellFor(op.getFirst());
    std::optional<Cell> last = cellFor(op.getLast());
    if (!first) {
      // An empty range starting at null: the result is `last`.
      defineCopy(result, op.getLast());
      return;
    }
    Cell cell(graph.makeNode());
    relate(*first, last ? *last : Cell(), cell);
    define(result, cell);
    return;
  }
  std::optional<AggregateSlots> first = slotsFor(op.getFirst());
  std::optional<AggregateSlots> last = slotsFor(op.getLast());
  if (!first || !last)
    return; // No pointers in the iterator.
  AggregateSlots slots;
  for (auto [f, l] : llvm::zip(*first, *last)) {
    Cell cell(graph.makeNode());
    if (f.second)
      relate(f.second, l.second, cell);
    else if (l.second)
      graph.unify(l.second, cell, why);
    slots.emplace_back(f.first, cell);
  }
  defineSlots(result, slots);
}

void LocalBuilder::visitArrayInit(Operation *op, Value addr,
                                  Value numElements) {
  Graph &graph = fg->getGraph();
  std::optional<Cell> cell = cellFor(addr);
  Region &body = op->getRegion(0);
  Type elemPtrType = body.empty() || body.front().getNumArguments() == 0
                         ? Type()
                         : body.front().getArgument(0).getType();
  if (cell) {
    graph.addFlags(*cell, NodeFlags::Mod | NodeFlags::Ref);
    // The constructor (destructor) runs on each element: the elements share
    // one abstract element, as after a dynamic index. A single element needs
    // no array.
    auto arrayTy = dyn_cast<cir::ArrayType>(
        cast<cir::PointerType>(addr.getType()).getPointee());
    bool single = !numElements && arrayTy && arrayTy.getSize() == 1;
    std::optional<int64_t> elemSize;
    if (auto elemPtrTy = dyn_cast_if_present<cir::PointerType>(elemPtrType))
      elemSize = getTypeSizeInBytes(elemPtrTy.getPointee(), dataLayout);
    if (arrayTy)
      if (std::optional<int64_t> total =
              getTypeSizeInBytes(arrayTy, dataLayout))
        graph.growSize(cell->getNode(), cell->getOffset() + *total);
    const Reason *why = graph.makeReason(Reason::Kind::CollapsedAt, op,
                                         "constructed element by element");
    if (!single) {
      if (elemSize && *elemSize > 0)
        graph.markArray(cell->getNode(), *elemSize, why);
      else
        graph.collapse(cell->getNode(), why);
    }
  }
  // The element pointer of each region is the array's cell, so the call in
  // the region is recorded with `this` = the array.
  for (Region &region : op->getRegions()) {
    if (region.empty())
      continue;
    for (BlockArgument arg : region.front().getArguments()) {
      if (!isPointerType(arg.getType()))
        continue;
      define(arg, cell ? *cell : Cell(graph.makeNode()));
    }
  }
}

void LocalBuilder::visitDeleteArray(cir::DeleteArrayOp op) {
  Graph &graph = fg->getGraph();
  std::optional<Cell> cell = cellFor(op.getAddress());
  if (!cell)
    return; // delete[] of null does nothing.
  auto addSite = [&](FlatSymbolRefAttr callee) {
    CallSiteRecord site;
    site.call = op;
    site.callee = callee;
    site.synthetic = true;
    site.args.push_back(*cell);
    fg->callSites.push_back(std::move(site));
    dispatchCall(fg->callSites.size() - 1, CIRCallOpInterface());
  };
  if (FlatSymbolRefAttr dtor = op.getElementDtorAttr()) {
    // The destructor runs on each element (C++20 [expr.delete]p6).
    Type elemType =
        cast<cir::PointerType>(op.getAddress().getType()).getPointee();
    std::optional<int64_t> elemSize = getTypeSizeInBytes(elemType, dataLayout);
    const Reason *why = graph.makeReason(Reason::Kind::CollapsedAt, op,
                                         "destroyed element by element");
    if (elemSize && *elemSize > 0)
      graph.markArray(cell->getNode(), *elemSize, why);
    else
      graph.collapse(cell->getNode(), why);
    addSite(dtor);
  }
  addSite(op.getDeleteFnAttr());
}

//===----------------------------------------------------------------------===//
// Copy closure
//===----------------------------------------------------------------------===//

/// More (offset) pairs than this in one copy, or more rounds than this over
/// all copies, and the copy is done field-insensitively instead.
static constexpr size_t kMaxTransferSlots = 1024;
static constexpr unsigned kMaxTransferRounds = 64;

/// The offsets k in [0, length) at which the cell \p at (resolved, on a node
/// that is not collapsed) has a link on its node. With \p onePerLink, only
/// the smallest such k per link (on an array node the others are the same
/// slot). False if there are more than kMaxTransferSlots.
static bool collectLinkedOffsets(Cell at, int64_t length,
                                 SmallVectorImpl<int64_t> &out,
                                 bool onePerLink = false) {
  const Node *node = at.getRawNode();
  int64_t base = at.getRawOffset();
  std::optional<uint64_t> elem = node->getArrayElemSize();
  for (const auto &link : node->getLinks()) {
    int64_t k = link.first - base;
    if (!elem) {
      if (k >= 0 && k < length)
        out.push_back(k);
      continue;
    }
    int64_t e = static_cast<int64_t>(*elem);
    for (k = ((k % e) + e) % e; k < length; k += e) {
      out.push_back(k);
      if (onePerLink)
        break;
      if (out.size() > kMaxTransferSlots)
        return false;
    }
  }
  return out.size() <= kMaxTransferSlots;
}

/// The offsets k in [0, length) at which the cell \p at (resolved, on a node
/// that is not collapsed) is pointer-aligned within its node, one per
/// distinct normalized offset. False if there are more than
/// kMaxTransferSlots.
static bool collectAlignedOffsets(Cell at, int64_t length, int64_t pointerSize,
                                  SmallVectorImpl<int64_t> &out) {
  const Node *node = at.getRawNode();
  int64_t base = at.getRawOffset();
  int64_t k = ((-base % pointerSize) + pointerSize) % pointerSize;
  // On an array node the offsets repeat every element: the ones below one
  // element are a representative of each.
  std::optional<uint64_t> elem = node->getArrayElemSize();
  int64_t end = elem ? std::min<int64_t>(length, *elem) : length;
  for (; k < end; k += pointerSize) {
    out.push_back(k);
    if (out.size() > kMaxTransferSlots)
      return false;
  }
  return true;
}

namespace {
/// Applies one MemTransferRecord.
struct TransferApplier {
  Graph &graph;
  const MemTransferRecord &record;
  const Reason *why = nullptr;

  const Reason *reason() {
    if (!why)
      why = graph.makeReason(Reason::Kind::CollapsedAt, record.op,
                             "copied range");
    return why;
  }

  /// Unify the targets of the slots \p a and \p b. Return whether that
  /// changed the graph.
  bool unifySlots(Cell a, Cell b) {
    std::optional<Cell> la = graph.lookupLink(a);
    std::optional<Cell> lb = graph.lookupLink(b);
    if (la && lb && la->resolve() == lb->resolve())
      return false;
    graph.unify(graph.getLink(a, reason()), graph.getLink(b, reason()),
                reason());
    return true;
  }

  /// Collapse both sides and unify their single links.
  bool smash() {
    Cell dst = record.dst.resolve(), src = record.src.resolve();
    bool changed =
        !dst.getRawNode()->isCollapsed() || !src.getRawNode()->isCollapsed();
    graph.collapse(dst.getRawNode(), reason());
    graph.collapse(src.getRawNode(), reason());
    return unifySlots(record.dst, record.src) || changed;
  }

  /// Copy the ghost links in the range: a copy of an object's bytes is the
  /// same object to a model (a SYCL handle copied by memcpy refers to the
  /// same storage). A ghost of one side at offset k of the range is set at
  /// k on the other side, for every k an array side allows. A collapsed side
  /// holds its ghosts at every offset: each ghost the other side has, at any
  /// offset, is unified with the collapsed side's ghost of that field. (A
  /// ghost the other side lacks stays missing, which a client must read as
  /// unknown.) Too many offsets, and the ghost targets of both sides become
  /// External: what they describe is unknown from now on.
  bool applyGhosts() {
    Cell dst = record.dst.resolve(), src = record.src.resolve();
    Node *dstNode = dst.getRawNode(), *srcNode = src.getRawNode();
    if (!dstNode->getNumGhostLinks() && !srcNode->getNumGhostLinks())
      return false;
    int64_t length = static_cast<int64_t>(
        std::min<uint64_t>(record.length, std::numeric_limits<int64_t>::max()));
    bool changed = false;
    auto relate = [&](Cell at, GhostField field, Cell target) {
      std::optional<Cell> have = graph.lookupGhostLink(at, field);
      if (have && have->resolve() == target.resolve())
        return;
      graph.setGhostLink(at, field, target, reason());
      changed = true;
    };
    bool dstCollapsed = dstNode->isCollapsed();
    bool srcCollapsed = srcNode->isCollapsed();
    if (dstCollapsed || srcCollapsed) {
      Cell collapsed = dstCollapsed ? dst : src;
      Cell other = dstCollapsed ? src : dst;
      for (const GhostLink &ghost : other.getRawNode()->getGhostLinks())
        relate(collapsed, ghost.field, ghost.target);
      if (dstCollapsed && srcCollapsed)
        for (const GhostLink &ghost : srcNode->getGhostLinks())
          relate(dst, ghost.field, ghost.target);
      return changed;
    }
    // The offsets k in [0, length) at which a ghost at `offset` on the node
    // of `at` sits, relative to `at`.
    SmallVector<std::tuple<int64_t, GhostField, Cell, bool>, 8> moves;
    auto collect = [&](Cell at, bool toDst) {
      const Node *node = at.getRawNode();
      std::optional<uint64_t> elem = node->getArrayElemSize();
      for (const GhostLink &ghost : node->getGhostLinks()) {
        int64_t k = ghost.offset - at.getRawOffset();
        if (!elem) {
          if (k >= 0 && k < length)
            moves.emplace_back(k, ghost.field, ghost.target, toDst);
          continue;
        }
        int64_t e = static_cast<int64_t>(*elem);
        for (k = ((k % e) + e) % e; k < length; k += e) {
          moves.emplace_back(k, ghost.field, ghost.target, toDst);
          if (moves.size() > kMaxTransferSlots)
            return false;
        }
      }
      return true;
    };
    if (!collect(src, /*toDst=*/true) || !collect(dst, /*toDst=*/false)) {
      const Reason *external = graph.makeReason(
          Reason::Kind::EscapedTo, record.op,
          "ghost link copied by a byte copy that cannot be placed");
      for (Node *node : {dstNode, srcNode})
        for (const GhostLink &ghost : node->getGhostLinks())
          if (!ghost.target.getNode()->hasFlags(NodeFlags::External)) {
            graph.markReachable(ghost.target, NodeFlags::External, external);
            changed = true;
          }
      return changed;
    }
    for (auto [k, field, target, toDst] : moves)
      relate(shift(toDst ? record.dst : record.src, k), field, target);
    return changed;
  }

  bool apply() { return applyLinks() | (record.ghosts && applyGhosts()); }

  bool applyLinks() {
    Cell dst = record.dst.resolve(), src = record.src.resolve();
    int64_t length = static_cast<int64_t>(
        std::min<uint64_t>(record.length, std::numeric_limits<int64_t>::max()));
    bool dstCollapsed = dst.getRawNode()->isCollapsed();
    bool srcCollapsed = src.getRawNode()->isCollapsed();
    if (dstCollapsed && srcCollapsed)
      return unifySlots(dst, src);

    SmallVector<int64_t> offsets;
    if (dstCollapsed || srcCollapsed) {
      // The collapsed side holds its pointers at every offset, so every slot
      // of the other side in the range gets its link: the ones it has and,
      // if the collapsed side holds a pointer, every pointer-aligned one.
      Cell collapsed = dstCollapsed ? dst : src;
      Cell other = dstCollapsed ? src : dst;
      if (!collectLinkedOffsets(other, length, offsets, /*onePerLink=*/true))
        return smash();
      if (graph.lookupLink(collapsed) &&
          !collectAlignedOffsets(other, length, graph.getPointerSize(),
                                 offsets))
        return smash();
      llvm::sort(offsets);
      offsets.erase(llvm::unique(offsets), offsets.end());
      bool changed = false;
      for (int64_t k : offsets)
        changed |= unifySlots(shift(other, k), collapsed);
      return changed;
    }

    if (!collectLinkedOffsets(dst, length, offsets) ||
        !collectLinkedOffsets(src, length, offsets) ||
        offsets.size() > kMaxTransferSlots) {
      // Too many offsets: a long (or unknown) range over an array. The
      // offsets of a range over arrays repeat every element, so make both
      // sides arrays of one element size and take one offset per class.
      offsets.clear();
      std::optional<uint64_t> dstElem = dst.getRawNode()->getArrayElemSize();
      std::optional<uint64_t> srcElem = src.getRawNode()->getArrayElemSize();
      if ((!dstElem && !srcElem) ||
          (dstElem && srcElem && *dstElem != *srcElem))
        return smash();
      if (!dstElem || !srcElem) {
        graph.markArray((dstElem ? src : dst).getRawNode(),
                        dstElem ? *dstElem : *srcElem, reason());
        return true; // Applied as two arrays in the next round.
      }
      int64_t elem = static_cast<int64_t>(*dstElem);
      for (Cell side : {dst, src})
        for (const auto &link : side.getRawNode()->getLinks()) {
          int64_t k = ((link.first - side.getRawOffset()) % elem + elem) % elem;
          if (k < length)
            offsets.push_back(k);
        }
    }
    llvm::sort(offsets);
    offsets.erase(llvm::unique(offsets), offsets.end());
    bool changed = false;
    for (int64_t k : offsets)
      changed |= unifySlots(shift(record.dst, k), shift(record.src, k));
    return changed;
  }
};
} // namespace

namespace {
/// The copies derived from the subobject links in the ranges of the copies:
/// a copy of an object is a copy of each of its children into the matching
/// child of the other side (made there if it is missing), recursively, plus
/// one of the node's own cells. A child the range covers only partly is
/// copied over the part it covers.
struct SubobjectTransfers {
  Graph &graph;
  SmallVector<MemTransferRecord, 0> records;
  llvm::DenseSet<
      std::tuple<const Node *, int64_t, const Node *, int64_t, uint64_t>>
      seen;

  /// Add the copy of \p length bytes from \p src to \p dst made for
  /// \p record, unless it is there. Return whether it was added.
  bool add(const MemTransferRecord &record, Cell dst, Cell src,
           uint64_t length) {
    dst = dst.resolve();
    src = src.resolve();
    if (dst == src)
      return false;
    if (!seen.insert({dst.getRawNode(), dst.getRawOffset(), src.getRawNode(),
                      src.getRawOffset(), length})
             .second)
      return false;
    records.push_back({record.op, dst, src, length, record.ghosts});
    return true;
  }

  /// Derive the copies of the children in the ranges of every record (the
  /// derived ones included). Return whether a record was added.
  bool expand() {
    bool grew = false;
    for (size_t i = 0; i < records.size(); ++i) {
      MemTransferRecord record = records[i];
      int64_t length = static_cast<int64_t>(std::min<uint64_t>(
          record.length, std::numeric_limits<int64_t>::max()));
      for (bool fromDst : {false, true}) {
        Cell x = (fromDst ? record.dst : record.src).resolve();
        Cell y = fromDst ? record.src : record.dst;
        const Node *node = x.getRawNode();
        if (!node->getNumSubobjects())
          continue;
        std::optional<uint64_t> elem = node->getArrayElemSize();
        for (const SubobjectLink &sub : node->getSubobjects()) {
          // The positions of the child relative to x: once, or once per
          // element the range covers on an array node.
          int64_t first = sub.offset - x.getRawOffset();
          int64_t step = elem ? static_cast<int64_t>(*elem) : 0;
          auto size = static_cast<int64_t>(sub.size);
          if (step && length / step > static_cast<int64_t>(kMaxTransferSlots)) {
            // Too many elements to list: the other side becomes an array of
            // the same elements, where one position stands for all.
            graph.markArray(y.getNode(), *elem,
                            graph.makeReason(Reason::Kind::CollapsedAt,
                                             record.op, "copied range"));
            step = 0;
          }
          for (int64_t k = first; k < length; k += step) {
            if (k + size > 0) {
              Cell xc, yc;
              uint64_t n;
              if (k >= 0 && k + size <= length) {
                xc = sub.child;
                yc = graph.getSubobject(
                    Cell(y.getRawNode(), y.getRawOffset() + k), sub.size);
                n = sub.size;
              } else {
                int64_t a = std::max<int64_t>(k, 0);
                int64_t b = std::min<int64_t>(k + size, length);
                xc = Cell(sub.child.getRawNode(),
                          sub.child.getRawOffset() + (a - k));
                yc = Cell(y.getRawNode(), y.getRawOffset() + a);
                n = static_cast<uint64_t>(b - a);
              }
              grew |= fromDst ? add(record, xc, yc, n) : add(record, yc, xc, n);
            }
            if (!step)
              break;
          }
        }
      }
    }
    return grew;
  }
};
} // namespace

bool cir::pointsto::closeMemTransfers(Graph &graph,
                                      ArrayRef<MemTransferRecord> transfers) {
  // Subobjects (PointsToGraph.h): the copies of the children in the ranges
  // are copies too, made as the children appear.
  std::optional<SubobjectTransfers> subobjects;
  if (graph.hasSubobjects()) {
    subobjects.emplace(SubobjectTransfers{graph, {}, {}});
    subobjects->records.assign(transfers.begin(), transfers.end());
  }
  // Unification is symmetric, so this treats dst and src alike: it only
  // ever adds links and merges targets, which over-approximates the
  // one-directional copy. That is the DSA treatment of memcpy, with offsets.
  for (unsigned round = 0; round != kMaxTransferRounds; ++round) {
    bool changed = false;
    if (subobjects) {
      changed |= subobjects->expand();
      transfers = subobjects->records;
    }
    for (const MemTransferRecord &record : transfers)
      changed |= TransferApplier{graph, record}.apply();
    if (!changed)
      return true;
  }
  for (const MemTransferRecord &record : transfers)
    TransferApplier{graph, record}.smash();
  return false;
}

void LocalBuilder::visitReturn(cir::ReturnOp op) {
  Graph &graph = fg->getGraph();
  for (Value value : op.getOperands()) {
    if (std::optional<AggregateSlots> slots = slotsFor(value)) {
      if (!fg->returnSlots) {
        fg->returnSlots = std::move(*slots);
        continue;
      }
      const Reason *why = graph.makeReason(Reason::Kind::MergedAt, op);
      for (auto [have, give] : llvm::zip(*fg->returnSlots, *slots)) {
        if (!give.second)
          continue;
        if (!have.second)
          have.second = give.second;
        else
          graph.unify(have.second, give.second, why);
      }
      continue;
    }
    std::optional<Cell> cell = cellFor(value);
    if (!cell)
      continue;
    if (!fg->returnCell) {
      fg->returnCell = *cell;
      continue;
    }
    graph.unify(*fg->returnCell, *cell,
                graph.makeReason(Reason::Kind::MergedAt, op));
  }
}

void LocalBuilder::visitRegionEntry(Operation *op) {
  auto branch = cast<RegionBranchOpInterface>(op);
  // The results are defined by the terminators that yield them; make sure
  // they have cells even if no terminator does.
  for (Value result : op->getResults())
    definePlaceholder(result);
  SmallVector<Attribute> noConstants(op->getNumOperands());
  SmallVector<RegionSuccessor> successors;
  branch.getEntrySuccessorRegions(noConstants, successors);
  for (RegionSuccessor successor : successors) {
    ValueRange inputs;
    if (successor.isOperation())
      inputs = op->getResults();
    else if (!successor.getSuccessor()->empty())
      inputs = successor.getSuccessor()->front().getArguments();
    OperandRange forwarded = branch.getEntrySuccessorOperands(successor);
    for (auto [operand, input] : llvm::zip(forwarded, inputs))
      if (hasPointers(input.getType()))
        unifyValues(operand, input, op);
  }
}

void LocalBuilder::visitRegionTerminator(Operation *op) {
  auto parent = dyn_cast_if_present<RegionBranchOpInterface>(op->getParentOp());
  if (!parent)
    return;
  auto terminator = cast<RegionBranchTerminatorOpInterface>(op);
  SmallVector<Attribute> noConstants(op->getNumOperands());
  SmallVector<RegionSuccessor> successors;
  terminator.getSuccessorRegions(noConstants, successors);
  for (RegionSuccessor successor : successors) {
    // CIR ops don't implement RegionBranchOpInterface::getSuccessorInputs, so
    // pair the forwarded operands up with the parent's results or the entry
    // arguments of the successor region by position.
    ValueRange inputs;
    if (successor.isOperation())
      inputs = parent->getResults();
    else if (!successor.getSuccessor()->empty())
      inputs = successor.getSuccessor()->front().getArguments();
    OperandRange forwarded = terminator.getSuccessorOperands(successor);
    for (auto [operand, input] : llvm::zip(forwarded, inputs))
      if (hasPointers(input.getType()))
        unifyValues(operand, input, op);
  }
}

void LocalBuilder::visitBranch(Operation *op) {
  auto branch = cast<BranchOpInterface>(op);
  for (unsigned i = 0, e = op->getNumSuccessors(); i != e; ++i) {
    SuccessorOperands operands = branch.getSuccessorOperands(i);
    Block *dest = op->getSuccessor(i);
    unsigned n = std::min<unsigned>(operands.size(), dest->getNumArguments());
    for (unsigned j = 0; j != n; ++j)
      if (Value operand = operands[j])
        unifyValues(operand, dest->getArgument(j), op);
  }
}
