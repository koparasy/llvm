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
//  - Calls: recorded only. A pointer result gets a fresh node, an aggregate
//    result fresh slot nodes.
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
    if (auto callee = site.call->getAttrOfType<FlatSymbolRefAttr>("callee"))
      os << " " << callee;
    if (site.calleePtr) {
      os << " callee=";
      Graph::printCell(os, *site.calleePtr);
    }
    os << " args={";
    bool first = true;
    for (auto [i, arg] : llvm::enumerate(site.args)) {
      if (!isPointerType(
              cast<CIRCallOpInterface>(site.call).getArgOperand(i).getType()))
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
                           ReasonTable *reasons)
    : func(func), dataLayout(getDataLayoutFor(func)), symbols(symbols) {
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

  // Formals: what a caller passes in is not known yet.
  Block &entry = func.getBody().front();
  for (BlockArgument arg : entry.getArguments()) {
    if (!hasPointers(arg.getType())) {
      fg->formals.push_back(std::nullopt);
      continue;
    }
    const Reason *why =
        graph.makeReason(Reason::Kind::UnknownCaller, func,
                         "formal #" + std::to_string(arg.getArgNumber()));
    auto fresh = [&] {
      Cell cell(graph.makeNode());
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
        .Case<cir::StrLenOp>([&](cir::StrLenOp op) {
          visitModRef(op.getString(), {}, /*mod=*/false, /*ref=*/true);
        })
        .Case<cir::VTableGetVPtrOp>([&](cir::VTableGetVPtrOp op) {
          // The vptr is at offset 0 of the object: like a bitcast.
          defineCopy(op.getResult(), op.getSrc());
        })
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

  if (step.offset) {
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
  if (op.isIndirect())
    site.calleePtr = cellFor(op.getIndirectCall());
  for (unsigned i = 0, e = op.getNumArgOperands(); i != e; ++i) {
    Value arg = op.getArgOperand(i);
    site.args.push_back(cellFor(arg));
    if (std::optional<AggregateSlots> slots = slotsFor(arg))
      site.aggregateArgs.emplace_back(i, std::move(*slots));
  }
  Graph &graph = fg->getGraph();
  for (Value result : op->getResults()) {
    // TODO(T06/T08): the callee's effects and return value.
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

void LocalBuilder::visitMemTransfer(Operation *op, Value dstValue,
                                    Value srcValue, Value lengthValue) {
  std::optional<Cell> dst = cellFor(dstValue);
  std::optional<Cell> src = cellFor(srcValue);
  if (!dst || !src)
    return; // Copying from or to null is UB.
  Graph &graph = fg->getGraph();
  graph.addFlags(*dst, NodeFlags::Mod);
  graph.addFlags(*src, NodeFlags::Ref);
  // Like cir.copy over the bytes [0, length), without a type: done by the
  // closure after the walk, once every link of the function is known. An
  // unknown length is a range without an end: the closure relates every
  // offset at or after the two cells, which over-approximates any length.
  std::optional<int64_t> length = getConstantIndex(lengthValue);
  if (length && *length <= 0)
    return;
  if (length) {
    graph.growSize(dst->getNode(), dst->getOffset() + *length);
    graph.growSize(src->getNode(), src->getOffset() + *length);
  }
  fg->memTransfers.push_back(
      {op, *dst, *src,
       length ? static_cast<uint64_t>(*length) : MemTransferRecord::kUnknown});
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

  bool apply() {
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

void cir::pointsto::closeMemTransfers(Graph &graph,
                                      ArrayRef<MemTransferRecord> transfers) {
  // Unification is symmetric, so this treats dst and src alike: it only
  // ever adds links and merges targets, which over-approximates the
  // one-directional copy. That is the DSA treatment of memcpy, with offsets.
  for (unsigned round = 0; round != kMaxTransferRounds; ++round) {
    bool changed = false;
    for (const MemTransferRecord &record : transfers)
      changed |= TransferApplier{graph, record}.apply();
    if (!changed)
      return;
  }
  for (const MemTransferRecord &record : transfers)
    TransferApplier{graph, record}.smash();
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
