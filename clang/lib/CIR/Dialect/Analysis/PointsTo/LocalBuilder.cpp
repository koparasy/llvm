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
//    clear_padding, masked vector loads): Mod and/or Ref only.
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
//  - cir.get_global: one Global (or Function) node per symbol.
//  - cir.const: a null pointer has no cell, a zero/undef/poison aggregate
//    null slots; any other pointer-holding constant is pessimistic (T05).
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
                                  SmallVectorImpl<int64_t> &out) {
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
                            dataLayout, out);
    return;
  }
  if (auto arrayTy = dyn_cast<cir::ArrayType>(type)) {
    SmallVector<int64_t> elemOffsets;
    collectPointerOffsets(arrayTy.getElementType(), 0, dataLayout, elemOffsets);
    std::optional<int64_t> elemSize =
        getTypeSizeInBytes(arrayTy.getElementType(), dataLayout);
    if (elemOffsets.empty() || !elemSize)
      return;
    uint64_t count =
        std::min<uint64_t>(arrayTy.getSize(), kMaxScannedArrayElems);
    for (uint64_t k = 0; k != count; ++k)
      for (int64_t off : elemOffsets)
        out.push_back(base + k * *elemSize + off);
  }
}

SmallVector<int64_t, 2>
cir::pointsto::getPointerSlotOffsets(Type type,
                                     const mlir::DataLayout &dataLayout) {
  SmallVector<int64_t, 2> offsets;
  if (isPointerType(type))
    return offsets;
  collectPointerOffsets(type, 0, dataLayout, offsets);
  llvm::sort(offsets);
  offsets.erase(llvm::unique(offsets), offsets.end());
  return offsets;
}

SmallVector<int64_t, 2> LocalBuilder::getPointerOffsets(Type type) {
  auto [it, inserted] = pointerOffsets.try_emplace(type);
  if (inserted)
    it->second = getPointerSlotOffsets(type, dataLayout);
  return it->second;
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
          visitModRef(op.getDst(), {}, /*mod=*/true, /*ref=*/false);
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

  if (isPointerType(type)) {
    define(op.getResult(), graph.getLink(*addr));
    return;
  }
  SmallVector<int64_t, 2> offsets = getPointerOffsets(type);
  if (offsets.empty())
    return;
  AggregateSlots slots;
  for (int64_t off : offsets)
    slots.emplace_back(off, graph.getLink(shift(*addr, off)));
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

void LocalBuilder::visitConstant(cir::ConstantOp op) {
  Type type = op.getType();
  if (!hasPointers(type) || isNullPointer(op.getResult()))
    return; // A null pointer constant has no cell (see cellFor).
  if (isPointerType(type)) {
    // TODO(T05): #cir.global_view and other non-null pointer constants.
    visitUnhandled(op, "non-null pointer constant");
    return;
  }
  if (isa<cir::ZeroAttr, cir::UndefAttr, cir::PoisonAttr>(op.getValue())) {
    AggregateSlots slots;
    for (int64_t off : getPointerOffsets(type))
      slots.emplace_back(off, Cell());
    defineSlots(op.getResult(), slots);
    return;
  }
  // TODO(T05): walk the initializer (null, global_view, function pointers).
  visitUnhandled(op, "aggregate constant holding pointers");
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
  Graph &graph = fg->getGraph();
  FlatSymbolRefAttr sym = op.getNameAttr();
  Node *&node = globalNodes[sym];
  if (!node) {
    // TODO(T05): the initializer.
    Operation *target = symbols->lookupNearestSymbolFrom(op, sym);
    if (isa_and_present<cir::FuncOp>(target)) {
      node = graph.makeNode(Origin{Origin::Kind::FuncSym, nullptr, sym},
                            NodeFlags::Function);
    } else {
      node = graph.makeNode(Origin{Origin::Kind::GlobalSym, nullptr, sym},
                            NodeFlags::Global);
      if (auto globalOp = dyn_cast_if_present<cir::GlobalOp>(target))
        if (std::optional<int64_t> size =
                getTypeSizeInBytes(globalOp.getSymType(), dataLayout))
          graph.growSize(node, *size);
    }
  }
  define(op.getAddr(), Cell(node));
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
