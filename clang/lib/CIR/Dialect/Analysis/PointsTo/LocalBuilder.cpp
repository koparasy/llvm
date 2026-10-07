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
//    target of the slot loaded from. An aggregate result remembers the
//    targets of the pointer slots inside it, for a later store.
//  - cir.store: Mod on the address node. A pointer value is linked from the
//    slot; an aggregate value links each pointer slot it carries.
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
//    block arguments (through BranchOpInterface) and cir.return (into the
//    single return cell): unification.
//  - Atomics: a load of the old value plus a store of the new one.
//  - Calls: recorded only. A pointer result gets a fresh node.
//  - cir.get_global: one Global (or Function) node per symbol.
//  - Entry block arguments: a fresh Incomplete node each.
//
// Anything else goes to visitUnhandled.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/PointsTo/LocalBuilder.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "clang/CIR/Dialect/Analysis/CIRPointerDecomposition.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace cir;
using namespace cir::pointsto;

/// Element counts beyond this are not scanned for pointer slots in an
/// aggregate SSA value.
static constexpr uint64_t kMaxScannedArrayElems = 256;

static bool isPointerType(Type type) {
  return isa<cir::PointerType, cir::VPtrType>(type);
}

static bool isNullPointer(Value value) {
  auto constOp = dyn_cast_if_present<cir::ConstantOp>(value.getDefiningOp());
  if (!constOp)
    return false;
  auto ptrAttr = dyn_cast<cir::ConstPtrAttr>(constOp.getValue());
  return ptrAttr && ptrAttr.isNullValue();
}

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

void FunctionGraph::print(raw_ostream &os) const {
  graph.print(os);
  for (auto [i, formal] : llvm::enumerate(formals)) {
    if (!formal)
      continue;
    os << "formal #" << i << " -> ";
    Graph::printCell(os, *formal);
    os << "\n";
  }
  if (returnCell) {
    os << "return -> ";
    Graph::printCell(os, *returnCell);
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
  auto [it, inserted] = fg->scalarMap.try_emplace(value, cell);
  if (!inserted)
    fg->getGraph().unify(it->second, cell);
}

void LocalBuilder::unifyValues(Value a, Value b, Operation *at) {
  std::optional<Cell> ca = cellFor(a);
  std::optional<Cell> cb = cellFor(b);
  if (!ca || !cb)
    return;
  Graph &graph = fg->getGraph();
  graph.unify(*ca, *cb, graph.makeReason(Reason::Kind::MergedAt, at));
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

ArrayRef<int64_t> LocalBuilder::getPointerOffsets(Type type) {
  auto [it, inserted] = pointerOffsets.try_emplace(type);
  if (inserted) {
    collectPointerOffsets(type, 0, dataLayout, it->second);
    llvm::sort(it->second);
    it->second.erase(llvm::unique(it->second), it->second.end());
  }
  return it->second;
}

std::unique_ptr<FunctionGraph> LocalBuilder::build() {
  assert(fg && "build() called twice");
  Graph &graph = fg->getGraph();

  // Formals: what a caller passes in is not known yet.
  Block &entry = func.getBody().front();
  for (BlockArgument arg : entry.getArguments()) {
    if (!isPointerType(arg.getType())) {
      fg->formals.push_back(std::nullopt);
      continue;
    }
    Cell cell(graph.makeNode());
    graph.addFlags(
        cell, NodeFlags::Incomplete,
        graph.makeReason(Reason::Kind::UnknownCaller, func,
                         "formal #" + std::to_string(arg.getArgNumber())));
    define(arg, cell);
    fg->formals.push_back(cell);
  }

  func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op != func.getOperation())
      visit(op);
  });
  return std::move(fg);
}

void LocalBuilder::visit(Operation *op) {
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
  if (isa<RegionBranchOpInterface>(op))
    isControlFlow = true;

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
      .Case<cir::GetRuntimeMemberOp>(
          [&](cir::GetRuntimeMemberOp op) { visitCollapse(op, op.getAddr()); })
      .Case<cir::DynamicCastOp>(
          [&](cir::DynamicCastOp op) { visitCollapse(op, op.getSrc()); })
      .Case<cir::SelectOp>([&](cir::SelectOp op) {
        unifyValues(op.getResult(), op.getTrueValue(), op);
        unifyValues(op.getResult(), op.getFalseValue(), op);
      })
      .Case<cir::AtomicFetchOp>([&](cir::AtomicFetchOp op) {
        visitAtomic(op, op.getPtr(), op.getVal(), op.getResult());
      })
      .Case<cir::AtomicXchgOp>([&](cir::AtomicXchgOp op) {
        visitAtomic(op, op.getPtr(), op.getVal(), op.getResult());
      })
      .Case<cir::AtomicCmpXchgOp>([&](cir::AtomicCmpXchgOp op) {
        visitAtomic(op, op.getPtr(), op.getDesired(), op.getOld());
      })
      .Case<CIRCallOpInterface>([&](auto op) { visitCall(op); })
      .Case<cir::GetGlobalOp>([&](auto op) { visitGetGlobal(op); })
      .Case<cir::ReturnOp>([&](auto op) { visitReturn(op); })
      // A null pointer constant has no cell (see cellFor).
      .Case<cir::ConstantOp>([&](cir::ConstantOp op) {
        if (!isNullPointer(op.getResult()))
          visitUnhandled(op);
      })
      .Default([&](Operation *op) {
        if (!isControlFlow)
          visitUnhandled(op);
      });
}

void LocalBuilder::visitUnhandled(Operation *op) {
  // TODO(T04): be pessimistic about what the op does with its pointers.
  StringRef name = op->getName().getStringRef();
  ++unhandledOps[name];
  auto isPtr = [](Value v) { return isPointerType(v.getType()); };
  if (llvm::any_of(op->getOperands(), isPtr) ||
      llvm::any_of(op->getResults(), isPtr))
    ++unhandledPointerOps[name];
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
  ArrayRef<int64_t> offsets = getPointerOffsets(type);
  if (offsets.empty())
    return;
  auto &slots = aggregates[op.getResult()];
  for (int64_t off : offsets)
    slots.emplace_back(off, graph.getLink(shift(*addr, off)));
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
  auto it = aggregates.find(value);
  if (it == aggregates.end())
    return;
  for (auto [off, target] : it->second)
    graph.setLink(shift(*addr, off), target, reason());
}

void LocalBuilder::visitAddressStep(Operation *op) {
  Value result = op->getResult(0);
  PointerDecomposition step = decomposePointer(result, dataLayout,
                                               /*maxDepth=*/1);
  assert(step.base != result && "expected a one-step address computation");
  std::optional<Cell> base = cellFor(step.base);
  if (!base)
    return; // An address computed from null.
  Graph &graph = fg->getGraph();
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
      if (std::optional<Cell> src = cellFor(op.getSrc()))
        define(op.getResult(), *src);
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
  if (isPointerType(op.getSrc().getType()) || isPointerType(op.getType()))
    visitUnhandled(op);
}

void LocalBuilder::visitCollapse(Operation *op, Value src) {
  std::optional<Cell> base = cellFor(src);
  if (!base)
    return;
  Graph &graph = fg->getGraph();
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
  for (unsigned i = 0, e = op.getNumArgOperands(); i != e; ++i)
    site.args.push_back(cellFor(op.getArgOperand(i)));
  for (Value result : op->getResults()) {
    if (!isPointerType(result.getType()))
      continue;
    // TODO(T06/T08): the callee's effects and return value.
    Cell cell(fg->getGraph().makeNode());
    define(result, cell);
    site.result = cell;
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
  for (Value value : op.getOperands()) {
    std::optional<Cell> cell = cellFor(value);
    if (!cell)
      continue;
    if (!fg->returnCell) {
      fg->returnCell = *cell;
      continue;
    }
    Graph &graph = fg->getGraph();
    graph.unify(*fg->returnCell, *cell,
                graph.makeReason(Reason::Kind::MergedAt, op));
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
      if (isPointerType(input.getType()))
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
