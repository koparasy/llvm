//===- GlobalNodes.cpp - Points-to nodes of global symbols ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/PointsTo/GlobalNodes.h"
#include "clang/CIR/Dialect/Analysis/CIRPointerDecomposition.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace cir;
using namespace cir::pointsto;

/// Aliases of aliases are followed this deep; deeper (or cyclic) chains give
/// an unknown node.
static constexpr unsigned kMaxAliasDepth = 8;

//===----------------------------------------------------------------------===//
// Constant attributes
//===----------------------------------------------------------------------===//

/// \p a + \p b * \p c, or std::nullopt on overflow.
static std::optional<int64_t> addMul(int64_t a, int64_t b, int64_t c) {
  int64_t product, sum;
  if (llvm::MulOverflow(b, c, product) || llvm::AddOverflow(a, product, sum))
    return std::nullopt;
  return sum;
}

std::optional<int64_t>
cir::pointsto::getGlobalViewOffset(cir::GlobalViewAttr view, Type symbolType,
                                   const mlir::DataLayout &dataLayout) {
  ArrayAttr indices = view.getIndices();
  if (!indices || indices.empty())
    return 0;
  if (!symbolType)
    return std::nullopt;

  // The lowering emits a GEP over the symbol's type, with a leading zero
  // index when that type is an aggregate: the indices step into it. On any
  // other type the first index steps over whole objects.
  int64_t offset = 0;
  Type type = symbolType;
  auto it = indices.begin();
  if (!isa<cir::ArrayType, cir::RecordType>(type)) {
    auto index = dyn_cast<IntegerAttr>(*it++);
    std::optional<int64_t> size = getTypeSizeInBytes(type, dataLayout);
    if (!index || !size)
      return std::nullopt;
    std::optional<int64_t> next =
        addMul(offset, index.getValue().getSExtValue(), *size);
    if (!next)
      return std::nullopt;
    offset = *next;
  }
  for (; it != indices.end(); ++it) {
    auto index = dyn_cast<IntegerAttr>(*it);
    if (!index)
      return std::nullopt;
    int64_t i = index.getValue().getSExtValue();
    if (auto arrayTy = dyn_cast<cir::ArrayType>(type)) {
      std::optional<int64_t> elemSize =
          getTypeSizeInBytes(arrayTy.getElementType(), dataLayout);
      std::optional<int64_t> next =
          elemSize ? addMul(offset, i, *elemSize) : std::nullopt;
      if (!next)
        return std::nullopt;
      offset = *next;
      type = arrayTy.getElementType();
      continue;
    }
    auto recordTy = dyn_cast<cir::RecordType>(type);
    if (!recordTy || recordTy.isIncomplete() || i < 0 ||
        static_cast<uint64_t>(i) >= recordTy.getNumElements())
      return std::nullopt;
    std::optional<int64_t> next =
        addMul(offset, 1,
               static_cast<int64_t>(recordTy.getElementOffset(dataLayout, i)));
    if (!next)
      return std::nullopt;
    offset = *next;
    type = recordTy.getElementType(i);
  }
  return offset;
}

namespace {
/// The recursive walk behind walkConstantPointers.
struct ConstantWalker {
  const mlir::DataLayout &dataLayout;
  SymbolTypeFn symbolType;
  ConstantPointerFn fn;

  void unknown(int64_t base) {
    ConstantPointer pointer;
    pointer.kind = ConstantPointer::Kind::Unknown;
    fn(base, pointer);
  }

  void symbol(FlatSymbolRefAttr sym, std::optional<int64_t> offset,
              Type valueType, int64_t base) {
    ConstantPointer pointer;
    pointer.kind = ConstantPointer::Kind::Symbol;
    pointer.symbol = sym;
    pointer.offset = offset;
    pointer.asInteger = isa<cir::IntType>(valueType);
    fn(base, pointer);
  }

  /// Walk the members of a record-like constant of type \p type.
  void members(Type type, ArrayAttr elements, int64_t base) {
    auto recordTy = dyn_cast<cir::RecordType>(type);
    if (!recordTy || recordTy.isIncomplete()) {
      unknown(base);
      return;
    }
    for (auto [i, member] : llvm::enumerate(elements)) {
      // Members past the record's elements initialize a flexible array
      // member; their layout is not in the type.
      if (i >= recordTy.getNumElements()) {
        unknown(base);
        return;
      }
      walk(member, base + static_cast<int64_t>(
                              recordTy.getElementOffset(dataLayout, i)));
    }
  }

  void walk(Attribute value, int64_t base) {
    if (auto view = dyn_cast<cir::GlobalViewAttr>(value)) {
      FlatSymbolRefAttr sym = view.getSymbol();
      symbol(sym, getGlobalViewOffset(view, symbolType(sym), dataLayout),
             view.getType(), base);
      return;
    }
    if (auto offset = dyn_cast<cir::GlobalOffsetAttr>(value)) {
      symbol(offset.getSymbol(), offset.getOffset(), offset.getType(), base);
      return;
    }
    if (auto ptr = dyn_cast<cir::ConstPtrAttr>(value)) {
      if (ptr.isNullValue())
        return;
      ConstantPointer pointer;
      pointer.kind = ConstantPointer::Kind::Absolute;
      fn(base, pointer);
      return;
    }
    // Null, undefined, or values that cannot hold a pointer.
    if (isa<cir::ZeroAttr, cir::UndefAttr, cir::PoisonAttr, cir::IntAttr,
            cir::FPAttr, cir::BoolAttr, cir::ConstComplexAttr,
            cir::ConstVectorAttr, cir::DataMemberAttr, cir::MethodAttr>(value))
      return;
    if (auto record = dyn_cast<cir::ConstRecordAttr>(value)) {
      members(record.getType(), record.getMembers(), base);
      return;
    }
    if (auto vtable = dyn_cast<cir::VTableAttr>(value)) {
      members(vtable.getType(), vtable.getData(), base);
      return;
    }
    if (auto typeInfo = dyn_cast<cir::TypeInfoAttr>(value)) {
      members(typeInfo.getType(), typeInfo.getData(), base);
      return;
    }
    if (auto array = dyn_cast<cir::ConstArrayAttr>(value)) {
      auto elements = dyn_cast<ArrayAttr>(array.getElts());
      if (!elements)
        return; // A string.
      auto arrayTy = dyn_cast<cir::ArrayType>(array.getType());
      std::optional<int64_t> elemSize =
          arrayTy ? getTypeSizeInBytes(arrayTy.getElementType(), dataLayout)
                  : std::nullopt;
      if (!elemSize) {
        unknown(base);
        return;
      }
      // The trailing zeros are null.
      for (auto [i, element] : llvm::enumerate(elements))
        walk(element, base + static_cast<int64_t>(i) * *elemSize);
      return;
    }
    unknown(base);
  }
};
} // namespace

void cir::pointsto::walkConstantPointers(Attribute value, int64_t base,
                                         const mlir::DataLayout &dataLayout,
                                         SymbolTypeFn symbolType,
                                         ConstantPointerFn fn) {
  ConstantWalker{dataLayout, symbolType, fn}.walk(value, base);
}

//===----------------------------------------------------------------------===//
// GlobalNodes
//===----------------------------------------------------------------------===//

GlobalNodes::GlobalNodes(Graph &graph, const mlir::DataLayout &dataLayout,
                         SymbolTableCollection &symbols, Operation *scope)
    : graph(graph), dataLayout(dataLayout), symbols(symbols), scope(scope) {}

Operation *GlobalNodes::lookup(FlatSymbolRefAttr sym) {
  return symbols.lookupNearestSymbolFrom(scope, sym);
}

Type GlobalNodes::getSymbolType(FlatSymbolRefAttr sym) {
  if (auto global = dyn_cast_if_present<cir::GlobalOp>(lookup(sym)))
    return global.getSymType();
  return {};
}

Cell GlobalNodes::makeUnknown(const Reason *why) {
  Cell cell(graph.makeNode(NodeFlags::UnknownObj));
  graph.addFlags(cell, NodeFlags::External | NodeFlags::Incomplete, why);
  return cell;
}

SmallVector<std::pair<FlatSymbolRefAttr, Node *>, 0>
GlobalNodes::getSymbolNodes() const {
  SmallVector<std::pair<FlatSymbolRefAttr, Node *>, 0> result;
  for (const auto &[sym, node] : nodes)
    result.emplace_back(cast<FlatSymbolRefAttr>(sym), node);
  llvm::sort(result, [](const auto &a, const auto &b) {
    return a.first.getValue() < b.first.getValue();
  });
  return result;
}

Node *GlobalNodes::getNode(FlatSymbolRefAttr sym) {
  Node *node = makeNode(sym, /*depth=*/0);
  // Read the initializers of the node made and of every global they reach.
  // A worklist rather than recursion: initializers form arbitrary graphs.
  if (!draining) {
    draining = true;
    while (!pending.empty()) {
      auto [pendingNode, global] = pending.pop_back_val();
      initialize(pendingNode, global);
    }
    draining = false;
  }
  return node;
}

Node *GlobalNodes::makeNode(FlatSymbolRefAttr sym, unsigned depth) {
  if (Node *node = nodes.lookup(sym))
    return node;
  Operation *target = lookup(sym);

  // An alias denotes the object of its aliasee.
  std::optional<StringRef> aliasee;
  if (auto global = dyn_cast_if_present<cir::GlobalOp>(target))
    aliasee = global.getAliasee();
  else if (auto func = dyn_cast_if_present<cir::FuncOp>(target))
    aliasee = func.getAliasee();
  if (aliasee && depth < kMaxAliasDepth) {
    Node *node =
        makeNode(FlatSymbolRefAttr::get(sym.getContext(), *aliasee), depth + 1);
    nodes[sym] = node;
    return node;
  }

  Node *node;
  if (isa_and_present<cir::FuncOp>(target) && !aliasee) {
    node = graph.makeNode(Origin{Origin::Kind::FuncSym, nullptr, sym},
                          NodeFlags::Function);
  } else {
    node = graph.makeNode(Origin{Origin::Kind::GlobalSym, nullptr, sym},
                          NodeFlags::Global);
    auto global = dyn_cast_if_present<cir::GlobalOp>(target);
    if (global && !aliasee) {
      if (std::optional<int64_t> size =
              getTypeSizeInBytes(global.getSymType(), dataLayout))
        graph.growSize(node, *size);
      pending.emplace_back(node, global);
    } else {
      graph.addFlags(Cell(node), NodeFlags::External | NodeFlags::Incomplete,
                     graph.makeReason(Reason::Kind::EscapedTo, target,
                                      aliasee ? "alias chain too deep"
                                              : "symbol not found"));
    }
  }
  nodes[sym] = node;
  return node;
}

void GlobalNodes::initialize(Node *node, cir::GlobalOp global) {
  cir::GlobalLinkageKind linkage = global.getLinkage();
  std::optional<Attribute> init = global.getInitialValue();

  // SOUNDNESS: what another TU may store into the global is not in this
  // module. Only an internal/private definition, or a constant whose
  // initializer cannot be replaced at link time, has known contents.
  bool trustedConstant =
      global.getConstant() && init && !cir::isInterposableLinkage(linkage);
  if (global.isDeclaration() ||
      (!cir::isLocalLinkage(linkage) && !trustedConstant))
    graph.addFlags(Cell(node), NodeFlags::External | NodeFlags::Incomplete,
                   graph.makeReason(Reason::Kind::EscapedTo, global,
                                    "global visible to other TUs"));

  // SOUNDNESS: the ctor and dtor regions run code no function graph walks.
  if (!global.getCtorRegion().empty() || !global.getDtorRegion().empty())
    graph.addFlags(Cell(node), NodeFlags::Incomplete,
                   graph.makeReason(Reason::Kind::UnknownCaller, global,
                                    "dynamic initialization not analysed"));

  if (!init)
    return;
  const Reason *why = nullptr;
  auto reason = [&](Reason::Kind kind, StringRef detail) {
    return graph.makeReason(kind, global, detail);
  };
  walkConstantPointers(
      *init, 0, dataLayout,
      [&](FlatSymbolRefAttr sym) { return getSymbolType(sym); },
      [&](int64_t offset, const ConstantPointer &pointer) {
        switch (pointer.kind) {
        case ConstantPointer::Kind::Symbol: {
          Cell target = getCell(pointer, global);
          if (pointer.asInteger) {
            graph.addFlags(target, NodeFlags::External,
                           reason(Reason::Kind::EscapedTo,
                                  "address escaped as an integer"));
            return;
          }
          if (!why)
            why = reason(Reason::Kind::CollapsedAt, "initializer");
          graph.setLink(Cell(node, offset), target, why);
          return;
        }
        case ConstantPointer::Kind::Absolute:
          graph.setLink(
              Cell(node, offset),
              makeUnknown(reason(Reason::Kind::EscapedTo, "absolute address")));
          return;
        case ConstantPointer::Kind::Unknown: {
          const Reason *unknown =
              reason(Reason::Kind::UnhandledOp, "initializer not understood");
          graph.collapse(node, unknown);
          graph.setLink(Cell(node), makeUnknown(unknown));
          return;
        }
        }
      });
}

Cell GlobalNodes::getCell(const ConstantPointer &pointer, Operation *at) {
  assert(pointer.kind == ConstantPointer::Kind::Symbol);
  Node *node = getNode(pointer.symbol);
  if (pointer.offset)
    return Cell(node, *pointer.offset);
  graph.collapse(node, graph.makeReason(Reason::Kind::CollapsedAt, at,
                                        "global_view of unknown offset"));
  return Cell(node);
}
