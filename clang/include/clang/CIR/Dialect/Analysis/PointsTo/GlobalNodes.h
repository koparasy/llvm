//===- GlobalNodes.h - Points-to nodes of global symbols --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares how the CIR points-to analysis models global symbols and
// the pointers inside constant attributes:
//
//  - walkConstantPointers enumerates the pointers a constant attribute holds
//    (#cir.global_view, #cir.global_offset, #cir.ptr, and the aggregates
//    #cir.const_record, #cir.const_array, #cir.vtable and #cir.typeinfo that
//    contain them), with their byte offsets;
//  - GlobalNodes gives each global symbol of one Graph its node (a Global
//    node per cir.global, a Function node per cir.func) and, the first time
//    the node is made, links the pointer slots of the global's initializer to
//    the nodes of the symbols they name.
//
// GlobalNodes only depends on a Graph, a data layout and a symbol table, so
// the same code fills the nodes of a function graph today and the shared
// globals graph of the interprocedural phase later.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_GLOBALNODES_H
#define CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_GLOBALNODES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToGraph.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace cir::pointsto {

/// A pointer held by a constant attribute.
struct ConstantPointer {
  enum class Kind : uint8_t {
    /// The address of a symbol plus a byte offset (#cir.global_view,
    /// #cir.global_offset).
    Symbol,
    /// A non-null integer address (#cir.ptr<N>, N != 0): it may point to any
    /// object, including ones we never see.
    Absolute,
    /// An attribute the walker does not understand, in a position that may
    /// hold pointers. Where they point and at which offsets is unknown.
    Unknown,
  };

  Kind kind = Kind::Unknown;
  /// The symbol, for Kind::Symbol.
  mlir::FlatSymbolRefAttr symbol;
  /// The byte offset from the start of the symbol, for Kind::Symbol;
  /// std::nullopt when the view's indices could not be evaluated.
  std::optional<int64_t> offset;
  /// Set for a symbol address that the constant holds as an integer (an
  /// integer-typed #cir.global_view): the address escapes, but no pointer
  /// slot holds it.
  bool asInteger = false;
};

/// Called with the byte offset of each pointer inside a constant and the
/// pointer.
using ConstantPointerFn =
    llvm::function_ref<void(int64_t offset, const ConstantPointer &pointer)>;

/// Returns the type of the storage of a symbol: the type of a cir.global, a
/// null type for a function or a missing symbol.
using SymbolTypeFn = llvm::function_ref<mlir::Type(mlir::FlatSymbolRefAttr)>;

/// The byte offset of the address \p view denotes from the start of its
/// symbol, whose storage has type \p symbolType. Follows the lowering: on an
/// array or record the indices step into sub-elements, on any other type the
/// first index steps over whole objects of that type. std::nullopt if an index
/// can't be evaluated (an incomplete or unsized type, an index out of range,
/// overflow); 0 if there are no indices.
std::optional<int64_t> getGlobalViewOffset(cir::GlobalViewAttr view,
                                           mlir::Type symbolType,
                                           const mlir::DataLayout &dataLayout);

/// Call \p fn for every non-null pointer inside the constant \p value, which
/// is placed at byte offset \p base, and for every symbol address it holds as
/// an integer. Null pointers (#cir.ptr<null>, #cir.zero, undef, poison,
/// trailing zeros) are not reported, nor are values that cannot hold pointers
/// (integers, floats, vectors, complex numbers, strings, member pointers).
/// A sub-attribute the walker does not understand is reported once as
/// ConstantPointer::Kind::Unknown at its offset.
void walkConstantPointers(mlir::Attribute value, int64_t base,
                          const mlir::DataLayout &dataLayout,
                          SymbolTypeFn symbolType, ConstantPointerFn fn);

/// The nodes of the global symbols of one Graph: one Global node per
/// cir.global (an alias shares the node of its aliasee) and one Function node
/// per cir.func.
///
/// The first time a global's node is made, its initializer is read: every
/// pointer slot it holds is linked to the node of the symbol it names (made,
/// and initialized, in turn) at the right offset. The node is also flagged:
///
///  - External|Incomplete, Reason{EscapedTo, "global visible to other TUs"},
///    for a declaration and for a definition that code in another TU can
///    write (any linkage but internal/private). Exception: a `constant`
///    global with an initial value is trusted whatever its linkage, unless
///    the linkage is interposable (weak, linkonce_any, common, extern_weak),
///    since then another definition may replace the initializer. Writing a
///    constant is UB. SOUNDNESS DEPENDS ON THIS CHECK.
///  - Incomplete, Reason{UnknownCaller, "dynamic initialization not
///    analysed"}, for a global with a ctor or dtor region. The regions run code
///    (constructors, destructors) that is not part of any function graph; the
///    shared globals graph of the interprocedural phase is the place to walk
///    them. A static local's dynamic initialization is a cir.local_init in its
///    function, which the function's graph walks.
///  - External|Incomplete, Reason{EscapedTo, "symbol not found"}, for a
///    symbol that names neither a cir.global nor a cir.func.
class GlobalNodes {
public:
  /// Symbols are looked up from \p scope (the nearest symbol table that
  /// encloses it). \p dataLayout and \p symbols must outlive this object.
  GlobalNodes(Graph &graph, const mlir::DataLayout &dataLayout,
              mlir::SymbolTableCollection &symbols, mlir::Operation *scope);

  /// The node of the global or function \p sym, made and initialized the
  /// first time it is asked for.
  Node *getNode(mlir::FlatSymbolRefAttr sym);

  /// The cell a Kind::Symbol \p pointer denotes. An unknown offset collapses
  /// the symbol's node (\p at is the op to blame).
  Cell getCell(const ConstantPointer &pointer, mlir::Operation *at);

  /// The type of the storage of \p sym: the cir.global's type, or a null type
  /// for a function or a missing symbol.
  mlir::Type getSymbolType(mlir::FlatSymbolRefAttr sym);

  /// A fresh UnknownObj node, External and Incomplete for \p why: what an
  /// absolute address or an uninterpretable constant points to.
  Cell makeUnknown(const Reason *why);

private:
  mlir::Operation *lookup(mlir::FlatSymbolRefAttr sym);
  Node *makeNode(mlir::FlatSymbolRefAttr sym, unsigned depth);
  void initialize(Node *node, cir::GlobalOp global);

  Graph &graph;
  const mlir::DataLayout &dataLayout;
  mlir::SymbolTableCollection &symbols;
  mlir::Operation *scope;
  llvm::DenseMap<mlir::Attribute, Node *> nodes;
  /// Globals whose node is made but whose initializer is not read yet.
  llvm::SmallVector<std::pair<Node *, cir::GlobalOp>, 4> pending;
  bool draining = false;
};

} // namespace cir::pointsto

#endif // CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_GLOBALNODES_H
