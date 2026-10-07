//===- LocalBuilder.h - Intraprocedural points-to graphs --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the local (intraprocedural) phase of the CIR points-to
// analysis: FunctionGraph, the memory graph of one cir.func together with the
// map from its pointer-typed SSA values to cells, its formals, its return cell
// and its call sites; and LocalBuilder, which fills a FunctionGraph from one
// flow-insensitive, unification-based walk over the function body.
//
// Every pointer-typed SSA value maps to the Cell it points to. A null pointer
// (`cir.const #cir.ptr<null>`) points to nothing and has no cell: it is never
// put in the scalar map, cellFor returns std::nullopt for it, and every rule
// that would unify with or store such a value skips it.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_LOCALBUILDER_H
#define CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_LOCALBUILDER_H

#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToGraph.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <memory>
#include <optional>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace cir::pointsto {

/// A call recorded by the local phase. The local phase gives calls no
/// semantics: it only remembers the cells involved, for the library models
/// (T06) and the interprocedural phases (T08/T10) to act on.
struct CallSiteRecord {
  /// The cir.call or cir.try_call.
  mlir::Operation *call = nullptr;
  /// The cell of the callee pointer of an indirect call; std::nullopt for a
  /// direct call (or an indirect call through a null pointer).
  std::optional<Cell> calleePtr;
  /// One entry per argument operand (the indirect callee excluded): the cell
  /// of a pointer argument, std::nullopt for a non-pointer or null argument.
  llvm::SmallVector<std::optional<Cell>, 4> args;
  /// A fresh node for a pointer result, std::nullopt otherwise.
  std::optional<Cell> result;
};

/// The points-to graph of one function: the memory graph and the roots that
/// connect it to the function's SSA values and to its callers and callees.
class FunctionGraph {
public:
  /// \p pointerSize and \p reasons are passed on to the Graph.
  explicit FunctionGraph(cir::FuncOp func, unsigned pointerSize = 8,
                         ReasonTable *reasons = nullptr)
      : func(func), graph(pointerSize, reasons) {}

  cir::FuncOp getFunction() const { return func; }
  Graph &getGraph() { return graph; }
  const Graph &getGraph() const { return graph; }

  /// The cell \p value points to, or std::nullopt if it has none (not a
  /// pointer, a null pointer, or a value the builder never saw).
  std::optional<Cell> lookup(mlir::Value value) const {
    auto it = scalarMap.find(value);
    if (it == scalarMap.end())
      return std::nullopt;
    return it->second;
  }

  /// Print the graph, then one line per formal, the return cell and one line
  /// per call site:
  /// \code
  ///   formal #<i> -> <cell>
  ///   return -> <cell>
  ///   call <op> [callee=<cell>] args={<i>: <cell>, ...} result=<cell>
  /// \endcode
  /// Only pointer formals and arguments are listed; a missing cell prints as
  /// `null`.
  void print(llvm::raw_ostream &os) const;

  /// What each pointer-typed SSA value of the function points to.
  llvm::DenseMap<mlir::Value, Cell> scalarMap;
  /// One entry per entry-block argument: its cell, or std::nullopt for a
  /// non-pointer argument.
  llvm::SmallVector<std::optional<Cell>, 4> formals;
  /// What the function returns, if it returns a non-null pointer.
  std::optional<Cell> returnCell;
  /// The calls in the body, in walk order.
  llvm::SmallVector<CallSiteRecord, 4> callSites;

private:
  cir::FuncOp func;
  Graph graph;
};

/// Builds the local points-to graph of one cir.func with a body.
///
/// The walk is flow-insensitive: each op is visited once, in pre-order, and
/// merges unify. A value used before the op defining it is visited (a block
/// argument, a result in a block placed after its uses) first gets a fresh
/// empty node, which is unified with the real cell once the definition is
/// reached, so the visit order does not change the result.
///
/// Ops without a rule here are passed to visitUnhandled, which only counts
/// them for now; T04 makes it pessimistic.
///
/// The data layout comes from the function's nearest enclosing module rather
/// than from DataLayout::closest on an arbitrary op: in the offload container
/// the host and device modules nest in one top-level module and have
/// different layouts, and the function's own module holds the right one.
class LocalBuilder {
public:
  /// \p symbols, if given, caches symbol lookups across builders.
  explicit LocalBuilder(cir::FuncOp func,
                        mlir::SymbolTableCollection *symbols = nullptr,
                        ReasonTable *reasons = nullptr);
  ~LocalBuilder();

  /// Walk the body and return the finished graph. Call once.
  std::unique_ptr<FunctionGraph> build();

  /// The ops that hit the default rule, by op name: those with a pointer
  /// operand or result in the first map, all of them in the second.
  const llvm::StringMap<unsigned> &getUnhandledPointerOps() const {
    return unhandledPointerOps;
  }
  const llvm::StringMap<unsigned> &getUnhandledOps() const {
    return unhandledOps;
  }

  /// The data layout a function is analysed with: the one of its nearest
  /// enclosing module.
  static mlir::DataLayout getDataLayoutFor(cir::FuncOp func);

private:
  /// The cell \p value points to, creating a fresh empty node the first time
  /// a pointer value is seen. std::nullopt for non-pointers and null.
  std::optional<Cell> cellFor(mlir::Value value);
  /// Record that \p value points to \p cell, unifying with any cell it got
  /// before.
  void define(mlir::Value value, Cell cell);
  /// Unify the cells of two values (nothing if either has none).
  void unifyValues(mlir::Value a, mlir::Value b, mlir::Operation *at);
  /// The byte offsets of the pointers inside a value of type \p type.
  llvm::ArrayRef<int64_t> getPointerOffsets(mlir::Type type);

  void visit(mlir::Operation *op);
  void visitUnhandled(mlir::Operation *op);
  void visitAlloca(cir::AllocaOp op);
  void visitLoad(cir::LoadOp op);
  void visitStore(cir::StoreOp op);
  void visitAddressStep(mlir::Operation *op);
  void visitCast(cir::CastOp op);
  void visitCollapse(mlir::Operation *op, mlir::Value src);
  void visitAtomic(mlir::Operation *op, mlir::Value addr, mlir::Value stored,
                   mlir::Value loaded);
  void visitCall(cir::CIRCallOpInterface op);
  void visitGetGlobal(cir::GetGlobalOp op);
  void visitReturn(cir::ReturnOp op);
  void visitRegionTerminator(mlir::Operation *op);
  void visitBranch(mlir::Operation *op);

  cir::FuncOp func;
  mlir::DataLayout dataLayout;
  mlir::SymbolTableCollection *symbols;
  std::unique_ptr<mlir::SymbolTableCollection> ownedSymbols;
  std::unique_ptr<FunctionGraph> fg;
  /// One Global node per symbol.
  llvm::DenseMap<mlir::Attribute, Node *> globalNodes;
  /// The pointers inside aggregate SSA values: (byte offset, target).
  llvm::DenseMap<mlir::Value, llvm::SmallVector<std::pair<int64_t, Cell>, 2>>
      aggregates;
  llvm::DenseMap<mlir::Type, llvm::SmallVector<int64_t, 2>> pointerOffsets;
  llvm::StringMap<unsigned> unhandledPointerOps;
  llvm::StringMap<unsigned> unhandledOps;
};

} // namespace cir::pointsto

#endif // CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_LOCALBUILDER_H
