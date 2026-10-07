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
// An aggregate SSA value (a record or array) that holds pointers maps to its
// AggregateSlots instead: one entry per pointer inside it, giving the cell
// that pointer points to. Every aggregate value whose type contains pointers
// has such a record, whatever produced it.
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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <memory>
#include <optional>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace cir::pointsto {

/// Whether values of \p type are pointers to the analysis (!cir.ptr and
/// !cir.vptr).
bool isPointerValueType(mlir::Type type);

/// Whether \p value is a null pointer constant (`#cir.ptr<null>` or
/// `#cir.zero` of pointer type), which has no cell.
bool isNullPointerConstant(mlir::Value value);

/// Whether \p value is a null pointer constant (`#cir.ptr<null>` or
/// `#cir.zero`), which points to nothing and has no cell.
bool isNullPointerConstant(mlir::Value value);

/// The byte offsets of the pointers inside a value of \p type, sorted and
/// unique; empty for a pointer or a type without pointers. Arrays are scanned
/// up to their first 256 elements.
llvm::SmallVector<int64_t, 2>
getPointerSlotOffsets(mlir::Type type, const mlir::DataLayout &dataLayout);

/// The pointers inside an aggregate SSA value: (byte offset, target cell),
/// one entry per offset getPointerSlotOffsets returns for the value's type, in
/// the same order. An empty Cell is a null pointer.
using AggregateSlots = llvm::SmallVector<std::pair<int64_t, Cell>, 2>;

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
  /// The slots of each aggregate argument that holds pointers: (argument
  /// index, slots).
  llvm::SmallVector<std::pair<unsigned, AggregateSlots>, 0> aggregateArgs;
  /// Fresh slot nodes for an aggregate result that holds pointers.
  std::optional<AggregateSlots> resultSlots;
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

  /// The slots of the aggregate \p value, or nullptr if it has none (its type
  /// holds no pointers, or the builder never saw it).
  const AggregateSlots *lookupSlots(mlir::Value value) const {
    auto it = aggregateMap.find(value);
    return it == aggregateMap.end() ? nullptr : &it->second;
  }

  /// Print the graph, then one line per formal, the return cell and one line
  /// per call site:
  /// \code
  ///   formal #<i> -> <cell>
  ///   return -> <cell>
  ///   call <op> [callee=<cell>] args={<i>: <cell>, ...} result=<cell>
  ///       [slots={<i>: <slots>, ...}] [result.slots=<slots>]
  /// \endcode
  /// Only pointer formals and arguments are listed; a missing cell prints as
  /// `null`. Aggregate formals and returns print `formal #<i> -> <slots>` and
  /// `return -> <slots>`, where <slots> is `{<offset>: <cell>, ...}`.
  void print(llvm::raw_ostream &os) const;

  /// What each pointer-typed SSA value of the function points to.
  llvm::DenseMap<mlir::Value, Cell> scalarMap;
  /// The pointer slots of each aggregate SSA value that holds pointers.
  llvm::DenseMap<mlir::Value, AggregateSlots> aggregateMap;
  /// One entry per entry-block argument: its cell, or std::nullopt for a
  /// non-pointer argument.
  llvm::SmallVector<std::optional<Cell>, 4> formals;
  /// What the function returns, if it returns a non-null pointer.
  std::optional<Cell> returnCell;
  /// The slots of what the function returns, if it returns an aggregate that
  /// holds pointers.
  std::optional<AggregateSlots> returnSlots;
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
/// What happens to an op depends on its OpClass (OpClassification.h).
/// Handled ops get their rule; NoPointerSemantics ops are skipped; Pessimistic
/// ops, and ops missing from the table, go to visitUnhandled: the nodes of
/// their pointer operands (and of the pointers inside aggregate operands)
/// become Collapsed and External, and their pointer results (and the pointer
/// slots of aggregate results) point to fresh UnknownObj nodes, External and
/// Incomplete. Each such flag carries an UnhandledOp reason naming the op. A
/// pointer or aggregate result that the op's rule left without a cell gets
/// the same pessimistic treatment, so every value has a cell or slots.
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

  /// Walk the body and return the finished graph. Call once. Under
  /// -cir-points-to-verify the graph is verified before it is returned.
  std::unique_ptr<FunctionGraph> build();

  /// Whether build() verified the graph and found violations (reported as
  /// errors on the ops).
  bool hadVerifyErrors() const { return verifyErrors; }

  /// The ops that went to the pessimistic default, by op name: those that
  /// touched a pointer (operand, result or a slot of either) or have memory
  /// effects in the first map, all of them in the second, and those missing
  /// from the classification table in the third.
  const llvm::StringMap<unsigned> &getPessimisticOps() const {
    return pessimisticOps;
  }
  const llvm::StringMap<unsigned> &getUnhandledOps() const {
    return unhandledOps;
  }
  const llvm::StringMap<unsigned> &getUnclassifiedOps() const {
    return unclassifiedOps;
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
  /// Unify the cells of two values (nothing if either has none), or the
  /// slots of two aggregate values slot by slot.
  void unifyValues(mlir::Value a, mlir::Value b, mlir::Operation *at);
  /// The byte offsets of the pointers inside a value of type \p type
  /// (cached; returned by value, as the cache may grow while it is used).
  llvm::SmallVector<int64_t, 2> getPointerOffsets(mlir::Type type);
  /// Whether \p type is a pointer or an aggregate holding pointers.
  bool hasPointers(mlir::Type type);
  /// The slots of the aggregate \p value, creating a fresh empty node per
  /// slot the first time it is seen. std::nullopt if its type holds no
  /// pointers.
  std::optional<AggregateSlots> slotsFor(mlir::Value value);
  /// Record the slots of \p value, unifying with any it got before.
  void defineSlots(mlir::Value value, const AggregateSlots &slots);
  /// Give \p value, a pointer or an aggregate holding pointers, fresh
  /// UnknownObj targets, External and Incomplete, for reason \p why.
  void defineUnknown(mlir::Value value, const Reason *why);
  /// Define \p result as a copy of \p src (same cell or same slots).
  void defineCopy(mlir::Value result, mlir::Value src);

  void visit(mlir::Operation *op);
  void visitUnhandled(mlir::Operation *op, llvm::StringRef detail = {});
  /// Give the results the op's rule did not define the pessimistic default.
  void defineMissingResults(mlir::Operation *op);
  /// Fill block arguments that no visited edge defined.
  void defineBlockArguments();
  void visitModRef(mlir::Value addr, mlir::Type accessType, bool mod, bool ref);
  void visitConstant(cir::ConstantOp op);
  void visitExtractMember(cir::ExtractMemberOp op);
  void visitInsertMember(cir::InsertMemberOp op);
  void visitRegionEntry(mlir::Operation *op);
  /// Make sure \p value has a cell or slots and count it as defined.
  void definePlaceholder(mlir::Value value);
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
  llvm::DenseMap<mlir::Type, llvm::SmallVector<int64_t, 2>> pointerOffsets;
  /// The values a rule defined (as opposed to placeholders created by a use
  /// seen before the definition).
  llvm::DenseSet<mlir::Value> defined;
  llvm::StringMap<unsigned> pessimisticOps;
  llvm::StringMap<unsigned> unhandledOps;
  llvm::StringMap<unsigned> unclassifiedOps;
  bool verifyErrors = false;
};

} // namespace cir::pointsto

#endif // CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_LOCALBUILDER_H
