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
#include "clang/CIR/Dialect/Analysis/PointsTo/GlobalNodes.h"
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

class FunctionGraph;
class PointsToModelRegistry;

/// Whether values of \p type are pointers to the analysis (!cir.ptr and
/// !cir.vptr).
bool isPointerValueType(mlir::Type type);

/// Whether \p value is a null pointer constant (`#cir.ptr<null>` or
/// `#cir.zero` of pointer type), which has no cell.
bool isNullPointerConstant(mlir::Value value);

/// What getPointerSlotOffsets left out of a type's pointer offsets.
struct SlotTruncation {
  /// Set when an array holding pointers was too long to scan: the pointer
  /// slots past its first 256 elements are missing from the offsets.
  bool truncated = false;
  /// When every truncated array starts at offset 0 of the type and they all
  /// have the same element size, that size: memory accessed through the type
  /// must then be treated as an array of it, which folds the missing slots
  /// onto the scanned ones. std::nullopt otherwise: the memory must be
  /// collapsed.
  std::optional<uint64_t> arrayElemSize;
};

/// The byte offsets of the pointers inside a value of \p type, sorted and
/// unique; empty for a pointer or a type without pointers. Arrays are scanned
/// up to their first 256 elements; if one was longer, \p truncation (when
/// given) says how to stay sound.
llvm::SmallVector<int64_t, 2>
getPointerSlotOffsets(mlir::Type type, const mlir::DataLayout &dataLayout,
                      SlotTruncation *truncation = nullptr);

/// The pointers inside an aggregate SSA value: (byte offset, target cell),
/// one entry per offset getPointerSlotOffsets returns for the value's type, in
/// the same order. An empty Cell is a null pointer.
using AggregateSlots = llvm::SmallVector<std::pair<int64_t, Cell>, 2>;

/// A memory-to-memory copy (cir.copy, cir.libc.memcpy or cir.libc.memmove).
/// After the walk, the builder unifies, for every offset k in [0, length) at
/// which either side has a link, the target of dst+k with the target of
/// src+k, until nothing changes. An unknown length (kUnknown) relates every
/// offset k >= 0, which over-approximates whatever the length is.
///
/// The interprocedural phase must apply the same closure again after it adds
/// links to the nodes (by cloning callees or callers into the graph): a link
/// that appears later on one side is not on the other side yet.
struct MemTransferRecord {
  static constexpr uint64_t kUnknown = ~uint64_t(0);
  mlir::Operation *op = nullptr;
  Cell dst;
  Cell src;
  uint64_t length = 0;
  /// Whether the ghost links in the range are copied too (a byte copy of a
  /// handle is the same handle). False for a model that relates the contents
  /// of two objects whose ghosts describe different things (a SYCL buffer's
  /// storage and host memory).
  bool ghosts = true;
};

/// Apply the copies \p transfers to \p graph until no link changes (see
/// MemTransferRecord). Return false if that took too many rounds and every
/// copy was done field-insensitively instead. A side whose node is collapsed
/// holds its pointers at every offset: the other side gets that link at each
/// pointer-aligned offset in the range. A range over arrays with too many
/// offsets to list (a long or unknown length) makes both sides arrays of one
/// element size and relates one offset per element class. A copy that fits none
/// of this, or a closure that does not settle, collapses both nodes and unifies
/// their links instead, which is sound but coarse.
bool closeMemTransfers(Graph &graph,
                       llvm::ArrayRef<MemTransferRecord> transfers);

/// Record a memory-to-memory copy of \p length bytes (std::nullopt: unknown)
/// from \p src to \p dst made by \p op: Ref on src, Mod on dst, the sizes
/// grown by a constant length, and a MemTransferRecord for the closure. A
/// constant length of zero or less copies nothing. Used by cir.libc.memcpy,
/// cir.libc.memmove and the memcpy-family library models.
void recordMemTransfer(FunctionGraph &fg, mlir::Operation *op, Cell dst,
                       Cell src, std::optional<int64_t> length);
/// The same on \p graph, appending the record to \p transfers; \p ghosts as
/// in MemTransferRecord.
void recordMemTransfer(Graph &graph,
                       llvm::SmallVectorImpl<MemTransferRecord> &transfers,
                       mlir::Operation *op, Cell dst, Cell src,
                       std::optional<int64_t> length, bool ghosts = true);

/// How the local phase treated a call.
enum class CallKind : uint8_t {
  /// A direct call to a function with a body: recorded only, the
  /// interprocedural phase (T08) connects it.
  Defined,
  /// A library model gave the call its effect (CallSiteRecord::modelled).
  Modelled,
  /// A direct call to a declaration without a model (or to a missing
  /// symbol): the unknown-call rule.
  UnknownDecl,
  /// An indirect call whose callee pointer points only to functions known by
  /// symbol: the targets that are declarations give their effect (model or
  /// unknown-call rule) now; the defined ones are not bound. Only without
  /// setDeferIndirectCalls, as is the next kind; PointsToAnalysis defers.
  IndirectResolved,
  /// An indirect call whose targets are not known: the unknown-call rule.
  IndirectUnknown,
  /// An indirect call left to the interprocedural phase
  /// (LocalBuilder::setDeferIndirectCalls): recorded only, no effect yet.
  IndirectDeferred,
};

llvm::StringRef getCallKindName(CallKind kind);

/// A call recorded by the local phase, with the cells involved. Calls to
/// declarations get their effect here, from a library model or from the
/// unknown-call rule; calls to defined functions are only recorded, for the
/// interprocedural phases (T08/T10) to act on.
struct CallSiteRecord {
  /// The cir.call or cir.try_call; for a synthetic site, the op that implies
  /// the call (cir.delete_array).
  mlir::Operation *call = nullptr;
  /// The callee of a direct call or of a synthetic site.
  mlir::FlatSymbolRefAttr callee;
  /// Set for a call the IR does not spell as a call op: the element
  /// destructor and the operator delete of a cir.delete_array. `args` then
  /// holds the cells passed (the first is `this` or the freed pointer).
  bool synthetic = false;
  CallKind kind = CallKind::Defined;
  /// The name of the model that gave the call its effect, or empty.
  llvm::StringRef modelled;
  /// Set by the unknown-call rule: the callee may read or write any
  /// non-constant global with non-internal linkage. The globals graph (T08)
  /// applies this; the local phase only records it.
  bool clobbersGlobals = false;
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

/// A kernel launch a model recorded (the SYCL plug-in's launch sinks): what
/// the launched kernel receives. The interprocedural phase maps the cells
/// into the TU graph (PointsToAnalysis::getLaunches).
struct LaunchRecord {
  /// The launching call.
  mlir::Operation *call = nullptr;
  /// The model that recorded it.
  llvm::StringRef model;
  /// The device kernels the call may launch, by symbol; empty if the model
  /// could not tell.
  llvm::SmallVector<mlir::StringAttr, 2> kernelNames;
  /// The kernel function object (closure) the kernel receives a copy of, if
  /// the model found it, and its type (the record a pointer to it points to).
  std::optional<Cell> closure;
  mlir::Type closureType;
  /// The launch geometry object (range or nd_range), if passed by address.
  std::optional<Cell> geometry;
  /// The bottom-up phase: for a record cloned into a caller's graph, the
  /// calls from that caller down to the function holding `call` (a
  /// CallPathTable id, outermost call first). 0 for the record of the
  /// launching function itself.
  uint32_t ctx = 0;
};

/// An effect on what a pointer reaches that the local phase applies after
/// the walk (and after closeMemTransfers), once every link of the function
/// is known: flow-insensitively, what is stored into a node after a call is
/// also visible to the callee.
struct DeferredEffect {
  enum class Kind : uint8_t {
    /// Set `flags` on the node of `cell` and on every node reachable from it
    /// (Graph::markReachable).
    Reachable,
  };
  Kind kind;
  Cell cell;
  NodeFlags flags;
  const Reason *why;
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
  ///   call <op> [@<callee>] [synthetic] [callee=<cell>]
  ///       args={<i>: <cell>, ...} result=<cell>
  ///       [slots={<i>: <slots>, ...}] [result.slots=<slots>]
  ///       kind=<CallKind>[ model=<name>][ clobbers-globals]
  /// \endcode
  /// (the call on one line). Only pointer formals and arguments are listed
  /// (every argument of a synthetic site); a missing cell prints as
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
  /// The calls in the body, in walk order. A synthetic site (a call a model
  /// makes on behalf of the call it models) follows its call's record.
  llvm::SmallVector<CallSiteRecord, 4> callSites;
  /// The copies, in walk order.
  llvm::SmallVector<MemTransferRecord, 0> memTransfers;
  /// The kernel launches the models recorded, in walk order.
  llvm::SmallVector<LaunchRecord, 0> launches;
  /// The node of each global and function symbol the graph names (its
  /// storage starts at the cell), sorted by symbol name. The
  /// interprocedural phase unifies the copies of one symbol across graphs.
  llvm::SmallVector<std::pair<mlir::FlatSymbolRefAttr, Cell>, 0> symbolCells;

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
  /// \p models, if given, gives calls to the functions it models their
  /// effect; a call to a declaration it does not model (or any declaration,
  /// without a registry) gets the unknown-call rule.
  explicit LocalBuilder(cir::FuncOp func,
                        mlir::SymbolTableCollection *symbols = nullptr,
                        ReasonTable *reasons = nullptr,
                        const PointsToModelRegistry *models = nullptr);
  ~LocalBuilder();

  /// Declare that every caller of the function is known and will be bound
  /// to it by the interprocedural phase: the formals then do not start
  /// Incomplete. Call before build(). Off by default.
  void setFormalsComplete(bool complete) { formalsComplete = complete; }

  /// Leave every indirect call to the interprocedural phase: record it as
  /// CallKind::IndirectDeferred and give it no effect. Its callee pointer is
  /// only known once the graphs of the callers and callees are joined, so a
  /// local decision would apply the unknown-call rule to calls that the TU
  /// graph resolves. The graph is then only sound together with that phase.
  /// Call before build(). Off by default.
  void setDeferIndirectCalls(bool defer) { deferIndirectCalls = defer; }

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
  void visitVTableAddrPoint(cir::VTableAddrPointOp op);
  void visitVTTAddrPoint(cir::VTTAddrPointOp op);
  /// cir.vtable.get_virtual_fn_addr / get_type_info: the cell \p offset
  /// bytes from the one \p vptr points to.
  void visitVPtrStep(mlir::Operation *op, mlir::Value vptr, int64_t offset);
  void visitGetMethod(cir::GetMethodOp op);
  void visitCollapse(mlir::Operation *op, mlir::Value src);
  void visitAtomic(mlir::Operation *op, mlir::Value addr, mlir::Value stored,
                   mlir::Value loaded);
  void visitCall(cir::CIRCallOpInterface op);
  void visitGetGlobal(cir::GetGlobalOp op);
  void visitReturn(cir::ReturnOp op);
  void visitCopy(cir::CopyOp op);
  void visitMemTransfer(mlir::Operation *op, mlir::Value dst, mlir::Value src,
                        mlir::Value length);
  void visitMemChr(cir::MemChrOp op);
  void visitStdFind(cir::StdFindOp op);
  /// cir.array.ctor / cir.array.dtor: the element pointer of each region is
  /// the array's cell, on a node made an array of the element size.
  void visitArrayInit(mlir::Operation *op, mlir::Value addr,
                      mlir::Value numElements);
  void visitDeleteArray(cir::DeleteArrayOp op);
  /// Give the call site \p index of fg->callSites its effect: a model, the
  /// unknown-call rule, or nothing (a defined callee). \p call is null for a
  /// synthetic site.
  void dispatchCall(size_t index, cir::CIRCallOpInterface call);
  /// The unknown-call rule: every node reachable from a pointer argument
  /// becomes External (with Mod and Ref), the result points to an unknown
  /// object, and the site clobbers the globals.
  void applyUnknownCall(CallSiteRecord &site, const Reason *why);
  /// Decide the indirect calls once the walk is over: resolved if the callee
  /// pointer points only to functions, the unknown-call rule otherwise.
  void resolveIndirectCalls();
  /// Give a resolved indirect call the effect of each target that is a
  /// declaration (its model, or the unknown-call rule).
  void applyResolvedTargets(CallSiteRecord &site);
  void applyDeferredEffects();
  /// Collapse the nodes of \p dst and \p src and unify their single links:
  /// a copy whose extent or layout is unknown.
  void copyFieldInsensitive(Cell dst, Cell src, mlir::Operation *op,
                            llvm::StringRef detail);
  /// If \p type has pointer slots getPointerSlotOffsets did not scan, smash
  /// the node of \p addr, which holds a value of \p type, so that the
  /// scanned slots stand for the missing ones (see SlotTruncation).
  void smashIfTruncated(Cell addr, mlir::Type type, mlir::Operation *op);
  /// The slots of the constant aggregate \p value of type \p type, or
  /// std::nullopt if part of it is not understood.
  std::optional<AggregateSlots>
  constantSlots(mlir::Attribute value, mlir::Type type, mlir::Operation *op);
  /// The cell a pointer constant points to.
  Cell constantPointerCell(const ConstantPointer &pointer, mlir::Operation *op);
  void visitRegionTerminator(mlir::Operation *op);
  void visitBranch(mlir::Operation *op);

  cir::FuncOp func;
  mlir::DataLayout dataLayout;
  mlir::SymbolTableCollection *symbols;
  std::unique_ptr<mlir::SymbolTableCollection> ownedSymbols;
  std::unique_ptr<FunctionGraph> fg;
  /// One Global or Function node per symbol, with its initializer linked.
  std::unique_ptr<GlobalNodes> globals;
  struct PointerLayout {
    llvm::SmallVector<int64_t, 2> offsets;
    SlotTruncation truncation;
  };
  /// Cached; returned by value, as the cache may grow while it is used.
  PointerLayout getPointerLayout(mlir::Type type);
  llvm::DenseMap<mlir::Type, PointerLayout> pointerLayouts;
  /// The values a rule defined (as opposed to placeholders created by a use
  /// seen before the definition).
  llvm::DenseSet<mlir::Value> defined;
  llvm::StringMap<unsigned> pessimisticOps;
  llvm::StringMap<unsigned> unhandledOps;
  llvm::StringMap<unsigned> unclassifiedOps;
  const PointsToModelRegistry *models;
  /// Effects on reachable nodes, applied after the walk.
  llvm::SmallVector<DeferredEffect, 0> deferredEffects;
  /// The indices in fg->callSites of the indirect calls, decided after the
  /// walk.
  llvm::SmallVector<size_t, 0> indirectCalls;
  bool verifyErrors = false;
  bool formalsComplete = false;
  bool deferIndirectCalls = false;
};

} // namespace cir::pointsto

#endif // CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_LOCALBUILDER_H
