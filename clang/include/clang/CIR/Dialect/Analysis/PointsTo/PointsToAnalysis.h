//===- PointsToAnalysis.h - Interprocedural CIR points-to -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the driver of the CIR points-to analysis. It builds the
// local graph of every function of a module (LocalBuilder), merges them into
// one graph for the whole translation unit, and connects the call sites to
// their callees there.
//
// This is the context-insensitive analysis (Steensgaard with byte offsets
// over the whole TU): each function has one set of formals and one return
// cell, which all of its callers share. A function called from two places
// therefore merges what the two callers pass it, and an object allocated in
// a function stands for every object allocated there, whoever the caller.
// A context-sensitive analysis that clones callee graphs into their callers
// is a later step; this one stays as its fallback.
//
// The steps, in order:
//
//  1. Build the local graph of each cir.func with a body, in call-graph SCC
//     post-order, and clone it into the TU graph. All the graphs share one
//     ReasonTable, so reasons are not copied.
//  2. Unify the per-function copies of each global and function symbol into
//     one node per symbol, whose initializer the TU's GlobalNodes reads once.
//  3. Bind every direct call to a defined callee: actuals to formals (slot by
//     slot for aggregates), the result to the return cell. An argument past
//     the callee's formals (a variadic argument) escapes.
//  4. Give the indirect calls with a defined target the unknown-call rule
//     (resolving them is a later step). Calls to declarations already got
//     their effect, from a model or the unknown-call rule, in the local
//     phase.
//  5. Rerun the byte-range copy closure (closeMemTransfers) over every copy
//     of the TU: the links the bindings added can match more offsets.
//  6. Propagate External and Incomplete along the links to everything
//     reachable, each node keeping its first reason. An indirect call the
//     local phase resolved whose callee node is now External, Incomplete or
//     not only functions gets the unknown-call rule; repeat until that
//     changes nothing.
//  7. Apply clobbersGlobals: every non-constant global that another TU can
//     name or whose address escaped (its node is External or Incomplete)
//     becomes External, Mod and Ref, the reason naming the first such call.
//     Then propagate once more.
//
// Formals: a function's formals start Incomplete (Reason UnknownCaller)
// unless all of its callers are visible: it has internal or private linkage
// and every use of its symbol is a direct call from a function this
// analysis walks. PointsToAnalysisOptions::assumeClosedTU treats every
// function that way. The return cell of a function that may have unknown
// callers is Incomplete too: what it returns reaches code we don't see.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_POINTSTOANALYSIS_H
#define CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_POINTSTOANALYSIS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/LocalBuilder.h"
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

class GlobalNodes;
class PointsToModelRegistry;

struct PointsToAnalysisOptions {
  /// Treat every function as if all of its callers were in the module: no
  /// formal starts Incomplete. Tests set it; the -cir-offload-assume-closed-tu
  /// flag maps onto it later.
  bool assumeClosedTU = false;
};

/// What the interprocedural phase did with a call site.
enum class CallResolution : uint8_t {
  /// Bound to the body of a defined callee.
  Direct,
  /// A library model gave the call its effect (in the local phase).
  Model,
  /// A call to a declaration without a model: the unknown-call rule (in the
  /// local phase).
  Unknown,
  /// An indirect call whose targets are not known, or include a defined
  /// function: the unknown-call rule.
  IndirectUnresolved,
};

llvm::StringRef getCallResolutionName(CallResolution resolution);

/// A call site of a function, its cells in the TU graph.
struct TUCallSite {
  /// The local record, its cells mapped into the TU graph.
  CallSiteRecord site;
  CallResolution resolution = CallResolution::Unknown;
  /// The callee bound, for CallResolution::Direct.
  cir::FuncOp callee;
};

/// The roots of one function in the TU graph.
struct TUFunction {
  cir::FuncOp func;
  /// Whether every caller is visible, so that the formals start complete.
  bool formalsComplete = false;
  /// One entry per entry-block argument: the cell of a pointer formal.
  llvm::SmallVector<std::optional<Cell>, 4> formals;
  /// One entry per entry-block argument: the slots of an aggregate formal
  /// that holds pointers.
  llvm::SmallVector<std::optional<AggregateSlots>, 4> formalSlots;
  std::optional<Cell> returnCell;
  std::optional<AggregateSlots> returnSlots;
  llvm::SmallVector<TUCallSite, 4> callSites;
};

/// Counters of one run, for statistics and tests.
struct PointsToAnalysisStats {
  size_t numFunctions = 0;
  /// Calls bound to a body, and the actual/formal and result/return pairs
  /// unified for them.
  size_t numDirectCalls = 0;
  size_t numBindings = 0;
  size_t numIndirectUnresolved = 0;
  size_t numClobberingCalls = 0;
  /// Functions whose formals start complete.
  size_t numCompleteFunctions = 0;
  /// Whether the TU-wide copy closure settled (rather than smashing).
  bool transfersSettled = true;
  size_t numLocalNodes = 0;
  double localSeconds = 0;
  double totalSeconds = 0;
};

/// The context-insensitive points-to analysis of a module (see the file
/// comment).
class PointsToAnalysis {
public:
  /// Analyse the cir.func ops with a body directly in \p module. Calls to
  /// declarations get the models of \p models (none if null, so they all get
  /// the unknown-call rule).
  explicit PointsToAnalysis(mlir::ModuleOp module,
                            const PointsToModelRegistry *models = nullptr,
                            PointsToAnalysisOptions options = {});
  ~PointsToAnalysis();
  PointsToAnalysis(const PointsToAnalysis &) = delete;
  PointsToAnalysis &operator=(const PointsToAnalysis &) = delete;

  /// The cell \p value points to, or an empty cell (not a pointer, a null
  /// pointer, or a value of a function without a body).
  Cell cellOf(mlir::Value value) const;
  /// The slots of the aggregate \p value, or null.
  const AggregateSlots *slotsOf(mlir::Value value) const;

  Graph &getGraph() { return *graph; }
  const Graph &getGraph() const { return *graph; }

  /// The roots of \p func in the TU graph, or null if it has no body here.
  const TUFunction *getFunction(cir::FuncOp func) const;
  /// Every analysed function, in the order they were processed (call-graph
  /// SCC post-order).
  llvm::ArrayRef<std::unique_ptr<TUFunction>> getFunctions() const {
    return functions;
  }

  /// The TU node of the global or function \p sym, if any function names it.
  Node *getSymbolNode(mlir::FlatSymbolRefAttr sym) const;

  const PointsToAnalysisStats &getStats() const { return stats; }

  /// The ops the local phase sent to the pessimistic default, summed over
  /// the functions (see LocalBuilder).
  const llvm::StringMap<unsigned> &getUnhandledOps() const {
    return unhandledOps;
  }

  /// Whether a local graph or the TU graph failed verification (under
  /// -cir-points-to-verify).
  bool hadVerifyErrors() const { return verifyErrors; }

private:
  void run();
  void buildLocal(cir::FuncOp func, bool formalsComplete);
  void computeVisibleCallers();
  /// Unify the actual/formal and result/return pairs of every direct call.
  /// With \p dryRun, change nothing and count the pairs that are not unified
  /// yet. Returns that count (or the number of pairs unified).
  size_t bindCalls(bool dryRun);
  void applyUnknownCall(TUCallSite &call, const Reason *why);
  void propagateStatus();
  bool clobberGlobals();

  mlir::ModuleOp module;
  const PointsToModelRegistry *models;
  PointsToAnalysisOptions options;
  mlir::SymbolTableCollection symbols;
  ReasonTable reasons;
  std::unique_ptr<Graph> graph;
  std::unique_ptr<mlir::DataLayout> dataLayout;
  std::unique_ptr<GlobalNodes> globals;

  llvm::SmallVector<std::unique_ptr<TUFunction>, 0> functions;
  llvm::DenseMap<mlir::Operation *, TUFunction *> functionMap;
  llvm::DenseMap<mlir::Value, Cell> scalarMap;
  llvm::DenseMap<mlir::Value, AggregateSlots> aggregateMap;
  llvm::SmallVector<MemTransferRecord, 0> memTransfers;
  /// The functions with local linkage whose every symbol use is a direct
  /// call from a function with a body.
  llvm::DenseSet<mlir::Operation *> visibleCallers;
  llvm::StringMap<unsigned> unhandledOps;
  PointsToAnalysisStats stats;
  bool verifyErrors = false;
};

} // namespace cir::pointsto

#endif // CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_POINTSTOANALYSIS_H
