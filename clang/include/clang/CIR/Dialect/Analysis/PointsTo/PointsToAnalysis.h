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
//  4. Rerun the byte-range copy closure (closeMemTransfers) over every copy
//     of the TU: the links the bindings added can match more offsets. Calls
//     to declarations already got their effect, from a model or the
//     unknown-call rule, in the local phase.
//  5. To a fixpoint of all of the following rules:
//     - propagate External and Incomplete along the links to everything
//       reachable, each node keeping its first reason;
//     - apply clobbersGlobals: every non-constant global that another TU can
//       name or whose address escaped (its node is External or Incomplete)
//       becomes External, Mod and Ref, the reason naming the first such
//       call; then propagate again;
//     - resolve the indirect calls (the local phase leaves them all here,
//       LocalBuilder::setDeferIndirectCalls). A call whose callee node has
//       only function origins and is neither External, Incomplete nor
//       UnknownObj calls exactly those functions: each new target with a
//       body is bound like a direct call, each declaration gets its model or
//       the unknown-call rule. Any other callee node gets the unknown-call
//       rule (Reason IndirectUnresolved). Binding merges nodes and so can
//       give other callee nodes more targets; the copy closure reruns after
//       every round that bound something. A round is one pass over the
//       calls; after PointsToAnalysisOptions::maxIndirectRounds rounds that
//       bound a target, a call that gains another one gets the unknown-call
//       rule instead (Reason Budget). A callee node that is still empty (no
//       origin, no flag) when nothing else changes gets the unknown-call
//       rule too.
//
//  6. Mark the summary nodes (markSummaries): a node that may stand for
//     more than one simultaneously live concrete object gets Summary, with a
//     Reason{SummaryObject, op, "loop" | "recursion" | "several callers" |
//     "merged origins"}. Alias queries never answer MustAlias for such a
//     node.
//
// Queries (PointsToQueries.cpp) run on the finished graph: alias, getModRef
// and explainAlias. CIRPointsToAliasAnalysis wraps them as an
// mlir::AliasAnalysis implementation.
//
// Replaced definitions: a call that a model which replacesDefinition()
// accepts is modelled in the local phase and never bound, even if its callee
// has a body. A function with a body every use of which is such a call, or a
// direct call from another such function, is skipped: not analysed at all.
// A function with a caller of either kind is not bound to that caller, so
// its formals are Incomplete whatever the other rules say.
//
// Formals: a function's formals start Incomplete (Reason UnknownCaller)
// unless all of its callers are visible: it has internal or private linkage
// and every use of its symbol is a direct call from a function this
// analysis walks, bound to the body. PointsToAnalysisOptions::assumeClosedTU
// treats every function that way. The return cell of a function that may have
// unknown callers is Incomplete too: what it returns reaches code we don't see.
// Complete formals are revised in the fixpoint of step 5: a function whose
// Function node turns External (its address escaped to code we don't see),
// or that is a target of an unresolved indirect call, may be called by code
// we don't see with anything, so its formals and return cell become
// Incomplete, Reason{UnknownCaller, func, "address escapes to unknown code"}
// chained to the cause, and TUFunction::formalsComplete becomes false.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_POINTSTOANALYSIS_H
#define CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_POINTSTOANALYSIS_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/LocalBuilder.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToGraph.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <functional>
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
  /// The number of rounds of indirect-call resolution that may bind new
  /// targets; past it, a call that gains a target gets the unknown-call
  /// rule (Reason Budget).
  unsigned maxIndirectRounds = 32;
  /// If set, called on a registry the analysis owns, which is asked before
  /// the registry passed to the constructor (its fallback): how a client
  /// adds a plug-in's models (the SYCL ones, from OffloadOpt) without this
  /// library depending on the plug-in.
  std::function<void(PointsToModelRegistry &)> extraModels;
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
  /// An indirect call whose callee pointer points only to known functions
  /// (TUCallSite::targets): bound to the bodies, the declarations given
  /// their model or the unknown-call rule.
  Indirect,
  /// An indirect call whose targets are not known (or that ran out of
  /// rounds): the unknown-call rule, for TUCallSite::unresolvedWhy. The
  /// targets bound before it became unresolved stay bound.
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
  /// The targets of an indirect call, in the order they were found (bound,
  /// or given their model or the unknown-call rule).
  llvm::SmallVector<mlir::FlatSymbolRefAttr, 2> targets;
  /// Why an indirect call is CallResolution::IndirectUnresolved.
  const Reason *unresolvedWhy = nullptr;
};

/// The roots of one function in the TU graph.
struct TUFunction {
  cir::FuncOp func;
  /// Whether every caller is visible, so that the formals are complete. Set
  /// before propagation, cleared if the function escapes (see the file
  /// comment).
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

/// What a pointer may point to (PointsToAnalysis::pointsTo) or a memory cell
/// may hold (PointsToAnalysis::objectsAt). With unification that is at most
/// one cell.
struct ObjectSet {
  /// The resolved cells; empty for a null pointer or an empty slot.
  llvm::SmallVector<Cell, 1> cells;
  /// Whether the cells are all it may point to: no node is Incomplete,
  /// External or UnknownObj, after status propagation.
  bool complete = true;
  /// Why the set is not complete: the first reason recorded on the node
  /// (Incomplete before External). Null for a complete set.
  const Reason *why = nullptr;
};

/// One line for \p set, the text remarks print:
///  - an incomplete set: "pointer may refer to unknown memory: <explain(why)>";
///  - a complete set: "pointer refers to known memory", followed by
///    "; <explain>" of its Summary reason if the object is a summary;
///  - a complete empty set: "pointer refers to no object".
std::string explain(const ObjectSet &set);

/// The rule of PointsToAnalysis::alias that decided a query, for
/// explanations and statistics. See PointsToQueries.cpp for each rule's
/// soundness argument.
enum class AliasRule : uint8_t {
  /// The same SSA value: MustAlias.
  SameValue,
  /// One side is a null pointer, which points to no object: NoAlias.
  NullPointer,
  /// Both sides are null, or derived from null: MayAlias.
  BothNull,
  /// A value the analysis has no cell for (a function without a body, or
  /// one in another module): MayAlias.
  NotAnalysed,
  /// Different nodes, neither reachable by unknown code: NoAlias.
  DistinctObjects,
  /// Different nodes; one may refer to unknown memory, but the other's
  /// object never escaped and came from no unknown code: NoAlias.
  UnescapedObject,
  /// Different nodes, both may refer to unknown memory: MayAlias.
  BothUnknown,
  /// The same node, which unknown code can reach: its offsets say nothing
  /// about where a pointer that code made points. MayAlias.
  SameNodeUnknown,
  /// The same collapsed node: MayAlias.
  Collapsed,
  /// The same array node, each access within one element at disjoint
  /// element-relative ranges: NoAlias.
  ArrayDisjoint,
  /// The same array node otherwise: MayAlias.
  ArrayMayOverlap,
  /// The same node at disjoint byte ranges: NoAlias.
  DisjointRanges,
  /// The same singleton node at the same offset: MustAlias.
  SameAddress,
  /// The same offset of a node that is not a singleton (a Summary, several
  /// or no origins): MayAlias.
  SameOffsetNotSingleton,
  /// The same node, an access size is unknown and the ranges are not
  /// provably disjoint: MayAlias.
  UnknownSize,
  /// Overlapping ranges at different offsets of a singleton: PartialAlias.
  PartialOverlap,
  /// Overlapping ranges of a node that is not a singleton: MayAlias.
  OverlapNotSingleton,
};

llvm::StringRef getAliasRuleName(AliasRule rule);

/// An alias answer and the rule that gave it.
struct AliasAnswer {
  mlir::AliasResult result;
  AliasRule rule;
};

/// Counters of one run, for statistics and tests.
struct PointsToAnalysisStats {
  size_t numFunctions = 0;
  /// Calls bound to a body, and the actual/formal and result/return pairs
  /// unified for them.
  size_t numDirectCalls = 0;
  size_t numBindings = 0;
  /// Indirect calls by final resolution, the targets with a body bound for
  /// them, the rounds that bound a target, and the calls that ran out of
  /// rounds.
  size_t numIndirectResolved = 0;
  size_t numIndirectUnresolved = 0;
  size_t numIndirectTargetsBound = 0;
  size_t numIndirectRounds = 0;
  size_t numIndirectBudget = 0;
  size_t numClobberingCalls = 0;
  /// Functions whose formals start complete.
  size_t numCompleteFunctions = 0;
  /// Of those, the functions whose formals became Incomplete because code we
  /// don't see may call them (markEscapedFunctions).
  size_t numEscapedFunctions = 0;
  /// Functions in a recursive call-graph SCC (direct calls and bound
  /// indirect targets), and nodes marked Summary.
  size_t numRecursiveFunctions = 0;
  size_t numSummaryNodes = 0;
  /// Calls to a function with a body that a model replaced, and the
  /// functions with a body that are not analysed because every use of them
  /// is such a call (or a call from another such function).
  size_t numReplacedCalls = 0;
  size_t numSkippedFunctions = 0;
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

  /// What the pointer \p value may point to. A null pointer constant gives
  /// a complete empty set; a value the analysis has no cell for (not a
  /// pointer, or not in an analysed function) an incomplete empty set.
  ObjectSet pointsTo(mlir::Value value) const;
  /// What the pointer slot at \p cell may hold: its link target. A slot
  /// without a link holds nothing seen, which is all it may hold only if
  /// its node is complete.
  ObjectSet objectsAt(Cell cell) const;

  /// An access size that reaches to the end of the object.
  static constexpr uint64_t kUnknownSize = ~uint64_t(0);

  /// Whether accesses of \p sizeA bytes at \p a and of \p sizeB bytes at
  /// \p b may overlap. A size of std::nullopt is the size of the pointee
  /// type (kUnknownSize if it has none), kUnknownSize reaches to the end of
  /// the object. Values of a function this analysis does not walk give
  /// MayAlias. The rules and their soundness arguments are in
  /// PointsToQueries.cpp.
  mlir::AliasResult alias(mlir::Value a, mlir::Value b,
                          std::optional<uint64_t> sizeA = std::nullopt,
                          std::optional<uint64_t> sizeB = std::nullopt) const;
  /// The same, with the rule that decided it.
  AliasAnswer aliasWithRule(mlir::Value a, mlir::Value b,
                            std::optional<uint64_t> sizeA = std::nullopt,
                            std::optional<uint64_t> sizeB = std::nullopt) const;
  /// One line saying why alias(a, b) gives its answer, for remarks. For a
  /// MayAlias it names what each side may refer to, for example
  ///   "a: reachable from an escaped pointer passed to @foo (t.cpp:3);
  ///    b: same node ..."
  /// A node with several origins also says how many objects were merged
  /// into it and, if it collapsed, why (not only the first reason it got).
  std::string explainAlias(mlir::Value a, mlir::Value b,
                           std::optional<uint64_t> sizeA = std::nullopt,
                           std::optional<uint64_t> sizeB = std::nullopt) const;

  /// Whether \p op may read (Ref) or write (Mod) the memory \p location
  /// points to, an access of its pointee type:
  ///  - a load, store, copy, memcpy-family op or atomic: from alias() of
  ///    its address operands with \p location;
  ///  - a call: NoModRef if the object of \p location is complete (neither
  ///    External, Incomplete nor UnknownObj) and is not reachable from the
  ///    call's pointer arguments, its result, or a global the callee may
  ///    name; otherwise Ref if no code ever writes the node, Mod if none
  ///    ever reads it, and ModRef if both (context-insensitive, so coarse).
  ///    A HasUSM node (a USM allocation, which device code may access while
  ///    the host runs) or AsyncAccess node (memory an asynchronous command
  ///    accesses) is ModRef for every call;
  ///  - any other op: NoModRef if it has no memory effect but allocation,
  ///    ModRef otherwise.
  mlir::ModRefResult getModRef(mlir::Operation *op, mlir::Value location) const;

  mlir::ModuleOp getModule() const { return module; }
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
  /// The TU node of every global and function symbol some function names,
  /// sorted by name.
  llvm::SmallVector<std::pair<mlir::FlatSymbolRefAttr, Node *>, 0>
  getSymbolNodes() const;

  /// Whether \p func is in a recursive call-graph SCC.
  bool isRecursive(cir::FuncOp func) const;

  /// The kernel launches the models recorded, their cells in the TU graph,
  /// in function processing order.
  llvm::ArrayRef<LaunchRecord> getLaunches() const { return launches; }

  /// Whether \p func has a body but is not analysed, because every use of it
  /// is a call that a model replaces (see PointsToModel::replacesDefinition)
  /// or a call from another such function.
  bool isSkipped(cir::FuncOp func) const {
    return skipped.contains(func.getOperation());
  }

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
  friend mlir::LogicalResult
  verifyPointsToAnalysis(const PointsToAnalysis &analysis);

  void run();
  void buildLocal(cir::FuncOp func, bool formalsComplete);
  /// Whether \p func has a body directly in the module and is not skipped.
  bool isAnalysed(cir::FuncOp func) const;
  /// Find the calls a model replaces, the functions skipped because of them,
  /// and the functions whose callers include an unbound one.
  void computeReplacedCalls();
  void computeVisibleCallers();
  /// Unify the actual/formal and result/return pairs of every direct call,
  /// and of every indirect call with the targets it was bound to. With
  /// \p dryRun, change nothing and count the pairs that are not unified
  /// yet. Returns that count (or the number of pairs unified).
  size_t bindCalls(bool dryRun);
  /// The same for \p call and the body of \p callee.
  size_t bindCall(TUCallSite &call, const TUFunction &callee, bool dryRun);
  /// One round of indirect-call resolution (see the file comment). With
  /// \p finalize, the calls whose callee node is still empty get the
  /// unknown-call rule. Returns whether a rule changed the graph; sets
  /// \p bound if a target was bound.
  bool resolveIndirectCalls(bool finalize, bool &bound);
  /// Give \p call the effect of its target \p target, which has no body
  /// here: a model, or the unknown-call rule.
  void applyDeclarationTarget(TUCallSite &call, mlir::FlatSymbolRefAttr target,
                              cir::FuncOp calleeOrNull);
  void applyUnknownCall(TUCallSite &call, const Reason *why);
  void propagateStatus();
  bool clobberGlobals();
  /// Make Incomplete the formals and the return cell of every function with
  /// complete formals that code we don't see may call: its Function node is
  /// External, or it is a target of an unresolved indirect call. Returns
  /// whether it marked a function.
  bool markEscapedFunctions();
  /// Find the recursive functions: the call-graph SCCs with a cycle, over
  /// the direct calls and the bound indirect targets.
  void computeRecursion();
  /// Set Summary on the nodes that may stand for more than one live object
  /// (see the comment in the .cpp file).
  void markSummaries();

  // Queries (PointsToQueries.cpp).
  uint64_t getAccessSize(mlir::Value ptr, std::optional<uint64_t> size) const;
  AliasAnswer aliasCells(Cell a, Cell b, uint64_t sizeA, uint64_t sizeB) const;
  std::string explainAnswer(const AliasAnswer &answer, mlir::Value a,
                            mlir::Value b) const;
  mlir::ModRefResult getAccessModRef(mlir::Value addr,
                                     std::optional<uint64_t> size, bool mod,
                                     bool ref, mlir::Value location) const;
  mlir::ModRefResult getCallModRef(mlir::Operation *call,
                                   mlir::Value location) const;
  const TUCallSite *findCallSite(mlir::Operation *call) const;
  /// The nodes a call may access: those reachable from its pointer
  /// arguments, its result and the globals its callees may name.
  const llvm::DenseSet<const Node *> &
  getCallReach(const TUCallSite &call) const;
  /// Add to \p out the symbol nodes \p fn and its callees, transitively, may
  /// name. Returns false if one of them may call code we don't see, which
  /// may name any global.
  bool collectNamedSymbols(const TUFunction &fn,
                           llvm::DenseSet<const TUFunction *> &visited,
                           llvm::SmallVectorImpl<Cell> &out) const;
  /// Fill the caches below the first time a query needs them.
  void buildQueryIndex() const;
  const TUFunction *lookupFunction(mlir::StringAttr name) const;
  Node *lookupSymbolNode(mlir::StringAttr name) const;

  mlir::ModuleOp module;
  const PointsToModelRegistry *models;
  /// The registry of PointsToAnalysisOptions::extraModels, if any.
  std::unique_ptr<PointsToModelRegistry> ownedModels;
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
  /// call from an analysed function, bound to its body.
  llvm::DenseSet<mlir::Operation *> visibleCallers;
  /// The calls to a function with a body that a model replaces.
  llvm::DenseSet<mlir::Operation *> replacedCalls;
  /// The functions with a body that are not analysed (isSkipped).
  llvm::DenseSet<mlir::Operation *> skipped;
  /// The functions some caller of which is not bound to their body: a call a
  /// model replaces, or a call from a skipped function. Their formals are
  /// never complete, closed TU or not.
  llvm::DenseSet<mlir::Operation *> unboundCallers;
  llvm::SmallVector<LaunchRecord, 0> launches;
  /// The functions in a recursive SCC, with Reason{SummaryObject, func,
  /// "recursion"}.
  llvm::DenseMap<mlir::Operation *, const Reason *> recursive;
  /// The reason pointsTo gives a value without a cell.
  const Reason *notAnalysedWhy = nullptr;
  llvm::StringMap<unsigned> unhandledOps;
  PointsToAnalysisStats stats;
  bool verifyErrors = false;

  // Query caches, filled lazily; the graph does not change after run().
  mutable bool queryIndexBuilt = false;
  mutable llvm::DenseMap<mlir::Operation *, const TUCallSite *> callSiteOf;
  /// The synthetic sites a model added on a call op (see
  /// ModelContext::addIndirectCall).
  mutable llvm::DenseMap<mlir::Operation *,
                         llvm::SmallVector<const TUCallSite *, 1>>
      syntheticSitesOf;
  mutable llvm::DenseMap<mlir::StringAttr, const TUFunction *> functionByName;
  mutable llvm::DenseMap<mlir::StringAttr, Node *> symbolNodeOf;
  mutable llvm::DenseMap<mlir::Operation *,
                         std::unique_ptr<llvm::DenseSet<const Node *>>>
      callReach;
};

} // namespace cir::pointsto

#endif // CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_POINTSTOANALYSIS_H
