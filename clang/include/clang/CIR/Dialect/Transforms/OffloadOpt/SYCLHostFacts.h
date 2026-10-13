//===- SYCLHostFacts.h - SYCL host facts from points-to ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The SYCL facade of the CIR points-to analysis: what the host code tells the
// combine about the kernels it launches, read from a PointsToAnalysis of the
// host module with the SYCL plug-in (SYCLPointsToModels.h).
//
// It answers three questions (--cir-offload-host-facts=ipo, see
// HostFactsMode):
//  - InferKernelParamNoAlias's: which captured USM pointers and accessors of
//    a kernel's closure refer to memory distinct from every other one, at
//    every launch of the kernel (noAliasFields);
//  - PropagateSYCLKernelArgs's: which scalar captures hold one constant at
//    every launch (constantFields, from the stored-value queries);
//  - StampReqdWorkGroupSize's: the constant global and local sizes every
//    launch uses (geometry, from the stored-value queries).
//
// Identity, not escape. A kernel launch makes everything its closure reaches
// External (applySYCLDeviceWrites), so the escape rule of the alias queries
// would reject every field. noAliasFields decides by allocation identity
// instead: two fields are distinct when each refers to one allocation site
// (a USM or heap allocation, a buffer's storage), the sites differ, and the
// pointer value each field holds can only be one the analysed code computed.
// See SYCLHostFacts.cpp for the rules and their soundness argument.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_SYCLHOSTFACTS_H
#define CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_SYCLHOSTFACTS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/LocalBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <map>
#include <memory>
#include <optional>
#include <string>

namespace mlir {
class DataLayout;
} // namespace mlir

namespace cir {

namespace pointsto {
class Node;
class PointsToAnalysis;
class PointsToModelRegistry;
} // namespace pointsto

class SYCLClosureLayout;

class SYCLHostFacts {
public:
  /// The memory a closure field of a launch refers to.
  struct MemIdentity {
    enum class Kind : uint8_t {
      /// An accessor: `node` is its `$storage` ghost node, `host` the
      /// storage's `$host` memory (buffer(hostData, ...)), if any.
      Storage,
      /// A pointer: `node` is the object it points to.
      Pointer,
    };
    Kind kind = Kind::Pointer;
    const pointsto::Node *node = nullptr;
    const pointsto::Node *host = nullptr;
    /// Whether `node` (and `host`) each are one allocation and the field can
    /// only hold the value the analysed code gave it (see the .cpp file).
    bool complete = false;
    /// A storage: the elements of the storage the accessor may access,
    /// [lo, hi) in units of the storage's element type (hi std::nullopt: to
    /// the end), if known; std::nullopt: anywhere in the storage. Only
    /// comparable between identities whose storage nodes are equal and
    /// `rangesComparable`.
    std::optional<std::pair<uint64_t, std::optional<uint64_t>>> range;
    /// Whether every origin of the storage is a call of one constructor, so
    /// that the ranges of its views are in units of one element type.
    bool rangesComparable = false;
    /// A storage over host memory: the byte offset in `host` its host
    /// pointer points to, if the analysis knows it exactly (`host` is
    /// neither collapsed nor an array), and the bytes [hostLo, hostHi) of
    /// `host` the accessor may access (hostHi std::nullopt: to the end).
    std::optional<int64_t> hostOffset;
    uint64_t hostLo = 0;
    std::optional<uint64_t> hostHi;
    /// What the field refers to, for a complete identity ("buffer 'ba'
    /// (ghost n12)", "a USM allocation (n7)"), or why it is not complete.
    std::string what;
  };

  /// Run the points-to analysis of the host module of \p container, with the
  /// SYCL plug-in and the generic library models. The device kernels are
  /// found in the container's device modules.
  SYCLHostFacts(mlir::ModuleOp container, const HostFactsOptions &options);
  /// Use \p analysis, which must have been built with the SYCL plug-in
  /// (addSYCLPointsToPlugin). The device kernels are the functions with a
  /// body under \p deviceScope whose arguments carry cir.sycl_kernel_param.
  SYCLHostFacts(const pointsto::PointsToAnalysis &analysis,
                mlir::Operation *deviceScope, bool assumeSpecHostPtrUB = true);
  ~SYCLHostFacts();
  SYCLHostFacts(const SYCLHostFacts &) = delete;
  SYCLHostFacts &operator=(const SYCLHostFacts &) = delete;

  const pointsto::PointsToAnalysis &getAnalysis() const { return *analysis; }

  /// The memory identity of the top-level closure field \p field of
  /// \p launch: an accessor field (a member that is not a pointer) gives its
  /// `$storage` ghost node and `$host`, a pointer field the object it points
  /// to.
  MemIdentity identityOfClosureField(const pointsto::LaunchRecord &launch,
                                     llvm::StringRef field) const;

  /// The host-only memory a pointer member of a launch's closure points
  /// into (the sycl-host-ptr-in-kernel diagnostic, HostDiagnostics.cpp).
  struct HostOnlyMemory {
    enum class Kind : uint8_t {
      /// Not known to be host-only memory: `why` says why.
      Unknown,
      /// A stack object of a host function.
      Stack,
      /// A global variable of the host module.
      Global,
      /// A host heap allocation (malloc, operator new, ...; not USM).
      Heap,
    };
    Kind kind = Kind::Unknown;
    /// The node of the allocation (the root of the subobject tree the
    /// pointer points into).
    const pointsto::Node *node = nullptr;
    std::string why;
  };
  /// Whether the pointer member \p path (member names from the closure, as
  /// cir.sycl_kernel_param records them) of the closure of \p launch can
  /// only point into host memory of one kind: every value it may hold at
  /// the launch is one the analysed code computed (rule (b) of the noalias
  /// rule), points into the node of a host stack object, global or non-USM
  /// heap allocation (every origin of that kind, no unknown or USM
  /// contribution), and no null or other constant is stored into it. See
  /// the .cpp file.
  /// \p elements, if given, continue the path by element index inside the
  /// record the member is (a std_layout capture: elements[0] of the
  /// member's record, elements[1] of that one's, ...).
  HostOnlyMemory
  hostOnlyMemoryOfClosureMember(const pointsto::LaunchRecord &launch,
                                mlir::ArrayAttr path,
                                llvm::ArrayRef<unsigned> elements = {}) const;

  /// The device USM allocations a host pointer must point into (the
  /// sycl-host-deref-device-usm diagnostic, HostDiagnostics.cpp).
  struct DeviceUSMMemory {
    /// The allocation nodes (the roots of the subobject trees the pointer
    /// points into), one per calling context the answer covers; empty if
    /// the pointer is not known to point only into device USM: `why` says
    /// why.
    llvm::SmallVector<const pointsto::Node *, 2> nodes;
    std::string why;
    /// Whether some node the pointer may point into (or its subobject
    /// root) is a USM allocation: the near misses -debug-only reports.
    bool reachesUSM = false;
  };
  /// Whether the host pointer \p ptr, a value of an analysed host function,
  /// can only point into device USM allocations: in every calling context
  /// the analysis covers, it points into a node every origin of which is a
  /// usm.alloc call of kind device (getSYCLUSMAllocKind), with no unknown,
  /// incomplete, may-be-shared, may-overlap or collapsed contribution, that
  /// no caller we don't see can pass in and that only analysed code holds
  /// pointers to. See the .cpp file.
  DeviceUSMMemory deviceUSMOfPointer(mlir::Value ptr) const;

  /// The noalias verdict of every candidate field of \p kernel (the
  /// candidates of its device kernels, getSYCLNoAliasCandidates): a field is
  /// no-alias iff at
  /// every launch of the kernel its identity is complete and distinct from
  /// that of every other candidate, the kernel is not poisoned, and every
  /// launch of the kernel is a recorded launch with a closure. Each decision
  /// carries a one-line note: the memory, or the first reason it was
  /// rejected.
  llvm::StringMap<NoAliasDecision> noAliasFields(llvm::StringRef kernel) const;
  /// The same, with the candidates given.
  llvm::StringMap<NoAliasDecision>
  noAliasFields(llvm::StringRef kernel,
                const SYCLNoAliasCandidates &candidates) const;

  /// The device kernels named \p kernel.
  llvm::ArrayRef<cir::FuncOp> getDeviceKernels(llvm::StringRef kernel) const;

  /// The constant a closure field holds, or why none is known.
  struct ConstFact {
    mlir::TypedAttr value;
    std::string note;
  };
  /// The scalar fields of \p kernel's closure that its device kernels take
  /// as scalar parameters (kind "scalar", a top-level int, float or bool
  /// member), with the parameter type.
  llvm::StringMap<mlir::Type> getConstCandidates(llvm::StringRef kernel) const;
  /// The constant the scalar closure field \p field holds at \p launch:
  /// the stored values of its cell, complete with exactly one constant.
  ConstFact constantOfLaunch(const pointsto::LaunchRecord &launch,
                             llvm::StringRef field, mlir::Type type) const;
  /// The constant \p field holds at every launch of \p kernel: every launch
  /// is known and recorded with a closure, and each gives the same one.
  ConstFact constantField(llvm::StringRef kernel, llvm::StringRef field) const;
  /// constantField for every candidate of \p kernel.
  llvm::StringMap<ConstFact> constantFields(llvm::StringRef kernel) const;

  /// The launch geometry, or why it is not known.
  struct GeometryFact {
    SyclGeometry geometry;
    std::string note;
  };
  /// The geometry of \p launch: the global and local sizes its range,
  /// nd_range or nd_range_view object holds, each component a constant the
  /// stored-value queries prove, or std::nullopt.
  GeometryFact geometryOfLaunch(const pointsto::LaunchRecord &launch) const;
  /// The geometry of every launch of \p kernel, ANDed: the same number of
  /// dimensions, a local size at every launch or at none, and per component
  /// the constant every launch agrees on. Invalid if a launch is not known,
  /// has no geometry the models recorded, or the launches disagree on the
  /// shape.
  GeometryFact geometry(llvm::StringRef kernel) const;

  /// Per launch: the identity of each candidate, and the per-launch verdicts
  /// before the AND over the launches (for the test pass).
  struct LaunchFacts {
    llvm::SmallVector<std::pair<std::string, MemIdentity>, 4> fields;
    llvm::StringMap<NoAliasDecision> decisions;
  };
  LaunchFacts factsOfLaunch(const pointsto::LaunchRecord &launch,
                            const SYCLNoAliasCandidates &candidates) const;

  /// Whether two identities may refer to the same memory: they share a node
  /// or an origin, unless they are two accessors of one storage whose ranges
  /// are disjoint, or two storages over one host object whose host pointers
  /// are equal (with \p assumeSpecHostPtrUB, see HostFactsOptions) or whose
  /// host bytes are disjoint. See the .cpp file.
  static bool sameMemory(const MemIdentity &a, const MemIdentity &b,
                         bool assumeSpecHostPtrUB = true);

  /// The budgets of the points-to analysis that tripped so far
  /// (PointsToAnalysis::formatBudgetTrips), if they changed since the last
  /// call: what the passes report in a "host-facts budget trips" remark.
  std::optional<std::string> takeBudgetReport() const;

private:
  /// The uncached queries of noAliasFields and geometry.
  llvm::StringMap<NoAliasDecision>
  computeNoAliasFields(llvm::StringRef kernel,
                       const SYCLNoAliasCandidates &candidates) const;
  GeometryFact computeGeometry(llvm::StringRef kernel) const;
  /// Whether the answers are memoized, and the memoized answers (see the
  /// .cpp file).
  bool memoize = false;
  mutable std::map<std::string, llvm::StringMap<NoAliasDecision>> noAliasMemo;
  mutable llvm::StringMap<llvm::StringMap<ConstFact>> constantsMemo;
  mutable llvm::StringMap<GeometryFact> geometryMemo;

  void collectDeviceKernels(mlir::Operation *scope);
  /// Index the finished graph: holders, handles, formal targets, and the
  /// launch paths no model records.
  void buildIndex();
  /// Why the value a pointer slot holds that points to \p node may not be
  /// one the analysed code computed, or std::nullopt. \p storage: \p node is
  /// a storage node reached through `$storage` ghost links.
  std::optional<std::string> unknownValueWhy(const pointsto::Node *node,
                                             bool storage) const;
  /// Why \p node is not one allocation of the kinds \p storage (a buffer
  /// storage) or not (a heap or USM allocation; with \p host also a stack
  /// object or a global) accepts, or std::nullopt.
  std::optional<std::string> notOneAllocationWhy(const pointsto::Node *node,
                                                 bool storage, bool host) const;
  /// BU mode: the nodes of the functions' own graphs reachable from their
  /// formals, and every parent of each (deviceUSMOfPointer and
  /// hostOnlyMemoryOfClosureMember); built on first
  /// use.
  const llvm::DenseSet<const pointsto::Node *> &getFormalReach() const;
  mutable std::optional<llvm::DenseSet<const pointsto::Node *>> formalReach;
  /// A short name of the allocation of \p node, for notes.
  std::string describeAllocation(const pointsto::Node *node) const;
  /// The part of a storage a view (`$view` ghost node) gives access to.
  struct ViewRange {
    /// Whether [lo, hi) is known: the elements (of the storage's element
    /// type) the view may access are in it. Otherwise: anywhere.
    bool known = false;
    uint64_t lo = 0;
    /// std::nullopt: to the end of the storage.
    std::optional<uint64_t> hi;
    /// The view's own range per dimension (what an accessor's offset is
    /// linearized against), if known; empty otherwise.
    llvm::SmallVector<uint64_t, 3> dims;
    /// The number of dimensions of the buffer, if known (0 otherwise).
    unsigned rank = 0;
    /// The elements of the whole storage, if known (the buffer's range).
    std::optional<uint64_t> storageElems;
    /// Why the range is not known, or what made it coarser.
    std::string why;
  };
  /// The range of \p view inside \p storage (see the .cpp file).
  ViewRange viewRange(const pointsto::Node *view, const pointsto::Node *storage,
                      unsigned depth = 0) const;
  /// The constant value of the range or id operand \p value of \p call
  /// (a 64-bit integer, or component \p index of the range or id record a
  /// pointer points to), if the stored-value queries prove one.
  std::optional<uint64_t> rangeComponent(mlir::Value value, unsigned index,
                                         mlir::Operation *call) const;
  /// The \p dims components of each range or id operand of \p call from
  /// operand \p first on (getSYCLRangeOperands): one entry per range, each
  /// component std::nullopt if not constant. std::nullopt if the operands
  /// do not group into ranges of \p dims components.
  std::optional<
      llvm::SmallVector<llvm::SmallVector<std::optional<uint64_t>, 3>, 2>>
  rangeOperands(mlir::Operation *call, unsigned first, unsigned dims) const;
  /// Why the launches of \p kernel are not all known, or std::nullopt.
  std::optional<std::string> unknownLaunchWhy(llvm::StringRef kernel) const;
  /// The launch records that name \p kernel; \p why is set if they are not
  /// all of its launches or one has no closure the analysis sees.
  llvm::SmallVector<const pointsto::LaunchRecord *, 2>
  launchesOf(llvm::StringRef kernel, std::optional<std::string> &why) const;
  /// Whether the closure of \p launch has every top-level field the
  /// parameters of \p kernels name.
  bool isClosureOf(const pointsto::LaunchRecord &launch,
                   llvm::ArrayRef<cir::FuncOp> kernels) const;
  /// The one constant of type \p type the \p size bytes at \p cell hold
  /// when \p at (the launch reading them) runs, or why not.
  ConstFact constantAt(pointsto::Cell cell, uint64_t size, mlir::Type type,
                       mlir::Operation *at) const;

  std::unique_ptr<pointsto::PointsToModelRegistry> ownedRegistry;
  std::unique_ptr<pointsto::PointsToAnalysis> ownedAnalysis;
  const pointsto::PointsToAnalysis *analysis = nullptr;
  std::unique_ptr<mlir::DataLayout> layout;
  std::unique_ptr<SYCLClosureLayout> closureLayout;
  /// The symbol tables of the host module, for the lookups of the queries.
  mutable mlir::SymbolTableCollection symbols;
  /// Why there are no facts (the analysis gave up), or empty.
  std::string gaveUp;
  /// HostFactsOptions::assumeSpecHostPtrUB.
  bool assumeSpecHostPtrUB = true;
  /// The budget trips last reported (takeBudgetReport).
  mutable std::string reportedBudget;

  llvm::StringMap<llvm::SmallVector<cir::FuncOp, 1>> deviceKernels;
  /// The nodes that hold a pointer to each node (link targets), and the
  /// handles whose `$storage` ghost link reaches each storage node.
  llvm::DenseMap<const pointsto::Node *,
                 llvm::SmallVector<const pointsto::Node *, 2>>
      holders, handles;
  /// The targets of the formals of the functions some caller of which is not
  /// analysed: what code we don't see may pass.
  llvm::DenseMap<const pointsto::Node *, mlir::Operation *> formalTargets;
  /// A launch that may launch any kernel (no kernel name found), if any.
  mlir::Operation *unnamedLaunch = nullptr;
  /// The kernels named by a kernel.info marker in an analysed function that
  /// does not feed a recorded launch: a launch path no model records.
  llvm::StringMap<mlir::Operation *> unrecordedMarkers;
};

/// The SYCLHostFacts of one offload container, built once per set of
/// analysis options (getHostFactsAnalysisKey), and the KernelBindingTable
/// built on them, once per mode and set of options, shared by the container
/// passes that read them (StampReqdWorkGroupSize, InferKernelParamNoAlias,
/// PropagateSYCLKernelArgs): an MLIR analysis of the container
/// (Pass::getAnalysis), which those passes preserve
/// (Pass::markAnalysesPreserved) even when they change the container.
///
/// Why their changes leave the facts and the table true. The facts are about
/// the host module: what the memory a recorded kernel launch reads (its
/// closure and its range) holds, which allocation each captured pointer or
/// accessor refers to, and which launches name each kernel. The table holds
/// the bindings (each kernel's device functions and host name string), the
/// launch sites (the kernel.info marker calls, the launch calls and their
/// closure operands) and those facts. The passes change only device modules:
/// attributes on kernels (llvm.noalias, cir.reqd_work_group_size) and
/// constants substituted for the uses of kernel parameters (a cir.const
/// added at the top of the kernel). None of that is host code, no symbol is
/// added, renamed or erased, and no device function the table holds is
/// erased; the candidates (cir.sycl_kernel_param) are unchanged. So every
/// pointer the analysis and the table hold stays valid and every answer they
/// give stays what it was. Any other pass that changes the container
/// invalidates the cache as usual (it does not preserve it); in the
/// cir-offload-merge pipeline the cache is built after launch-path
/// internalization and dropped by the prune, which runs after the last
/// consumer.
class SYCLHostFactsCache {
public:
  explicit SYCLHostFactsCache(mlir::Operation *container);
  ~SYCLHostFactsCache();
  SYCLHostFactsCache(SYCLHostFactsCache &&) = default;

  /// The facts of the container for the analysis options of \p options (its
  /// mode is not part of the key), built on first use.
  const SYCLHostFacts &get(const HostFactsOptions &options);
  /// The table of the container with the host facts of \p options (mode
  /// included), built on first use (from get(options)).
  const KernelBindingTable &getTable(const HostFactsOptions &options);
  /// The number of analyses built so far (one per key), for tests.
  unsigned getNumBuilt() const { return numBuilt; }
  /// The number of tables built so far (one per key), for tests.
  unsigned getNumTablesBuilt() const { return tables.size(); }

private:
  mlir::ModuleOp container;
  std::map<std::string, std::unique_ptr<SYCLHostFacts>> facts;
  std::map<std::string, std::unique_ptr<KernelBindingTable>> tables;
  unsigned numBuilt = 0;
};

} // namespace cir

#endif // CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_SYCLHOSTFACTS_H
