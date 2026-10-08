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
// host module with the SYCL plug-in (SYCLPointsToModels.h) instead of from
// KernelBindingTable's trace of a collapsed launch clone.
//
// So far it answers the question of InferKernelParamNoAlias: which captured
// USM pointers and accessors of a kernel's closure refer to memory distinct
// from every other one, at every launch of the kernel
// (--cir-offload-host-facts=ipo|both, see HostFactsMode).
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
#include "clang/CIR/Dialect/Analysis/PointsTo/LocalBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

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
                mlir::Operation *deviceScope);
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

  /// The noalias verdict of every candidate field of \p kernel (the
  /// candidates of its device kernels, getSYCLNoAliasCandidates), with the
  /// contract of KernelBindingTable's legacy fold: a field is no-alias iff at
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

  /// Per launch: the identity of each candidate, and the per-launch verdicts
  /// before the AND over the launches (for the test pass).
  struct LaunchFacts {
    llvm::SmallVector<std::pair<std::string, MemIdentity>, 4> fields;
    llvm::StringMap<NoAliasDecision> decisions;
  };
  LaunchFacts factsOfLaunch(const pointsto::LaunchRecord &launch,
                            const SYCLNoAliasCandidates &candidates) const;

  /// Whether two identities refer to the same memory: they share a node.
  static bool sameMemory(const MemIdentity &a, const MemIdentity &b);

private:
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
  /// A short name of the allocation of \p node, for notes.
  std::string describeAllocation(const pointsto::Node *node) const;
  /// Why the launches of \p kernel are not all known, or std::nullopt.
  std::optional<std::string> unknownLaunchWhy(llvm::StringRef kernel) const;

  std::unique_ptr<pointsto::PointsToModelRegistry> ownedRegistry;
  std::unique_ptr<pointsto::PointsToAnalysis> ownedAnalysis;
  const pointsto::PointsToAnalysis *analysis = nullptr;
  std::unique_ptr<mlir::DataLayout> layout;
  std::unique_ptr<SYCLClosureLayout> closureLayout;

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

} // namespace cir

#endif // CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_SYCLHOSTFACTS_H
