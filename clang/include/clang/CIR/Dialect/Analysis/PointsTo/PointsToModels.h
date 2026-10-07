//===- PointsToModels.h - Library models for CIR points-to ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the library models of the CIR points-to analysis. A
// model is a small transfer function that gives a call to a function the
// analysis cannot see (a declaration), or chooses not to look into, its
// effect on the caller's graph: the DSA "local graph of a declaration".
//
// Models are kept in a PointsToModelRegistry and looked up, in this order:
//
//  1. by the role the callee declares in its `cir.sycl_host_api` attribute
//     (no role models are registered yet; the SYCL plug-in adds them);
//  2. by the callee's symbol name;
//  3. by predicate, in registration order.
//
// The first model whose matches() accepts the call is applied. A call to a
// declaration that no model accepts gets the unknown-call rule (see
// LocalBuilder).
//
// Trust rule (plan D1): a model may only rely on what a standard guarantees
// (the C standard, the C++ standard, the SYCL specification). Each model
// cites the section it relies on in specCitation().
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_POINTSTOMODELS_H
#define CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_POINTSTOMODELS_H

#include "clang/CIR/Dialect/Analysis/PointsTo/LocalBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Interfaces/CIROpInterfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <optional>
#include <vector>

namespace mlir {
class DataLayout;
} // namespace mlir

namespace cir::pointsto {

/// What a model may use and do at a call site: the graph the site's cells
/// live in (a function's local graph, or the TU graph of the interprocedural
/// phase when it resolves an indirect call to a declaration), and where to
/// record deferred effects and copies.
class ModelContext {
public:
  ModelContext(Graph &graph, const mlir::DataLayout &dataLayout,
               llvm::SmallVectorImpl<DeferredEffect> &deferred,
               llvm::SmallVectorImpl<MemTransferRecord> &memTransfers)
      : graph(graph), dataLayout(dataLayout), deferred(deferred),
        memTransfers(memTransfers) {}
  /// The context of a call site of \p fg.
  ModelContext(FunctionGraph &fg, const mlir::DataLayout &dataLayout,
               llvm::SmallVectorImpl<DeferredEffect> &deferred)
      : ModelContext(fg.getGraph(), dataLayout, deferred, fg.memTransfers) {}

  Graph &getGraph() { return graph; }
  const mlir::DataLayout &getDataLayout() const { return dataLayout; }

  /// The value of argument \p index if it is an integer constant;
  /// std::nullopt otherwise and for a synthetic site.
  std::optional<int64_t> getConstantArg(const CallSiteRecord &site,
                                        unsigned index) const;

  /// The cell of pointer argument \p index, or std::nullopt (null, not a
  /// pointer, out of range).
  static std::optional<Cell> getArg(const CallSiteRecord &site,
                                    unsigned index) {
    return index < site.args.size() ? site.args[index] : std::nullopt;
  }

  /// Call \p fn for the cell of every pointer argument and of every pointer
  /// slot of an aggregate argument.
  static void forEachPointerArg(const CallSiteRecord &site,
                                llvm::function_ref<void(Cell)> fn);

  /// A reason of \p kind blaming the call site.
  const Reason *makeReason(const CallSiteRecord &site, Reason::Kind kind,
                           llvm::StringRef detail = {});

  /// Set Mod and/or Ref on the node of \p cell (nothing for std::nullopt).
  void modRef(std::optional<Cell> cell, bool mod, bool ref);

  /// A fresh Heap node allocated at the call site (origin HeapCall), of
  /// \p size bytes if known. It holds no links: fresh memory holds no
  /// pointers yet.
  Cell makeHeapObject(const CallSiteRecord &site, std::optional<int64_t> size);

  /// Make the pointer result of the call point to \p cell (nothing if the
  /// call has no pointer result).
  void setResult(const CallSiteRecord &site, Cell cell);

  /// Make the result of the call (its pointer, or every pointer slot of an
  /// aggregate result) an UnknownObj node with the status bits \p status.
  void setResultUnknown(const CallSiteRecord &site, NodeFlags status,
                        const Reason *why);

  /// Defer an effect to the end of the walk (see DeferredEffect).
  void defer(DeferredEffect::Kind kind, Cell cell, NodeFlags flags,
             const Reason *why) {
    deferred.push_back({kind, cell, flags, why});
  }

  /// Record a copy of \p length bytes (std::nullopt: unknown) from \p src to
  /// \p dst, like cir.libc.memcpy (see recordMemTransfer).
  void addMemTransfer(const CallSiteRecord &site, Cell dst, Cell src,
                      std::optional<int64_t> length) {
    recordMemTransfer(graph, memTransfers, site.call, dst, src, length);
  }

private:
  Graph &graph;
  const mlir::DataLayout &dataLayout;
  llvm::SmallVectorImpl<DeferredEffect> &deferred;
  llvm::SmallVectorImpl<MemTransferRecord> &memTransfers;
};

/// A library model.
class PointsToModel {
public:
  virtual ~PointsToModel();

  /// Whether the model applies to \p call (null for a synthetic site) whose
  /// callee is \p calleeOrNull (null if the symbol is not a cir.func in the
  /// module). The registry has already matched the role or the symbol name;
  /// this checks the rest (a declaration, the signature).
  virtual bool matches(cir::CIRCallOpInterface call,
                       cir::FuncOp calleeOrNull) const = 0;

  /// Give the call its effect on the graph of \p ctx. \p site holds the
  /// cells of the arguments and the result node or slots; `site.callee` is
  /// the modelled function, also when the call is indirect.
  virtual void apply(CallSiteRecord &site, ModelContext &ctx) const = 0;

  /// A short name, printed on the call site (`model=<name>`).
  virtual llvm::StringRef name() const = 0;

  /// The section of the standard the model relies on.
  virtual llvm::StringRef specCitation() const = 0;
};

/// The models, indexed by role, symbol name and predicate.
class PointsToModelRegistry {
public:
  PointsToModelRegistry();
  ~PointsToModelRegistry();
  PointsToModelRegistry(const PointsToModelRegistry &) = delete;
  PointsToModelRegistry &operator=(const PointsToModelRegistry &) = delete;

  /// Register \p model for the callees whose `cir.sycl_host_api` role is
  /// \p role.
  const PointsToModel *addRoleModel(llvm::StringRef role,
                                    std::unique_ptr<PointsToModel> model);
  /// Register \p model for the callees named by any of \p symbols.
  const PointsToModel *addSymbolModel(llvm::ArrayRef<llvm::StringRef> symbols,
                                      std::unique_ptr<PointsToModel> model);
  /// Register \p model to be asked for every call no role or symbol model
  /// took, after the predicate models registered before it.
  const PointsToModel *addPredicateModel(std::unique_ptr<PointsToModel> model);

  /// The model for \p call (null for a synthetic site) to the function named
  /// \p calleeName, which is \p calleeOrNull in the module; null if none
  /// applies.
  const PointsToModel *lookup(cir::CIRCallOpInterface call,
                              cir::FuncOp calleeOrNull,
                              llvm::StringRef calleeName) const;

  /// Every registered model, in registration order.
  llvm::ArrayRef<std::unique_ptr<PointsToModel>> getModels() const {
    return models;
  }

private:
  std::vector<std::unique_ptr<PointsToModel>> models;
  llvm::StringMap<llvm::SmallVector<const PointsToModel *, 1>> byRole;
  llvm::StringMap<llvm::SmallVector<const PointsToModel *, 1>> bySymbol;
  llvm::SmallVector<const PointsToModel *, 2> predicates;
};

/// Register the generic C and C++ models: allocators, deallocators, the C
/// string and memory functions, functions without a pointer effect (the
/// C++ ABI runtime, std::__throw_*, abort, exit, the printf family), and
/// declarations carrying #cir.memory_effects.
void registerGenericModels(PointsToModelRegistry &registry);

} // namespace cir::pointsto

#endif // CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_POINTSTOMODELS_H
