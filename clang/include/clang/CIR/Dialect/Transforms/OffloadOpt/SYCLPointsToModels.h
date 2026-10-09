//===- SYCLPointsToModels.h - SYCL models for CIR points-to -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The SYCL plug-in of the CIR points-to analysis: models of the SYCL host API
// functions CIRGen tags with a `cir.sycl_host_api` role, keyed only on the
// role. They describe buffer and accessor storage through ghost links, run
// the command-group function object a queue submission hands to the runtime,
// and record the kernel launches.
//
// Ghost fields:
//  - `$storage` on a buffer or accessor handle (at the handle's offset): the
//    storage object the handle refers to, a Ghost node made by the
//    constructor that owns the storage. Copies share it (reference
//    semantics). A handle without a `$storage` ghost link, or whose storage
//    node has no origin, was not built by a model: its storage is unknown.
//  - `$view` on a buffer or accessor handle: the part of its storage the
//    handle gives access to, a Ghost node made by the call that chose it
//    (SYCLHostFacts reads the range from that call's operands). A buffer
//    constructor that owns its storage makes a view of all of it; a
//    sub-buffer constructor, a ranged accessor constructor or
//    buffer::get_access with a range makes a view whose own `$view` is the
//    view of the buffer it was made from; reinterpret makes a view whose
//    range is not known (the element type changes); copies share it. Every
//    model that sets a handle's `$storage` sets its `$view`, so a handle
//    unified with another has the views of both.
//  - `$host` on a storage node: the host memory the runtime may use for it
//    (buffer(hostData, ...)).
//  - `$final` on a storage node: the memory buffer::set_final_data(T *) made
//    the destructor's write-back destination.
//  - `$kernel` on a detail::HostKernel(Ref) object: the kernel function
//    object it wraps.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_SYCLPOINTSTOMODELS_H
#define CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_SYCLPOINTSTOMODELS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToGraph.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace cir::pointsto {
class PointsToAnalysis;
class PointsToModelRegistry;
struct PointsToAnalysisOptions;
} // namespace cir::pointsto

namespace cir {

/// The ghost field a buffer or accessor handle's storage is reached through.
pointsto::GhostField getSYCLStorageField();
/// The ghost field a buffer or accessor handle's view of its storage is
/// reached through (see the file comment).
pointsto::GhostField getSYCLViewField();
/// The ghost field a storage node's host memory is reached through.
pointsto::GhostField getSYCLHostField();

/// The operands of a SYCL host API call, from operand \p first on, that pass
/// a sycl::range or sycl::id by value, in order: 64-bit integers (a range<D>
/// or id<D> the ABI coerced to D integers) and pointers to a range or id
/// record (one passed indirectly, or by reference). The other operands (the
/// handler, property lists, code locations) are pointers to other records,
/// and empty tag classes are dropped. A ranged accessor constructor and
/// buffer::get_access(handler &, range, id) give the access range and then
/// the access offset; the sub-buffer constructor &baseIndex and &subRange.
llvm::SmallVector<mlir::Value, 6> getSYCLRangeOperands(mlir::Operation *call,
                                                       unsigned first);
/// The ghost field a storage node's set_final_data destination is reached
/// through.
pointsto::GhostField getSYCLFinalField();
/// The ghost field a HostKernel(Ref) object's kernel object is reached
/// through.
pointsto::GhostField getSYCLKernelField();

/// Register the SYCL host API models (see the file comment) by role.
void registerSYCLPointsToModels(pointsto::PointsToModelRegistry &registry);

/// The device-write rule, a rule of the points-to fixpoint
/// (PointsToAnalysisOptions::extraFixpointRule). A kernel is code we don't
/// see that runs on what its closure reaches: it may store any pointer value
/// it can read into any memory it can write, USM memory or buffer storage,
/// where the host may load it later (a memcpy back, a host accessor, the
/// write-back to a buffer's host memory). So at each recorded launch,
/// everything the closure's pointer slots and `$storage` ghost links reach
/// becomes External, as the arguments of an unknown call do: the objects
/// themselves (the device may store pointers to them) and their contents,
/// transitively. The closure object is not: the kernel gets a copy of it.
/// Nothing is merged and no origin is added, so the identity of each
/// allocation stays known: device code cannot create host-visible
/// allocations. Returns whether a node got the flag.
bool applySYCLDeviceWrites(pointsto::PointsToAnalysis &analysis);

/// Add the SYCL plug-in to \p options: the host API models
/// (registerSYCLPointsToModels, through extraModels) and the device-write
/// rule (applySYCLDeviceWrites, through extraFixpointRule).
void addSYCLPointsToPlugin(pointsto::PointsToAnalysisOptions &options);

/// The kernel name a `kernel.info` marker function names: the string its
/// CompileTimeKernelInfo record's first member (a string_view) points to,
/// found structurally from the globals its body reads. Empty if none.
/// \p symbols, if given, are the symbol tables to look the globals up in.
std::string
getSYCLKernelInfoName(cir::FuncOp kernelInfo,
                      mlir::SymbolTableCollection *symbols = nullptr);

/// The byte offset of the member \p path (one member name per nesting level)
/// in the record \p closure, from the names CIRGen gives the
/// `cir.get_member` ops on it in \p module. std::nullopt if a name is not
/// found or a member is not a record where the path goes on.
std::optional<int64_t>
getSYCLClosureFieldOffset(mlir::ModuleOp module, mlir::Type closure,
                          mlir::ArrayAttr path, const mlir::DataLayout &layout);

/// getSYCLClosureFieldOffset for many queries on one module: the member
/// names are read once.
class SYCLClosureLayout {
public:
  explicit SYCLClosureLayout(mlir::ModuleOp module);

  /// The byte offset and the type of the member \p path of \p closure.
  std::optional<std::pair<int64_t, mlir::Type>>
  getField(mlir::Type closure, mlir::ArrayAttr path,
           const mlir::DataLayout &layout) const;

private:
  llvm::DenseMap<std::pair<mlir::Type, mlir::StringAttr>, unsigned> members;
};

} // namespace cir

#endif // CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_SYCLPOINTSTOMODELS_H
