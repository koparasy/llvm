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
//  - `$host` on a storage node: the host memory the runtime may use for it
//    (buffer(hostData, ...)).
//  - `$kernel` on a detail::HostKernel(Ref) object: the kernel function
//    object it wraps.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_SYCLPOINTSTOMODELS_H
#define CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_SYCLPOINTSTOMODELS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToGraph.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace cir::pointsto {
class PointsToModelRegistry;
} // namespace cir::pointsto

namespace cir {

/// The ghost field a buffer or accessor handle's storage is reached through.
pointsto::GhostField getSYCLStorageField();
/// The ghost field a storage node's host memory is reached through.
pointsto::GhostField getSYCLHostField();
/// The ghost field a HostKernel(Ref) object's kernel object is reached
/// through.
pointsto::GhostField getSYCLKernelField();

/// Register the SYCL host API models (see the file comment) by role.
void registerSYCLPointsToModels(pointsto::PointsToModelRegistry &registry);

/// The kernel name a `kernel.info` marker function names: the string its
/// CompileTimeKernelInfo record's first member (a string_view) points to,
/// found structurally from the globals its body reads. Empty if none.
std::string getSYCLKernelInfoName(cir::FuncOp kernelInfo);

/// The byte offset of the member \p path (one member name per nesting level)
/// in the record \p closure, from the names CIRGen gives the
/// `cir.get_member` ops on it in \p module. std::nullopt if a name is not
/// found or a member is not a record where the path goes on.
std::optional<int64_t>
getSYCLClosureFieldOffset(mlir::ModuleOp module, mlir::Type closure,
                          mlir::ArrayAttr path, const mlir::DataLayout &layout);

} // namespace cir

#endif // CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_SYCLPOINTSTOMODELS_H
