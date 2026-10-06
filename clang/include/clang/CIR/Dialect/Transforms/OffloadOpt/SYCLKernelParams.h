//===-- SYCLKernelParams.h - SYCL kernel parameter mapping ----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Readers for the SYCL kernel parameter mapping CIRGen records on device
// spir_kernel functions, so offload passes need not reverse-engineer the
// kernel prologue:
//
//  - `cir.sycl_kernel_param` on a kernel argument names the closure member the
//    argument initializes, as SemaSYCL built it:
//      {kind = "pointer" | "scalar" | "std_layout" | <SYCL special type, e.g.
//               "accessor", "local_accessor", "stream">,
//       field = <top-level closure field>,
//       path = [<member names from the closure; "[i]" for array elements>],
//       init_arg = <position in the special type's __init call>,
//       accessor_part = "ptr" | "access_range" | "mem_range" | "offset"}
//    The last two are present only for special types. An argument without the
//    attribute has no known field.
//  - `cir.sycl_kernel_object` marks the alloca of the kernel's local closure
//    object.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_SYCLKERNELPARAMS_H
#define LLVM_CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_SYCLKERNELPARAMS_H

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace cir {

/// The decoded `cir.sycl_kernel_param` of one kernel argument.
struct SYCLKernelParam {
  llvm::StringRef kind;
  llvm::StringRef field;
  mlir::ArrayAttr path;
  std::optional<unsigned> initArg;
  llvm::StringRef accessorPart;

  /// Whether the argument initializes a direct member of the closure rather
  /// than a member nested inside one.
  bool isTopLevel() const { return path.size() == 1; }
};

/// The mapping of argument `argNo` of `kernel`, if CIRGen recorded one.
std::optional<SYCLKernelParam> getSYCLKernelParam(cir::FuncOp kernel,
                                                  unsigned argNo);

/// The closure field a SYCL kernel argument is copied into whole -- a captured
/// pointer (when `wantPointer`) or a captured scalar -- or empty when the
/// argument has no such direct field.
llvm::StringRef getSYCLCapturedField(cir::FuncOp kernel, unsigned argNo,
                                     bool wantPointer);

/// The alloca of the kernel's local closure object, or null.
mlir::Value getSYCLKernelObject(cir::FuncOp kernel);

} // namespace cir

#endif // LLVM_CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_SYCLKERNELPARAMS_H
