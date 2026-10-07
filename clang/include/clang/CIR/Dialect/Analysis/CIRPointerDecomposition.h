//===- CIRPointerDecomposition.h - CIR pointer base/offset ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares decomposePointer, which traces a CIR pointer back to the
// object it points into and computes its constant byte offset within that
// object, together with the size and constant helpers it is built on.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_ANALYSIS_CIRPOINTERDECOMPOSITION_H
#define CLANG_CIR_DIALECT_ANALYSIS_CIRPOINTERDECOMPOSITION_H

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

#include <cstdint>
#include <optional>

namespace cir {

/// A pointer expressed as a byte offset into the object it points into.
struct PointerDecomposition {
  /// The value the pointer was traced back to. This is an allocation (for
  /// example a cir.alloca), a block argument (such as a function parameter), or
  /// the result of an operation decomposePointer cannot look through: a load,
  /// a call, a global address, an opaque cast, ... When the depth limit stops
  /// the walk early, it is the last pointer reached, which may itself still be
  /// an address computation.
  mlir::Value base;

  /// Byte offset of the pointer from the start of `base`. The offset can be
  /// negative. It is std::nullopt when some step on the way is not a
  /// compile-time constant (a dynamic cir.ptr_stride or cir.get_element index,
  /// a member of an incomplete record, an element of unsized type), or when
  /// the accumulated offset overflows int64_t.
  std::optional<int64_t> offset;
};

/// Trace \p ptr back to the object it points into, accumulating the byte offset
/// of \p ptr from the start of that object.
///
/// Walks through address-preserving casts (bitcast, address_space),
/// array_to_ptrdecay, cir.ptr_stride, cir.get_member, cir.get_element,
/// cir.base_class_addr, cir.derived_class_addr, cir.complex.real_ptr and
/// cir.complex.imag_ptr, following at most \p maxDepth such steps. The walk
/// stops at a block argument, at an allocation, or at any operation whose
/// result cannot be described as an offset from one of its operands. It never
/// looks through memory: a pointer loaded from a slot is its own base.
///
/// Steps contributing an offset that isn't a compile-time constant are still
/// traced through, leaving the offset unknown, since knowing which object a
/// pointer points into is useful even without the offset within it. The
/// returned base and offset always describe \p ptr, even when the depth limit
/// stops the walk early.
PointerDecomposition decomposePointer(mlir::Value ptr,
                                      const mlir::DataLayout &dataLayout,
                                      unsigned maxDepth = 6);

/// Return the size in bytes of \p type, or std::nullopt when that size isn't
/// statically known (void, function types, incomplete records, scalable
/// types, ...).
std::optional<int64_t> getTypeSizeInBytes(mlir::Type type,
                                          const mlir::DataLayout &dataLayout);

/// If \p val is defined by a cir.const integer that fits in an int64_t, return
/// its value. The constant is interpreted according to the signedness of its
/// type. Casts are not looked through.
std::optional<int64_t> getConstantIndex(mlir::Value val);

} // namespace cir

#endif // CLANG_CIR_DIALECT_ANALYSIS_CIRPOINTERDECOMPOSITION_H
