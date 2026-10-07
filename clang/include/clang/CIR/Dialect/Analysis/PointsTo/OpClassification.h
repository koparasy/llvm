//===- OpClassification.h - Points-to class of each CIR op ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Every CIR op is classified for the points-to analysis as exactly one of
// Handled, NoPointerSemantics or Pessimistic. A unit test checks that the
// table covers every op the CIR dialect registers, so a new op cannot be
// ignored silently: it has to be classified first.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_OPCLASSIFICATION_H
#define CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_OPCLASSIFICATION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace cir::pointsto {

enum class OpClass : uint8_t {
  /// LocalBuilder has a transfer rule for the op.
  Handled,
  /// The op can neither create, move, store nor publish a pointer (integer
  /// and floating-point arithmetic, comparisons, markers, ...). The builder
  /// skips it.
  NoPointerSemantics,
  /// The op matters but is not modelled yet. Its pointer operands' nodes
  /// become Collapsed and External, and its pointer results point to fresh
  /// unknown objects.
  Pessimistic,
};

llvm::StringRef getOpClassName(OpClass cls);

struct OpClassEntry {
  /// The full op name, e.g. "cir.load".
  llvm::StringRef name;
  OpClass cls;
};

/// The whole table, in source order (grouped by class, sorted by name within
/// each group or family).
llvm::ArrayRef<OpClassEntry> getOpClassificationTable();

/// The class of the op called \p opName, or std::nullopt if the table does
/// not list it (an op added to CIR after the table was last updated). Callers
/// must treat an unlisted op as Pessimistic.
std::optional<OpClass> lookupOpClass(llvm::StringRef opName);

/// Whether the LLVM intrinsic \p name (without the "llvm." prefix) is known
/// to have no pointer semantics, so a cir.call_llvm_intrinsic of it can be
/// skipped. Any other intrinsic is treated pessimistically.
bool isPointerFreeLLVMIntrinsic(llvm::StringRef name);

} // namespace cir::pointsto

#endif // CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_OPCLASSIFICATION_H
