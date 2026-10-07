//===- PointsToVerifier.h - Self-check of points-to graphs ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Checks the invariants a finished FunctionGraph must satisfy, so that a
// value or call the builder forgot shows up as an error instead of as a
// silently precise ("points to nothing") answer.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_POINTSTOVERIFIER_H
#define CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_POINTSTOVERIFIER_H

#include "mlir/Support/LLVM.h"

namespace cir::pointsto {

class FunctionGraph;

/// Check \p fg and emit an error on the offending op (or block argument) for
/// each violation:
///  - a pointer-typed SSA value of the function, other than a null pointer
///    constant, has no cell;
///  - an aggregate SSA value whose type holds pointers has no slot record, or
///    one whose offsets don't match its type;
///  - a cir.call or cir.try_call does not have exactly one CallSiteRecord;
///  - Graph::verify() fails.
mlir::LogicalResult verifyFunctionGraph(const FunctionGraph &fg);

/// Whether -cir-points-to-verify is on: LocalBuilder::build() then verifies
/// every graph it builds.
bool isPointsToVerifyEnabled();

} // namespace cir::pointsto

#endif // CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_POINTSTOVERIFIER_H
