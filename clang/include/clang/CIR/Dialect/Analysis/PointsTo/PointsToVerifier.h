//===- PointsToVerifier.h - Self-check of points-to graphs ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Checks the invariants a finished FunctionGraph, and a finished
// PointsToAnalysis, must satisfy, so that a value or call the builder forgot,
// or a rule the interprocedural phase did not apply to its fixpoint, shows up
// as an error instead of as a silently precise answer.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_POINTSTOVERIFIER_H
#define CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_POINTSTOVERIFIER_H

#include "mlir/Support/LLVM.h"

namespace cir::pointsto {

class FunctionGraph;
class PointsToAnalysis;

/// Check \p fg and emit an error on the offending op (or block argument) for
/// each violation:
///  - a pointer-typed SSA value of the function, other than a null pointer
///    constant, has no cell;
///  - an aggregate SSA value whose type holds pointers has no slot record, or
///    one whose offsets don't match its type;
///  - a cir.call or cir.try_call does not have exactly one CallSiteRecord;
///  - Graph::verify() fails.
mlir::LogicalResult verifyFunctionGraph(const FunctionGraph &fg);

/// Check the TU graph of \p analysis once it is finished, and emit an error
/// on the module (or the offending call) for each violation:
///  - a cell the analysis holds (scalar map, slots, formals, return cells,
///    call sites) does not resolve to a live node;
///  - an External (Incomplete) node links to a node without the flag:
///    status propagation is not closed;
///  - a Summary node has no Reason{SummaryObject};
///  - a call resolved as CallResolution::Indirect has a callee node that is
///    UnknownObj, External or Incomplete, or has an origin that is not a
///    function, or a function that is not among its targets;
///  - a call has clobbersGlobals, but a non-constant global that another TU
///    can name or whose address escaped lacks External, Mod or Ref.
mlir::LogicalResult verifyPointsToAnalysis(const PointsToAnalysis &analysis);

/// Whether -cir-points-to-verify is on: LocalBuilder::build() then verifies
/// every graph it builds.
bool isPointsToVerifyEnabled();

} // namespace cir::pointsto

#endif // CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_POINTSTOVERIFIER_H
