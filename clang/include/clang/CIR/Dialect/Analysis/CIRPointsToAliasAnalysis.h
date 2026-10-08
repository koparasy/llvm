//===- CIRPointsToAliasAnalysis.h - Points-to alias analysis ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines CIRPointsToAliasAnalysis, an mlir::AliasAnalysis
// implementation that answers alias and mod/ref queries from a finished
// cir::pointsto::PointsToAnalysis of the module. Register it after
// CIRBasicAliasAnalysis, for example with registerCIRAliasAnalyses(aa, op,
// &pointsTo): it only refines what the local analyses leave at MayAlias.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_ANALYSIS_CIRPOINTSTOALIASANALYSIS_H
#define CLANG_CIR_DIALECT_ANALYSIS_CIRPOINTSTOALIASANALYSIS_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace cir {
namespace pointsto {
class PointsToAnalysis;
} // namespace pointsto

/// The points-to analysis as an mlir::AliasAnalysis implementation. It holds
/// a reference to \p analysis, which must outlive it and must not change
/// (the IR must not be modified while queries run). In the bottom-up mode
/// (with PointsToAnalysisOptions::topDownQueries, the default) the answers
/// about the values of a covered function are the meet over its calling
/// contexts (PointsToTopDown.cpp).
///
/// Combining with the other implementations: mlir::AliasAnalysis::alias
/// returns the first answer that is not MayAlias, in registration order
/// (LocalAliasAnalysis, CIRBasicAliasAnalysis, then this one), and getModRef
/// intersects every implementation's answer. CIRBasicAliasAnalysis and this
/// analysis cannot disagree with one saying NoAlias and the other MustAlias
/// (or PartialAlias): CIRBasicAliasAnalysis answers MustAlias or
/// PartialAlias only for two pointers with the same base value and constant
/// offsets, which the points-to analysis places in the same node at the
/// same offset difference, where it never answers NoAlias for equal
/// offsets and computes the same ranges otherwise. It answers NoAlias only
/// for distinct allocas, whose nodes, if unified, have two origins, which
/// rules out MustAlias and PartialAlias here, or for disjoint ranges of one
/// base, which are disjoint here too. The top-down answers keep this: an
/// image of a node keeps the offset differences of the cells in it, and
/// distinct allocas of a function have distinct images, or one image with
/// two origins. The test pass
/// -test-cir-alias-analysis{use-points-to=1} asserts this on every pair it
/// prints.
class CIRPointsToAliasAnalysis {
public:
  explicit CIRPointsToAliasAnalysis(const pointsto::PointsToAnalysis &analysis)
      : analysis(&analysis) {}

  mlir::AliasResult alias(mlir::Value lhs, mlir::Value rhs);
  mlir::ModRefResult getModRef(mlir::Operation *op, mlir::Value location);

private:
  const pointsto::PointsToAnalysis *analysis;
};

} // namespace cir

#endif // CLANG_CIR_DIALECT_ANALYSIS_CIRPOINTSTOALIASANALYSIS_H
