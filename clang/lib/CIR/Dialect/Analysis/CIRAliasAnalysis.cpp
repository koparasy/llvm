//===- CIRAliasAnalysis.cpp - CIR Alias Analysis Suite --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/CIRAliasAnalysis.h"
#include "clang/CIR/Dialect/Analysis/CIRBasicAliasAnalysis.h"
#include "clang/CIR/Dialect/Analysis/CIRPointsToAliasAnalysis.h"

void cir::registerCIRAliasAnalyses(mlir::AliasAnalysis &aa, mlir::Operation *op,
                                   const pointsto::PointsToAnalysis *pointsTo) {
  aa.addAnalysisImplementation(CIRBasicAliasAnalysis(op));
  // After CIRBasicAliasAnalysis: alias() takes the first answer that is not
  // MayAlias, and the two never contradict each other (see
  // CIRPointsToAliasAnalysis.h).
  if (pointsTo)
    aa.addAnalysisImplementation(CIRPointsToAliasAnalysis(*pointsTo));
}
