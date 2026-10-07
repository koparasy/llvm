//===- CIRPointsToAliasAnalysis.cpp - Points-to alias analysis ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/CIRPointsToAliasAnalysis.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToAnalysis.h"

using namespace cir;

mlir::AliasResult CIRPointsToAliasAnalysis::alias(mlir::Value lhs,
                                                  mlir::Value rhs) {
  return analysis->alias(lhs, rhs);
}

mlir::ModRefResult CIRPointsToAliasAnalysis::getModRef(mlir::Operation *op,
                                                       mlir::Value location) {
  return analysis->getModRef(op, location);
}
