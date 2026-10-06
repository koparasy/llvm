//===- SYCLKernelParams.cpp - SYCL kernel parameter mapping ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLKernelParams.h"

using namespace cir;

std::optional<SYCLKernelParam> cir::getSYCLKernelParam(cir::FuncOp kernel,
                                                       unsigned argNo) {
  if (argNo >= kernel.getNumArguments())
    return std::nullopt;
  auto dict = kernel.getArgAttrOfType<mlir::DictionaryAttr>(
      argNo, cir::CIRDialect::getSYCLKernelParamAttrName());
  if (!dict)
    return std::nullopt;
  auto kind = dict.getAs<mlir::StringAttr>("kind");
  auto field = dict.getAs<mlir::StringAttr>("field");
  auto path = dict.getAs<mlir::ArrayAttr>("path");
  if (!kind || !field || !path || path.empty())
    return std::nullopt;
  SYCLKernelParam param;
  param.kind = kind.getValue();
  param.field = field.getValue();
  param.path = path;
  if (auto initArg = dict.getAs<mlir::IntegerAttr>("init_arg"))
    param.initArg = initArg.getInt();
  if (auto part = dict.getAs<mlir::StringAttr>("accessor_part"))
    param.accessorPart = part.getValue();
  return param;
}

llvm::StringRef cir::getSYCLCapturedField(cir::FuncOp kernel, unsigned argNo,
                                          bool wantPointer) {
  std::optional<SYCLKernelParam> param = getSYCLKernelParam(kernel, argNo);
  if (!param || !param->isTopLevel())
    return {};
  if (param->kind != (wantPointer ? "pointer" : "scalar"))
    return {};
  return param->field;
}

mlir::Value cir::getSYCLKernelObject(cir::FuncOp kernel) {
  mlir::Value result;
  kernel.walk([&](cir::AllocaOp alloca) {
    if (!alloca->hasAttr(cir::CIRDialect::getSYCLKernelObjectAttrName()))
      return mlir::WalkResult::advance();
    result = alloca.getResult();
    return mlir::WalkResult::interrupt();
  });
  return result;
}
