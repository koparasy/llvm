//===- CIRPointerDecomposition.cpp - CIR pointer base/offset --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/CIRPointerDecomposition.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "cir-pointer-decomposition"

using namespace llvm;
using namespace cir;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

std::optional<int64_t>
cir::getTypeSizeInBytes(mlir::Type type, const mlir::DataLayout &dataLayout) {
  if (!cir::isSized(type))
    return std::nullopt;

  llvm::TypeSize size = dataLayout.getTypeSize(type);
  if (size.isScalable())
    return std::nullopt;
  return size.getFixedValue();
}

std::optional<int64_t> cir::getConstantIndex(mlir::Value val) {
  auto constOp =
      mlir::dyn_cast_if_present<cir::ConstantOp>(val.getDefiningOp());
  if (!constOp)
    return std::nullopt;

  auto intAttr = mlir::dyn_cast<cir::IntAttr>(constOp.getValue());
  if (!intAttr)
    return std::nullopt;

  const APInt &value = intAttr.getValue();
  if (intAttr.isSigned())
    return value.trySExtValue();
  return value.tryZExtValue();
}

/// Return `count * size`, or std::nullopt if either input is unknown or the
/// product overflows.
static std::optional<int64_t> scaleOffset(std::optional<int64_t> count,
                                          std::optional<int64_t> size) {
  if (!count || !size)
    return std::nullopt;
  auto [product, overflow] = MulOverflow(*count, *size);
  if (overflow)
    return std::nullopt;
  return product;
}

/// Add \p delta bytes to \p offset, making the offset unknown if \p delta is
/// unknown or if the sum overflows.
static void addToOffset(std::optional<int64_t> &offset,
                        std::optional<int64_t> delta) {
  if (!offset)
    return;
  if (!delta) {
    offset.reset();
    return;
  }
  auto [sum, overflow] = AddOverflow(*offset, *delta);
  if (overflow)
    offset.reset();
  else
    offset = sum;
}

//===----------------------------------------------------------------------===//
// decomposePointer
//===----------------------------------------------------------------------===//

PointerDecomposition cir::decomposePointer(mlir::Value ptr,
                                           const mlir::DataLayout &dataLayout,
                                           unsigned maxDepth) {
  LDBG() << "Decomposing pointer: " << ptr;

  std::optional<int64_t> offset = 0;

  for (unsigned depth = 0; depth < maxDepth; ++depth) {
    mlir::Operation *defOp = ptr.getDefiningOp();
    if (!defOp) {
      LDBG() << "No defining operation, stopping";
      break; // Block argument (e.g. function parameter) — stop here.
    }

    // Bitcasts and address-space casts don't change the address, and
    // array_to_ptrdecay produces a pointer to the first element of the array.
    if (auto castOp = mlir::dyn_cast<cir::CastOp>(defOp)) {
      if (castOp.isAllocaPreservingCast() ||
          castOp.getKind() == cir::CastKind::array_to_ptrdecay) {
        LDBG() << "Walking past cast operation";
        ptr = castOp.getSrc();
        continue;
      }
      LDBG() << "Opaque cast operation, stopping";
      break;
    }

    // A stride moves the pointer by `stride * sizeof(pointee)` bytes.
    if (auto strideOp = mlir::dyn_cast<cir::PtrStrideOp>(defOp)) {
      LDBG() << "Walking past PtrStrideOp";
      addToOffset(offset,
                  scaleOffset(getConstantIndex(strideOp.getStride()),
                              getTypeSizeInBytes(strideOp.getElementType(),
                                                 dataLayout)));
      ptr = strideOp.getBase();
      continue;
    }

    // A record member sits at a fixed offset given by the record layout.
    if (auto memberOp = mlir::dyn_cast<cir::GetMemberOp>(defOp)) {
      LDBG() << "Walking past GetMemberOp";
      auto recordTy =
          mlir::cast<cir::RecordType>(memberOp.getAddrTy().getPointee());
      std::optional<int64_t> memberOffset;
      if (!recordTy.isIncomplete())
        memberOffset =
            recordTy.getElementOffset(dataLayout, memberOp.getIndex());
      addToOffset(offset, memberOffset);
      ptr = memberOp.getAddr();
      continue;
    }

    // An array element sits at `index * sizeof(element)` bytes into the array.
    if (auto elementOp = mlir::dyn_cast<cir::GetElementOp>(defOp)) {
      LDBG() << "Walking past GetElementOp";
      addToOffset(offset,
                  scaleOffset(getConstantIndex(elementOp.getIndex()),
                              getTypeSizeInBytes(elementOp.getElementType(),
                                                 dataLayout)));
      ptr = elementOp.getBase();
      continue;
    }

    // A base class subobject starts the given number of bytes into the derived
    // object. This may return null if the input is null, but accessing memory
    // based on that null pointer would be UB, so we always assume non-null
    // here.
    if (auto baseOp = mlir::dyn_cast<cir::BaseClassAddrOp>(defOp)) {
      LDBG() << "Walking past BaseClassAddrOp";
      addToOffset(offset, baseOp.getOffset().tryZExtValue());
      ptr = baseOp.getSrcAddr();
      continue;
    }

    // Conversely, the derived object starts that many bytes before the base
    // subobject, so the offset is applied as a negative adjustment. This may
    // return null if the input is null, but accessing memory based on that null
    // pointer would be UB, so we always assume non-null here.
    if (auto derivedOp = mlir::dyn_cast<cir::DerivedClassAddrOp>(defOp)) {
      LDBG() << "Walking past DerivedClassAddrOp";
      std::optional<int64_t> baseOffset = derivedOp.getOffset().tryZExtValue();
      if (baseOffset)
        baseOffset = -*baseOffset;
      addToOffset(offset, baseOffset);
      ptr = derivedOp.getSrcAddr();
      continue;
    }

    // The real part of a complex value is at offset zero, the imaginary part
    // right behind it.
    if (auto realOp = mlir::dyn_cast<cir::ComplexRealPtrOp>(defOp)) {
      LDBG() << "Walking past ComplexRealPtrOp";
      ptr = realOp.getOperand();
      continue;
    }
    if (auto imagOp = mlir::dyn_cast<cir::ComplexImagPtrOp>(defOp)) {
      LDBG() << "Walking past ComplexImagPtrOp";
      auto ptrTy = mlir::cast<cir::PointerType>(imagOp.getOperand().getType());
      auto complexTy = mlir::cast<cir::ComplexType>(ptrTy.getPointee());
      addToOffset(offset,
                  getTypeSizeInBytes(complexTy.getElementType(), dataLayout));
      ptr = imagOp.getOperand();
      continue;
    }

    LDBG() << "Unhandled operation, stopping";
    break; // Not expressible as an offset from another pointer.
  }

  return {ptr, offset};
}
