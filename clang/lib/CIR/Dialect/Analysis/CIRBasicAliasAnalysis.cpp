//===- CIRBasicAliasAnalysis.cpp - Basic CIR Alias Analysis ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/CIRBasicAliasAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "clang/CIR/Dialect/Analysis/CIRPointerDecomposition.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "cir-basic-alias-analysis"

using namespace llvm;
using namespace cir;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Return the number of bytes an access through \p ptr touches, that is, the
/// size of its pointee type.
static std::optional<int64_t>
getAccessSize(mlir::Value ptr, const mlir::DataLayout &dataLayout) {
  auto ptrTy = mlir::dyn_cast<cir::PointerType>(ptr.getType());
  if (!ptrTy)
    return std::nullopt;
  return getTypeSizeInBytes(ptrTy.getPointee(), dataLayout);
}

/// Return true if \p lhs and \p rhs are provably different objects.
///
/// TODO: Extend to cover global addresses, function arguments with noalias, and
/// heap allocations.
static bool areDistinctObjects(mlir::Value lhs, mlir::Value rhs) {
  // Distinct cir.alloca ops allocate distinct storage.
  return lhs != rhs &&
         mlir::isa_and_nonnull<cir::AllocaOp>(lhs.getDefiningOp()) &&
         mlir::isa_and_nonnull<cir::AllocaOp>(rhs.getDefiningOp());
}

//===----------------------------------------------------------------------===//
// CIRBasicAliasAnalysis
//===----------------------------------------------------------------------===//

mlir::AliasResult CIRBasicAliasAnalysis::alias(mlir::Value lhs,
                                               mlir::Value rhs) {
  LDBG() << "Checking alias between: " << lhs << " and " << rhs;

  if (lhs == rhs) {
    LDBG() << "Trivial alias between identical values";
    return mlir::AliasResult::MustAlias;
  }

  PointerDecomposition lhsPtr = decomposePointer(lhs, dataLayout);
  PointerDecomposition rhsPtr = decomposePointer(rhs, dataLayout);

  if (lhsPtr.base != rhsPtr.base) {
    if (areDistinctObjects(lhsPtr.base, rhsPtr.base)) {
      LDBG() << "No alias between pointers into distinct objects";
      return mlir::AliasResult::NoAlias;
    }
    LDBG() << "Unrelated base objects, may alias";
    return mlir::AliasResult::MayAlias;
  }

  // Both pointers point into the same object, so their offsets can be compared
  // directly.
  if (!lhsPtr.offset || !rhsPtr.offset) {
    LDBG() << "Same object at an unknown offset, may alias";
    return mlir::AliasResult::MayAlias;
  }

  // Equal offsets means both pointers start at exactly the same address, which
  // is all MustAlias claims. How many bytes each access touches doesn't matter.
  if (*lhsPtr.offset == *rhsPtr.offset) {
    LDBG() << "Must alias at the same address within the same object";
    return mlir::AliasResult::MustAlias;
  }

  // The pointers start at different addresses, so whether they overlap depends
  // on how far each access reaches. Without the extent of both accesses there
  // is no way to tell.
  std::optional<int64_t> lhsSize = getAccessSize(lhs, dataLayout);
  std::optional<int64_t> rhsSize = getAccessSize(rhs, dataLayout);
  if (!lhsSize || !rhsSize) {
    LDBG() << "Same object but unknown access size, may alias";
    return mlir::AliasResult::MayAlias;
  }

  auto [lhsEnd, lhsOverflow] = AddOverflow(*lhsPtr.offset, *lhsSize);
  auto [rhsEnd, rhsOverflow] = AddOverflow(*rhsPtr.offset, *rhsSize);
  if (lhsOverflow || rhsOverflow) {
    LDBG() << "Access range overflows, may alias";
    return mlir::AliasResult::MayAlias;
  }

  if (lhsEnd <= *rhsPtr.offset || rhsEnd <= *lhsPtr.offset) {
    LDBG() << "No alias between disjoint ranges of the same object";
    return mlir::AliasResult::NoAlias;
  }

  LDBG() << "Partial alias between overlapping ranges of the same object";
  return mlir::AliasResult::PartialAlias;
}

mlir::ModRefResult CIRBasicAliasAnalysis::getModRef(mlir::Operation *op,
                                                    mlir::Value location) {
  LDBG() << "getModRef: "
         << mlir::OpWithFlags(op, mlir::OpPrintingFlags().skipRegions())
         << " on location " << location;

  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
  if (!effects) {
    LDBG() << "No memory effect interface, returning ModAndRef";
    return mlir::ModRefResult::getModAndRef();
  }

  SmallVector<mlir::MemoryEffects::EffectInstance> effectList;
  effects.getEffects(effectList);

  auto classifyEffect = [location, this](
                            const mlir::MemoryEffects::EffectInstance &effect) {
    if (mlir::isa<mlir::MemoryEffects::Allocate>(effect.getEffect())) {
      LDBG() << "Skipping allocate effect";
      return mlir::ModRefResult::getNoModRef();
    }

    mlir::AliasResult aliasResult = mlir::AliasResult::MayAlias;
    if (mlir::Value affectedLocation = effect.getValue()) {
      LDBG() << "    Checking alias between affected location "
             << affectedLocation << " and query location " << location;
      aliasResult = alias(affectedLocation, location);
      LDBG() << "    Alias result: " << aliasResult;
    } else {
      // An effect on a non-addressable resource cannot affect a
      // pointer-based location.
      if (!effect.getResource()->isAddressable()) {
        LDBG() << "    Effect on non-addressable resource '"
               << effect.getResource()->getName() << "', skipping (NoAlias)";
        aliasResult = mlir::AliasResult::NoAlias;
      } else {
        LDBG() << "    No effect value, assuming MayAlias";
      }
    }

    // If the affected location doesn't alias with the query location,
    // ignore this effect.
    if (aliasResult.isNo()) {
      LDBG() << "No alias with affected location";
      return mlir::ModRefResult::getNoModRef();
    }

    // TODO: Consider whether Free should be NoModRef.
    if (mlir::isa<mlir::MemoryEffects::Free>(effect.getEffect())) {
      LDBG() << "Skipping free effect";
      return mlir::ModRefResult::getModAndRef();
    }

    if (mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect())) {
      LDBG() << "Write effect, adding Mod";
      return mlir::ModRefResult::getMod();
    }

    if (mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect())) {
      LDBG() << "Read effect, adding Ref";
      return mlir::ModRefResult::getRef();
    }

    LDBG() << "Unexpected memory effect: " << effect.getEffect();
    return mlir::ModRefResult::getNoModRef();
  };

  return llvm::accumulate(llvm::map_range(effectList, classifyEffect),
                          mlir::ModRefResult::getNoModRef(),
                          [](mlir::ModRefResult lhs, mlir::ModRefResult rhs) {
                            return lhs.merge(rhs);
                          });
}
