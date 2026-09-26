//===- TargetLowering.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the cir-target-lowering pass.
//
//===----------------------------------------------------------------------===//

#include "RecordTypeConverter.h"
#include "TargetLowering/LowerModule.h"
#include "TargetLowering/TargetLoweringInfo.h"

#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/CIRTransformUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_TARGETLOWERING
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

struct TargetLoweringPass
    : public impl::TargetLoweringBase<TargetLoweringPass> {
  TargetLoweringPass() = default;
  void runOnOperation() override;
};

/// Converts LangAddressSpaceAttr → TargetAddressSpaceAttr inside types.
///
/// Only the records that (transitively) contain such a pointer are rebuilt, so
/// that code without language address spaces is left untouched.
class TargetLoweringTypeConverter : public cir::RecordRewritingTypeConverter {
public:
  TargetLoweringTypeConverter(mlir::MLIRContext &ctx,
                              const cir::TargetLoweringInfo &targetInfo)
      : RecordRewritingTypeConverter(ctx), targetInfo(targetInfo) {
    addConversion([this](cir::PointerType type) -> mlir::Type {
      mlir::Type pointee = convertType(type.getPointee());
      if (!pointee)
        return {};
      auto addrSpace = type.getAddrSpace();
      if (auto langAS =
              mlir::dyn_cast_if_present<cir::LangAddressSpaceAttr>(addrSpace))
        addrSpace = lowerAddrSpace(type.getContext(), langAS);
      return cir::PointerType::get(type.getContext(), pointee, addrSpace);
    });
  }

  mlir::ptr::MemorySpaceAttrInterface
  lowerAddrSpace(mlir::MLIRContext *ctx,
                 cir::LangAddressSpaceAttr langAS) const {
    unsigned targetAS =
        targetInfo.getTargetAddrSpaceFromCIRAddrSpace(langAS.getValue());
    if (targetAS == 0)
      return {};
    return cir::TargetAddressSpaceAttr::get(ctx, targetAS);
  }

  /// Convert the types nested in \p attr (e.g. the type of an initializer).
  mlir::Attribute convertAttr(mlir::Attribute attr) const {
    return attrReplacer.replace(attr);
  }

protected:
  bool shouldConvertRecord(cir::RecordType type) override {
    auto it = recordNeedsConversion.find(type);
    if (it != recordNeedsConversion.end())
      return it->second;
    // A complete walk that finds nothing is conclusive, so the result can be
    // cached for the queried record.
    llvm::SmallPtrSet<mlir::Type, 8> visiting;
    bool result = containsLangAddrSpace(type, visiting);
    recordNeedsConversion[type] = result;
    return result;
  }

private:
  /// Whether \p type reaches a pointer in a language address space. Records
  /// are opaque to the generic sub-element walk, so recurse into their members
  /// explicitly, guarding against recursive records with \p visiting.
  bool containsLangAddrSpace(mlir::Type type,
                             llvm::SmallPtrSetImpl<mlir::Type> &visiting) {
    if (auto rt = mlir::dyn_cast<cir::RecordType>(type)) {
      if (rt.getName()) {
        auto it = recordNeedsConversion.find(type);
        if (it != recordNeedsConversion.end())
          return it->second;
        if (!visiting.insert(type).second)
          return false;
      }
      for (mlir::Type member : rt.getMembers())
        if (containsLangAddrSpace(member, visiting))
          return true;
      if (auto u = mlir::dyn_cast<cir::UnionType>(type))
        if (mlir::Type pad = u.getPadding())
          return containsLangAddrSpace(pad, visiting);
      return false;
    }

    bool found = false;
    type.walkImmediateSubElements(
        [&](mlir::Attribute attr) {
          found |= mlir::isa<cir::LangAddressSpaceAttr>(attr);
        },
        [&](mlir::Type sub) {
          found = found || containsLangAddrSpace(sub, visiting);
        });
    return found;
  }

  const cir::TargetLoweringInfo &targetInfo;
  llvm::DenseMap<mlir::Type, bool> recordNeedsConversion;
  mutable mlir::AttrTypeReplacer attrReplacer = [this] {
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement(
        [this](mlir::Type type) -> std::optional<mlir::Type> {
          if (mlir::Type converted = convertType(type))
            return converted;
          return std::nullopt;
        });
    return replacer;
  }();
};

/// A generic target lowering pattern that matches any CIR op whose operand or
/// result types need address space conversion. Clones the op with converted
/// types.
class CIRGenericTargetLoweringPattern : public mlir::ConversionPattern {
public:
  CIRGenericTargetLoweringPattern(
      mlir::MLIRContext *context,
      const TargetLoweringTypeConverter &typeConverter)
      : mlir::ConversionPattern(typeConverter, MatchAnyOpTypeTag(),
                                /*benefit=*/1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Do not match on operations that have dedicated lowering patterns.
    if (llvm::isa<cir::FuncOp, cir::GlobalOp>(op))
      return mlir::failure();

    const mlir::TypeConverter *typeConverter = getTypeConverter();
    assert(typeConverter &&
           "CIRGenericTargetLoweringPattern requires a type converter");
    bool operandsAndResultsLegal = typeConverter->isLegal(op);
    bool regionsLegal =
        std::all_of(op->getRegions().begin(), op->getRegions().end(),
                    [typeConverter](mlir::Region &region) {
                      return typeConverter->isLegal(&region);
                    });
    if (operandsAndResultsLegal && regionsLegal)
      return mlir::failure();

    mlir::OperationState loweredOpState(op->getLoc(), op->getName());
    loweredOpState.addOperands(operands);

    // Preserve auxiliary metadata verbatim.
    loweredOpState.propertiesAttr = op->getPropertiesAsAttribute();
    loweredOpState.addAttributes(op->getDiscardableAttrDictionary().getValue());

    loweredOpState.addSuccessors(op->getSuccessors());

    llvm::SmallVector<mlir::Type> loweredResultTypes;
    loweredResultTypes.reserve(op->getNumResults());
    for (mlir::Type result : op->getResultTypes())
      loweredResultTypes.push_back(typeConverter->convertType(result));
    loweredOpState.addTypes(loweredResultTypes);

    for (mlir::Region &region : op->getRegions()) {
      mlir::Region *loweredRegion = loweredOpState.addRegion();
      rewriter.inlineRegionBefore(region, *loweredRegion, loweredRegion->end());
      if (mlir::failed(
              rewriter.convertRegionTypes(loweredRegion, *getTypeConverter())))
        return mlir::failure();
    }

    mlir::Operation *loweredOp = rewriter.create(loweredOpState);
    // Convert the types nested in inherent attributes, so that semantics such
    // as AllocaOp's allocaType or a constant's value stay in sync with the
    // converted result types.
    const auto *tlTypeConverter =
        static_cast<const TargetLoweringTypeConverter *>(typeConverter);
    loweredOp->getName().walkInherentAttrs(
        loweredOp, [&](llvm::StringRef, mlir::Attribute &attr) {
          attr = tlTypeConverter->convertAttr(attr);
        });
    rewriter.replaceOp(op, loweredOp);
    return mlir::success();
  }
};

/// Pattern to lower GlobalOp address space attributes. GlobalOp carries
/// addr_space as a standalone attribute (not inside a type), so the
/// TypeConverter won't reach it automatically.
class CIRGlobalOpTargetLowering
    : public mlir::OpConversionPattern<cir::GlobalOp> {
  const cir::TargetLoweringInfo &targetInfo;

public:
  CIRGlobalOpTargetLowering(mlir::MLIRContext *context,
                            const TargetLoweringTypeConverter &typeConverter,
                            const cir::TargetLoweringInfo &targetInfo)
      : mlir::OpConversionPattern<cir::GlobalOp>(typeConverter, context,
                                                 /*benefit=*/1),
        targetInfo(targetInfo) {}

  mlir::LogicalResult
  matchAndRewrite(cir::GlobalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type loweredSymTy = getTypeConverter()->convertType(op.getSymType());
    if (!loweredSymTy)
      return mlir::failure();

    // Convert the addr_space attribute.
    mlir::ptr::MemorySpaceAttrInterface addrSpace = op.getAddrSpaceAttr();
    if (auto langAS =
            mlir::dyn_cast_if_present<cir::LangAddressSpaceAttr>(addrSpace)) {
      unsigned targetAS =
          targetInfo.getTargetAddrSpaceFromCIRAddrSpace(langAS.getValue());
      addrSpace =
          targetAS == 0
              ? nullptr
              : cir::TargetAddressSpaceAttr::get(op.getContext(), targetAS);
    }

    // Only rewrite if something actually changed.
    if (loweredSymTy == op.getSymType() && addrSpace == op.getAddrSpaceAttr())
      return mlir::failure();

    auto newOp = mlir::cast<cir::GlobalOp>(rewriter.clone(*op.getOperation()));
    newOp.setSymType(loweredSymTy);
    newOp.setAddrSpaceAttr(addrSpace);
    const auto *tlTypeConverter =
        static_cast<const TargetLoweringTypeConverter *>(getTypeConverter());
    if (mlir::Attribute init = newOp.getInitialValueAttr())
      newOp.setInitialValueAttr(tlTypeConverter->convertAttr(init));
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

/// Pattern to lower FuncOp types that contain address spaces.
class CIRFuncOpTargetLowering : public mlir::OpConversionPattern<cir::FuncOp> {
public:
  using mlir::OpConversionPattern<cir::FuncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::FuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    cir::FuncType opFuncType = op.getFunctionType();
    mlir::TypeConverter::SignatureConversion signatureConversion(
        opFuncType.getNumInputs());

    for (const auto &[i, argType] : llvm::enumerate(opFuncType.getInputs())) {
      mlir::Type loweredArgType = getTypeConverter()->convertType(argType);
      if (!loweredArgType)
        return mlir::failure();
      signatureConversion.addInputs(i, loweredArgType);
    }

    mlir::Type loweredReturnType =
        getTypeConverter()->convertType(opFuncType.getReturnType());
    if (!loweredReturnType)
      return mlir::failure();

    auto loweredFuncType = cir::FuncType::get(
        signatureConversion.getConvertedTypes(), loweredReturnType,
        /*isVarArg=*/opFuncType.getVarArg());

    // Nothing changed, skip.
    if (loweredFuncType == opFuncType)
      return mlir::failure();

    cir::FuncOp loweredFuncOp = rewriter.cloneWithoutRegions(op);
    loweredFuncOp.setFunctionType(loweredFuncType);
    rewriter.inlineRegionBefore(op.getBody(), loweredFuncOp.getBody(),
                                loweredFuncOp.end());
    if (mlir::failed(rewriter.convertRegionTypes(&loweredFuncOp.getBody(),
                                                 *getTypeConverter(),
                                                 &signatureConversion)))
      return mlir::failure();

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

} // namespace

static void convertSyncScopeIfPresent(mlir::Operation *op,
                                      cir::LowerModule &lowerModule) {
  auto syncScopeAttr =
      mlir::cast_if_present<cir::SyncScopeKindAttr>(op->getAttr("sync_scope"));
  if (syncScopeAttr) {
    cir::SyncScopeKind convertedSyncScope =
        lowerModule.getTargetLoweringInfo().convertSyncScope(
            syncScopeAttr.getValue());
    op->setAttr("sync_scope", cir::SyncScopeKindAttr::get(op->getContext(),
                                                          convertedSyncScope));
  }
}

static void
populateTargetLoweringConversionTarget(mlir::ConversionTarget &target,
                                       const mlir::TypeConverter &tc) {
  target.addLegalOp<mlir::ModuleOp>();

  target.addDynamicallyLegalDialect<cir::CIRDialect>(
      [&tc](mlir::Operation *op) {
        if (!tc.isLegal(op))
          return false;
        return std::all_of(
            op->getRegions().begin(), op->getRegions().end(),
            [&tc](mlir::Region &region) { return tc.isLegal(&region); });
      });

  target.addDynamicallyLegalOp<cir::FuncOp>(
      [&tc](cir::FuncOp op) { return tc.isLegal(op.getFunctionType()); });

  target.addDynamicallyLegalOp<cir::GlobalOp>([&tc](cir::GlobalOp op) {
    if (!tc.isLegal(op.getSymType()))
      return false;
    return !mlir::isa_and_present<cir::LangAddressSpaceAttr>(
        op.getAddrSpaceAttr());
  });
}

void TargetLoweringPass::runOnOperation() {
  auto mod = mlir::cast<mlir::ModuleOp>(getOperation());
  std::unique_ptr<cir::LowerModule> lowerModule = cir::createLowerModule(mod);
  // If lower module is not available, skip the target lowering pass.
  if (!lowerModule) {
    mod.emitWarning("Cannot create a CIR lower module, skipping the ")
        << getName() << " pass";
    return;
  }

  const auto &targetInfo = lowerModule->getTargetLoweringInfo();

  mod->walk([&](mlir::Operation *op) {
    if (mlir::isa<cir::LoadOp, cir::StoreOp, cir::AtomicXchgOp,
                  cir::AtomicCmpXchgOp, cir::AtomicFetchOp>(op))
      convertSyncScopeIfPresent(op, *lowerModule);
  });

  // Address space conversion: LangAddressSpaceAttr → TargetAddressSpaceAttr.
  TargetLoweringTypeConverter typeConverter(*mod.getContext(), targetInfo);

  mlir::RewritePatternSet patterns(mod.getContext());
  patterns.add<CIRGlobalOpTargetLowering>(mod.getContext(), typeConverter,
                                          targetInfo);
  patterns.add<CIRFuncOpTargetLowering>(typeConverter, mod.getContext());
  patterns.add<CIRGenericTargetLoweringPattern>(mod.getContext(),
                                                typeConverter);

  mlir::ConversionTarget target(*mod.getContext());
  populateTargetLoweringConversionTarget(target, typeConverter);

  llvm::SmallVector<mlir::Operation *> ops;
  ops.push_back(mod);
  cir::collectUnreachable(mod, ops);

  if (failed(mlir::applyPartialConversion(ops, target, std::move(patterns))))
    signalPassFailure();

  typeConverter.restoreRecordTypeNames();

  // Distinct language address spaces can map to the same target address space
  // (e.g. offload_generic and the default address space on NVPTX), which turns
  // casts between them into no-ops.
  mod->walk([](cir::CastOp castOp) {
    if (castOp.getKind() == cir::CastKind::address_space &&
        castOp.getSrc().getType() == castOp.getType()) {
      castOp.replaceAllUsesWith(castOp.getSrc());
      castOp.erase();
    }
  });
}

std::unique_ptr<Pass> mlir::createTargetLoweringPass() {
  return std::make_unique<TargetLoweringPass>();
}
