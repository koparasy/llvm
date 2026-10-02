//===- CIRInlinerInterface.cpp - CIR dialect inliner interface ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIRInlinerInterface.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace cir;

namespace {
/// DialectInlinerInterface for CIR, used by the generic MLIR inliner and by
/// analysis-only clients that force-inline a targeted call chain on a
/// throwaway clone (the SYCL launch-geometry recovery in the offload combine).
///
/// The `no_inline` #cir.inline_kind marker is a codegen hint and is not
/// consulted; callers decide which calls to inline. Legality is structural:
///   - launch markers stay calls: the offload pipeline rebuilds its kernel
///     binding table from `cu.kernel_name` stub calls, so inlining them away
///     would erase the launch sites;
///   - every `cir.return` must terminate a top-level block of the callee. A
///     return still nested in a `cir.scope` (FlattenCFG has not run) cannot be
///     rewired by the generic inliner, so such bodies are rejected. Flattened
///     multi-block bodies are accepted; their returns become branches to the
///     post-call block.
struct CIRInlinerInterface : public mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable,
                       bool wouldBeCloned) const final {
    if (call && call->hasAttr("cu.kernel_name"))
      return false;
    auto func = mlir::dyn_cast<cir::FuncOp>(callable);
    if (!func || func.isDeclaration())
      return false;
    mlir::Region &body = func.getBody();
    if (body.empty() || body.front().empty())
      return false;
    bool nestedReturn = false;
    func.walk([&](cir::ReturnOp ret) {
      if (ret->getParentRegion() != &body)
        nestedReturn = true;
    });
    return !nestedReturn;
  }

  bool isLegalToInline(mlir::Region *dest, mlir::Region *src, bool wouldBeCloned,
                       mlir::IRMapping &valueMapping) const final {
    return true;
  }
  bool isLegalToInline(mlir::Operation *op, mlir::Region *dest, bool wouldBeCloned,
                       mlir::IRMapping &valueMapping) const final {
    return true;
  }

  // The single-block fast path (InliningUtils.cpp) assumes the sole block's
  // terminator yields exactly the callee's results. A block terminated by
  // cir.unreachable yields none, so take the generic multi-block path instead
  // (mirrors the LLVM dialect).
  bool allowSingleBlockOptimization(
      llvm::iterator_range<mlir::Region::iterator> inlinedBlocks) const final {
    return !(
        !inlinedBlocks.empty() &&
        mlir::isa<cir::UnreachableOp>(inlinedBlocks.begin()->getTerminator()));
  }

  // Multi-block callee: a cir.return becomes a cir.br into the post-call block,
  // forwarding the returned values. Other terminators (cir.unreachable,
  // cir.trap) are left in place.
  void handleTerminator(mlir::Operation *op, mlir::Block *newDest) const final {
    auto ret = mlir::dyn_cast<cir::ReturnOp>(op);
    if (!ret)
      return;
    mlir::OpBuilder builder(op);
    cir::BrOp::create(builder, op->getLoc(), newDest, ret.getInput());
    op->erase();
  }

  // Single-block callee: the cir.return operands replace the call results.
  void handleTerminator(mlir::Operation *op,
                        mlir::ValueRange valuesToReplace) const final {
    auto ret = mlir::cast<cir::ReturnOp>(op);
    for (auto [oldVal, newVal] : llvm::zip(valuesToReplace, ret.getInput()))
      oldVal.replaceAllUsesWith(newVal);
  }
};
} // namespace

void cir::registerInlinerInterface(mlir::DialectRegistry &registry) {
  // Both the LLVM and func dialects promise DialectInlinerInterface; their
  // implementations are shipped as dialect extensions that must be registered
  // explicitly, and the generic inliner asserts on a promised-but-missing
  // interface as soon as a module mentions those dialects' callables.
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::func::registerInlinerExtension(registry);
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, cir::CIRDialect *dialect) {
        dialect->addInterfaces<CIRInlinerInterface>();
      });
}
