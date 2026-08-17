//===- InferKernelParamNoAlias.cpp - SYCL noalias from USM allocations ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Infers `llvm.noalias` on SYCL device kernel pointer parameters when the host
// passes each from a DISTINCT USM allocation.
//
// A spir_kernel pointer parameter is bound to a closure capture field (the
// prologue casts the address_space(1) argument and stores it into a
// cir.get_member of the closure record). KernelBindingTable already resolved,
// on the host side, which capture fields are backed at every launch site by
// pairwise-distinct USM allocations (malloc_device / malloc_shared /
// malloc_host) -- see KernelBinding::syclNoAliasFields. This pass maps each
// device parameter to its field and stamps `llvm.noalias` on the parameters
// whose field is in that set.
//
// In-place only: it annotates the shared device kernel and never clones it, so
// it is limited to TU-local/optimizable bindings whose every launch is visible.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_OFFLOADINFERKERNELPARAMNOALIAS
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace cir;

namespace {

// The single closure capture field a spir_kernel parameter is stored into, or
// empty if it reaches no field or more than one. The prologue routes the
// address_space(1) argument through offset-0 reinterprets (an address_space or
// bitcast cast) -- and, before mem2reg, a spill slot -- into a cir.get_member
// store; walk that forward. A by-value struct argument (copied via cir.copy)
// or a local_accessor argument (initialized via __init) never matches, so only
// USM pointer captures are mapped.
static llvm::StringRef closureFieldForArg(mlir::Value arg) {
  llvm::StringRef field;
  llvm::SmallVector<mlir::Value, 4> worklist{arg};
  llvm::SmallPtrSet<mlir::Value, 8> seen{arg};
  while (!worklist.empty()) {
    mlir::Value v = worklist.pop_back_val();
    for (mlir::Operation *user : v.getUsers()) {
      if (auto st = mlir::dyn_cast<cir::StoreOp>(user)) {
        if (st.getValue() != v)
          continue;
        if (auto gm = st.getAddr().getDefiningOp<cir::GetMemberOp>()) {
          if (!field.empty() && field != gm.getName())
            return {}; // Two distinct fields: ambiguous, give up.
          field = gm.getName();
          continue;
        }
        // A spill into a stack slot: follow the reloads of that slot.
        for (mlir::Operation *slotUser : st.getAddr().getUsers())
          if (auto load = mlir::dyn_cast<cir::LoadOp>(slotUser))
            if (seen.insert(load.getResult()).second)
              worklist.push_back(load.getResult());
        continue;
      }
      if (auto cast = mlir::dyn_cast<cir::CastOp>(user)) {
        if ((cast.getKind() == cir::CastKind::address_space ||
             cast.getKind() == cir::CastKind::bitcast) &&
            seen.insert(cast.getResult()).second)
          worklist.push_back(cast.getResult());
      }
    }
  }
  return field;
}

struct OffloadInferKernelParamNoAliasPass
    : public impl::OffloadInferKernelParamNoAliasBase<
          OffloadInferKernelParamNoAliasPass> {
  void runOnOperation() override;
};

void OffloadInferKernelParamNoAliasPass::runOnOperation() {
  mlir::ModuleOp container = getOperation();
  if (!cir::isOffloadContainer(container))
    return;
  cir::KernelBindingTable &table = getAnalysis<cir::KernelBindingTable>();
  mlir::MLIRContext *ctx = &getContext();
  llvm::StringRef noAliasName = mlir::LLVM::LLVMDialect::getNoAliasAttrName();
  bool changed = false;

  for (const auto &entry : table) {
    const cir::KernelBinding &binding = entry.second;

    // Only SYCL, TU-local kernels whose every launch this TU can see: another
    // TU could launch a cross-TU kernel with aliasing pointers. An empty field
    // set means host-side recovery could not prove any pair distinct.
    if (!binding.isTULocalOptimizable || binding.launchSites.empty())
      continue;
    if (binding.syclNoAliasFields.empty())
      continue;

    for (cir::FuncOp kernel : binding.deviceKernels) {
      if (kernel.getCallingConv() != cir::CallingConv::SpirKernel)
        continue;
      if (kernel.getBody().empty())
        continue;

      for (mlir::BlockArgument arg : kernel.getBody().front().getArguments()) {
        auto ptrType = mlir::dyn_cast<cir::PointerType>(arg.getType());
        if (!ptrType)
          continue;
        // Already noalias (e.g. a restrict-qualified capture): leave it.
        if (kernel.getArgAttr(arg.getArgNumber(), noAliasName))
          continue;
        llvm::StringRef field = closureFieldForArg(arg);
        if (field.empty() || !binding.syclNoAliasFields.contains(field))
          continue;
        kernel.setArgAttr(arg.getArgNumber(), noAliasName,
                          mlir::UnitAttr::get(ctx));
        changed = true;
      }
    }
  }

  if (!changed)
    markAllAnalysesPreserved();
}

} // namespace

std::unique_ptr<Pass> mlir::createOffloadInferKernelParamNoAliasPass() {
  return std::make_unique<OffloadInferKernelParamNoAliasPass>();
}
