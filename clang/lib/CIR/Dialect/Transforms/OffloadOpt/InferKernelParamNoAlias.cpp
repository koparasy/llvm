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
// Two paths. For a TU-local/optimizable binding (every launch visible, closure
// is a lambda) it annotates the shared device kernel in place. For the residual
// -- a cross-TU functor or a by-name-reachable kernel whose shared body another
// TU could launch with aliasing pointers -- it specializes by clone: mint a
// fresh kernel (copyKernel), redirect this TU's launches to it, and annotate
// the clone, leaving the generic original untouched. syclNoAliasFields already
// ANDs the distinctness across every launch this TU sees, so it holds at all
// the redirected sites.
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

  // Stamp llvm.noalias on each kernel parameter bound to a distinct-USM capture
  // field. Returns whether any parameter was newly annotated.
  auto applyNoAlias = [&](llvm::ArrayRef<cir::FuncOp> kernels,
                          const llvm::StringSet<> &fields) {
    bool did = false;
    for (cir::FuncOp kernel : kernels) {
      if (kernel.getCallingConv() != cir::CallingConv::SpirKernel)
        continue;
      if (kernel.getBody().empty())
        continue;
      for (mlir::BlockArgument arg : kernel.getBody().front().getArguments()) {
        if (!mlir::isa<cir::PointerType>(arg.getType()))
          continue;
        // Already noalias (e.g. a restrict-qualified capture): leave it.
        if (kernel.getArgAttr(arg.getArgNumber(), noAliasName))
          continue;
        llvm::StringRef field = closureFieldForArg(arg);
        if (field.empty() || !fields.contains(field))
          continue;
        kernel.setArgAttr(arg.getArgNumber(), noAliasName,
                          mlir::UnitAttr::get(ctx));
        did = true;
      }
    }
    return did;
  };

  unsigned specTag = 0;
  for (const auto &entry : table) {
    const cir::KernelBinding &binding = entry.second;

    // An empty field set means host-side recovery could not prove any pair
    // distinct at the launches this TU sees.
    if (binding.kind != cir::BindingKind::SYCLName ||
        binding.launchSites.empty() || binding.syclNoAliasFields.empty())
      continue;

    if (binding.isTULocalOptimizable) {
      // Every launch is visible and the kernel is TU-local: safe to annotate
      // the shared device kernel in place.
      changed |= applyNoAlias(binding.deviceKernels, binding.syclNoAliasFields);
    } else {
      // Residual: the shared kernel is cross-TU / by-name reachable, so another
      // TU could launch it with aliasing pointers. Specialize by clone --
      // redirect this TU's launches to a fresh kernel and annotate only the
      // clone, leaving the generic original for unseen launches.
      auto spec = cir::specializeSYCLKernelForLaunches(
          container, binding, binding.launchSites,
          "noalias." + std::to_string(specTag++));
      if (spec) {
        applyNoAlias(spec->deviceKernels, binding.syclNoAliasFields);
        changed = true; // The clone + redirect mutated the IR regardless.
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
