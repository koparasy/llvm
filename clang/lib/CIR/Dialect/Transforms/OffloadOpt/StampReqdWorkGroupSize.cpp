//===- StampReqdWorkGroupSize.cpp - SYCL nd_range geometry specialization -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stamps `cir.reqd_work_group_size` on SYCL device kernels from the host-side
// constant nd_range local work-group size recovered by KernelBindingTable.
//
// When every launch of a TU-local kernel uses the same compile-time-constant
// nd_range local size, that size is a required work-group size for the device
// kernel. The attribute records the sizes in natural {X,Y,Z} order (length ==
// number of dims); the SYCL-shaped reversal/padding is applied later at LLVM
// lowering (see LowerToLLVMIR.cpp amendReqdWorkGroupSize).
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir {
#define GEN_PASS_DEF_OFFLOADSTAMPREQDWORKGROUPSIZE
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace cir;

namespace {

// The single compile-time-constant local work-group size shared by every
// launch site of `binding`, or nullopt if the sizes are absent, non-constant,
// or the sites disagree.
//
// Agreement is a property of the whole set of sites, so a plain-range launch
// (empty local), a runtime local size, or two launches with different locals
// all forbid stamping a required work-group size on the kernel.
static std::optional<llvm::SmallVector<int32_t, 3>>
commonConstantLocal(const cir::KernelBinding &binding) {
  llvm::SmallVector<int32_t, 3> common;
  bool haveCommon = false;

  for (const cir::LaunchSite &site : binding.launchSites) {
    const cir::SyclGeometry &geom = site.getSYCLNDRange();
    // A plain `range` launch (RoundedRangeKernel) has no local size, and an
    // invalid geometry recovered nothing: both say "no required size".
    if (!geom.valid || geom.local.empty())
      return std::nullopt;

    llvm::SmallVector<int32_t, 3> cur;
    for (const std::optional<int64_t> &dim : geom.local) {
      if (!dim)
        return std::nullopt; // A runtime local size cannot be stamped.
      cur.push_back(static_cast<int32_t>(*dim));
    }

    if (!haveCommon) {
      common = std::move(cur);
      haveCommon = true;
    } else if (common != cur) {
      return std::nullopt; // Sites disagree.
    }
  }

  if (!haveCommon || common.empty())
    return std::nullopt;
  return common;
}

struct OffloadStampReqdWorkGroupSizePass
    : public impl::OffloadStampReqdWorkGroupSizeBase<
          OffloadStampReqdWorkGroupSizePass> {
  void runOnOperation() override;
};

void OffloadStampReqdWorkGroupSizePass::runOnOperation() {
  mlir::ModuleOp container = getOperation();
  if (!cir::isOffloadContainer(container))
    return;
  cir::KernelBindingTable &table = getAnalysis<cir::KernelBindingTable>();
  mlir::OpBuilder builder(container);
  bool changed = false;

  for (const auto &entry : table) {
    const cir::KernelBinding &binding = entry.second;

    // Only SYCL, TU-local kernels whose every launch this TU can see may be
    // specialized: another TU could launch a cross-TU kernel with a different
    // geometry.
    if (!binding.isTULocalOptimizable)
      continue;
    if (binding.launchSites.empty())
      continue;

    std::optional<llvm::SmallVector<int32_t, 3>> local =
        commonConstantLocal(binding);
    if (!local)
      continue;

    mlir::ArrayAttr attr = builder.getI32ArrayAttr(*local);

    // The attribute lands on the device-side spir_kernel copies (a name may
    // resolve in several device modules for multi-arch builds).
    for (cir::FuncOp kernel : binding.deviceKernels) {
      if (kernel.getCallingConv() != cir::CallingConv::SpirKernel)
        continue;
      // A required size already present is authoritative -- never override it.
      if (auto existing = kernel->getAttrOfType<mlir::ArrayAttr>(
              "cir.reqd_work_group_size")) {
        assert(existing == attr &&
               "recovered local size disagrees with an existing "
               "cir.reqd_work_group_size");
        (void)existing;
        continue;
      }
      kernel->setAttr("cir.reqd_work_group_size", attr);
      changed = true;
    }
  }

  if (!changed)
    markAllAnalysesPreserved();
}

} // namespace

std::unique_ptr<Pass> mlir::createOffloadStampReqdWorkGroupSizePass() {
  return std::make_unique<OffloadStampReqdWorkGroupSizePass>();
}
