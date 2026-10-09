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
// When every launch of a kernel uses the same compile-time-constant nd_range
// local size and no other TU launches it (KernelBinding::stampsInPlace: a
// TU-unique kernel or the kernel copy of an internalized launch path; under
// -cir-offload-assume-closed-tu, a lambda closure), that size is a required
// work-group size for the device kernel. A kernel other TUs can launch (a
// lambda in an inline function has one closure type and one weak_odr device
// kernel for the program) is not stamped: another TU may launch it with
// another local size. The attribute records the sizes in natural {X,Y,Z} order
// (length == number of dims); the SYCL-shaped reversal/padding is applied later
// at LLVM lowering (see LowerToLLVMIR.cpp amendReqdWorkGroupSize).
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Diagnostics.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLHostFacts.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>

namespace mlir {
#define GEN_PASS_DECL_OFFLOADSTAMPREQDWORKGROUPSIZE
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
// The constant local size of one geometry: nullopt if it has none (a plain
// range launch, an invalid geometry) or a component is not a constant.
static std::optional<llvm::SmallVector<int32_t, 3>>
constantLocal(const cir::SyclGeometry &geom) {
  if (!geom.valid || geom.local.empty())
    return std::nullopt;
  llvm::SmallVector<int32_t, 3> cur;
  for (const std::optional<int64_t> &dim : geom.local) {
    if (!dim)
      return std::nullopt;
    cur.push_back(static_cast<int32_t>(*dim));
  }
  return cur;
}

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

// The required work-group size the launches of `binding` agree on, if any.
static std::optional<llvm::SmallVector<int32_t, 3>>
recoveredLocal(const cir::KernelBinding &binding) {
  if (binding.launchSites.empty())
    return std::nullopt;
  return commonConstantLocal(binding);
}

// The required work-group size the pass stamps on `binding`'s kernels, if
// any: they agree on a constant local size and no other TU launches the
// kernel (or closed-TU is assumed).
static std::optional<llvm::SmallVector<int32_t, 3>>
stampedLocal(const cir::KernelBinding &binding, bool assumeClosedTU) {
  if (!binding.stampsInPlace(assumeClosedTU))
    return std::nullopt;
  return recoveredLocal(binding);
}

static std::string
formatLocal(const std::optional<llvm::SmallVector<int32_t, 3>> &local) {
  if (!local)
    return "none";
  std::string out = "[";
  for (auto [i, d] : llvm::enumerate(*local))
    out += (i ? "," : "") + std::to_string(d);
  return out + "]";
}

void OffloadStampReqdWorkGroupSizePass::runOnOperation() {
  mlir::ModuleOp container = getOperation();
  if (!cir::isOffloadContainer(container))
    return;
  std::optional<cir::HostFactsMode> mode = cir::parseHostFactsMode(hostFacts);
  if (!mode) {
    container.emitError() << "unknown host-facts mode '" << hostFacts
                          << "' (expected legacy, ipo or both)";
    return signalPassFailure();
  }
  // The legacy table is the cached analysis the other container passes share;
  // the other modes build their own.
  cir::HostFactsOptions options;
  options.mode = *mode;
  options.assumeClosedTU = hostFactsClosedTU;
  options.bottomUp = hostFactsBottomUp;
  options.subobjects = hostFactsSubobjects;
  options.budgetScale = hostFactsBudgetScale;
  options.relevanceSlice = hostFactsSlice;
  options.maxSeconds = hostFactsMaxSeconds;
  options.maxMemoryMB = hostFactsMaxMemoryMB;
  options.internalize = hostFactsInternalize;
  std::optional<cir::KernelBindingTable> ownTable;
  cir::SYCLHostFactsCache *factsCache = nullptr;
  if (*mode != cir::HostFactsMode::Legacy) {
    factsCache = &getAnalysis<cir::SYCLHostFactsCache>();
    unsigned builtBefore = factsCache->getNumBuilt();
    ownTable.emplace(container, options, factsCache);
    numHostFactsBuilt += factsCache->getNumBuilt() - builtBefore;
    // The budgets that tripped while this table was built (once per
    // change: the passes share the analysis).
    if (std::optional<std::string> trips =
            factsCache->get(options).takeBudgetReport())
      mlir::emitRemark(container.getLoc())
          << "host-facts budget trips: " << *trips << " "
          << cir::formatHostFactsOptions(options);
  }
  cir::KernelBindingTable &table =
      ownTable ? *ownTable : getAnalysis<cir::KernelBindingTable>();
  if (*mode == cir::HostFactsMode::Both) {
    // Legacy is the result; report where the two geometries differ, and, for
    // the census, what each source would stamp.
    unsigned disagreements = 0;
    for (const cir::HostFactsGeometryComparison &cmp :
         table.getHostFactsGeometryComparisons()) {
      mlir::Location loc = container.getLoc();
      const cir::KernelBinding *binding = table.lookup(cmp.kernel);
      if (binding && !binding->deviceKernels.empty())
        loc = binding->deviceKernels.front()->getLoc();
      if (cmp.disagrees()) {
        ++disagreements;
        mlir::emitRemark(loc) << cir::formatHostFactsDisagreement(cmp);
      } else if (hostFactsCensus) {
        mlir::emitRemark(loc)
            << "host-facts: " << cir::formatHostFactsComparison(cmp);
      }
      if (hostFactsCensus && binding) {
        cir::KernelBinding ipo = *binding;
        for (cir::LaunchSite &site : ipo.launchSites)
          site.syclNDRange = cmp.ipo;
        mlir::emitRemark(loc)
            << "host-facts: kernel " << cmp.kernel
            << " reqd-work-group-size legacy="
            << formatLocal(stampedLocal(*binding, hostFactsClosedTU))
            << " ipo=" << formatLocal(stampedLocal(ipo, hostFactsClosedTU));
      }
    }
    mlir::emitRemark(container.getLoc())
        << "host-facts-disagreements: " << disagreements << " of "
        << table.getHostFactsGeometryComparisons().size()
        << " kernel geometries " << cir::formatHostFactsOptions(options);
  }
  mlir::OpBuilder builder(container);
  bool changed = false;

  for (const auto &entry : table) {
    const cir::KernelBinding &binding = entry.second;

    // Only SYCL kernels no other TU launches may be specialized: another TU
    // could launch a shared kernel with a different geometry.
    std::optional<llvm::SmallVector<int32_t, 3>> local =
        stampedLocal(binding, hostFactsClosedTU);
    if (!local) {
      // A kernel other TUs can launch (no closed-TU): say why not.
      if (binding.kind != cir::BindingKind::SYCLName || hostFactsClosedTU ||
          binding.stampsInPlace(false) || binding.deviceKernels.empty())
        continue;
      if (std::optional<llvm::SmallVector<int32_t, 3>> recovered =
              recoveredLocal(binding)) {
        cir::FuncOp kernel = binding.deviceKernels.front();
        mlir::emitRemark(kernel.getLoc())
            << "reqd-work-group-size: kernel @" << kernel.getName() << " "
            << formatLocal(recovered) << " not stamped ("
            << cir::getSharedSYCLKernelNote(hostFactsInternalize) << ")";
      }
      continue;
    }

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
      if (cir::isInternalizedSYCLKernel(kernel)) {
        cir::noteInternalizedSYCLKernelUse(kernel);
        mlir::emitRemark(kernel.getLoc())
            << "reqd-work-group-size: kernel @" << kernel.getName() << " "
            << formatLocal(local)
            << cir::describeInternalizedSYCLKernel(kernel);
      }
    }
  }

  if (!changed)
    markAllAnalysesPreserved();
  else
    // SOUNDNESS: see SYCLHostFactsCache: the device-side changes and the
    // specializations leave the host facts true.
    markAnalysesPreserved<cir::SYCLHostFactsCache>();
}

} // namespace

std::unique_ptr<Pass> mlir::createOffloadStampReqdWorkGroupSizePass() {
  return std::make_unique<OffloadStampReqdWorkGroupSizePass>();
}
