//===- InferKernelParamNoAlias.cpp - SYCL noalias from USM allocations ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Infers `llvm.noalias` on SYCL device kernel pointer parameters when the host
// passes each from DISTINCT memory: a USM allocation, or a buffer behind an
// accessor.
//
// A spir_kernel pointer parameter is bound to the closure capture field CIRGen
// recorded for it in `cir.sycl_kernel_param` (see SYCLKernelParams.h); whole
// captured pointers and the `ptr` part of an accessor held directly by the
// closure qualify (getSYCLNoAliasField). KernelBindingTable already resolved,
// on the host side, which capture fields are backed at every launch site by
// pairwise-distinct memory (USM allocations, buffer storages) -- see
// KernelBinding::syclNoAliasFields, from the legacy trace or from SYCLHostFacts
// (--cir-offload-host-facts). This pass maps each device parameter to its field
// and stamps `llvm.noalias` on the parameters whose field is in that set. An
// accessor's other parameters (ranges, offset) and its `readonly` are left
// alone. One `kernel-param-noalias:` remark per candidate parameter reports
// the decision and its reason.
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
#include "mlir/IR/Diagnostics.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLHostFacts.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLKernelParams.h"

namespace mlir {
#define GEN_PASS_DECL_OFFLOADINFERKERNELPARAMNOALIAS
#define GEN_PASS_DEF_OFFLOADINFERKERNELPARAMNOALIAS
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace cir;

namespace {

/// Whether a kernel pointer parameter points into work-group local memory.
bool isLocalPointer(mlir::Type type) {
  auto ptrTy = mlir::dyn_cast<cir::PointerType>(type);
  auto as = ptrTy ? mlir::dyn_cast_if_present<cir::TargetAddressSpaceAttr>(
                        ptrTy.getAddrSpace())
                  : nullptr;
  return as && as.getValue() == 3;
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
  std::optional<cir::HostFactsMode> mode = cir::parseHostFactsMode(hostFacts);
  if (!mode) {
    container.emitError() << "unknown host-facts mode '" << hostFacts
                          << "' (expected legacy, ipo or both)";
    return signalPassFailure();
  }
  // The legacy table is the cached analysis the other container passes share;
  // the other modes build their own (only the noalias fields differ), on the
  // host facts the container passes share (SYCLHostFactsCache).
  std::optional<cir::KernelBindingTable> ownTable;
  cir::SYCLHostFactsCache *factsCache = nullptr;
  if (*mode != cir::HostFactsMode::Legacy) {
    cir::HostFactsOptions options;
    options.mode = *mode;
    options.assumeClosedTU = hostFactsClosedTU;
    options.bottomUp = hostFactsBottomUp;
    options.subobjects = hostFactsSubobjects;
    options.budgetScale = hostFactsBudgetScale;
    options.relevanceSlice = hostFactsSlice;
    options.maxSeconds = hostFactsMaxSeconds;
    options.maxMemoryMB = hostFactsMaxMemoryMB;
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
    // Legacy is the result; report where the two sources differ.
    unsigned disagreements = 0;
    for (const cir::HostFactsComparison &cmp :
         table.getHostFactsComparisons()) {
      mlir::Location loc = container.getLoc();
      if (const cir::KernelBinding *binding = table.lookup(cmp.kernel))
        if (!binding->deviceKernels.empty())
          loc = binding->deviceKernels.front()->getLoc();
      if (cmp.disagrees()) {
        ++disagreements;
        mlir::emitRemark(loc) << cir::formatHostFactsDisagreement(cmp);
      } else if (hostFactsCensus) {
        mlir::emitRemark(loc)
            << "host-facts: kernel " << cmp.kernel << " field " << cmp.field
            << " " << cir::formatHostFactsVerdict("legacy", cmp.legacy) << " "
            << cir::formatHostFactsVerdict("ipo", cmp.ipo);
      }
    }
    mlir::emitRemark(container.getLoc())
        << "host-facts-disagreements: " << disagreements << " of "
        << table.getHostFactsComparisons().size() << " candidate fields ("
        << (hostFactsClosedTU ? "closed-tu" : "open")
        << (hostFactsBottomUp ? ", bu" : ", ci")
        << (hostFactsSubobjects ? ", subobjects" : "") << ")";
  }
  mlir::MLIRContext *ctx = &getContext();
  llvm::StringRef noAliasName = mlir::LLVM::LLVMDialect::getNoAliasAttrName();
  bool changed = false;

  // Why a candidate field of `binding` is not stamped (or, when it is, the
  // memory it resolved to), for the remarks.
  auto noteFor = [](const cir::KernelBinding &binding,
                    llvm::StringRef field) -> std::string {
    if (binding.launchSites.empty())
      return "no launch of the kernel is recovered";
    auto it = binding.syclNoAliasNotes.find(field);
    if (it != binding.syclNoAliasNotes.end() && !it->second.empty())
      return it->second;
    if (!binding.syclNoAliasBlocked.empty())
      return binding.syclNoAliasBlocked;
    return binding.syclNoAliasFields.contains(field)
               ? ""
               : "no launch of the kernel is recovered";
  };

  // Visit every noalias candidate parameter of `kernels` (a captured USM
  // pointer or an accessor's `ptr` part; cir.sycl_kernel_param), stamp
  // llvm.noalias on those whose field is in `binding.syclNoAliasFields` when
  // `stamp`, and remark one decision per parameter. `notStamped`, if set,
  // overrides the reason for every parameter. Returns whether any parameter
  // was newly annotated.
  auto decide = [&](llvm::ArrayRef<cir::FuncOp> kernels,
                    const cir::KernelBinding &binding, bool stamp,
                    llvm::StringRef notStamped) {
    bool did = false;
    for (cir::FuncOp kernel : kernels) {
      if (kernel.getCallingConv() != cir::CallingConv::SpirKernel)
        continue;
      if (kernel.getBody().empty())
        continue;
      for (mlir::BlockArgument arg : kernel.getBody().front().getArguments()) {
        if (!mlir::isa<cir::PointerType>(arg.getType()))
          continue;
        unsigned argNo = arg.getArgNumber();
        bool isAccessor = false;
        llvm::StringRef field =
            cir::getSYCLNoAliasField(kernel, argNo, isAccessor);
        if (field.empty())
          continue;
        // A pointer to work-group local memory is not a candidate
        // (classifySYCLKernelParams): no decision to report.
        if (isLocalPointer(arg.getType()))
          continue;
        auto remark = [&]() {
          mlir::InFlightDiagnostic diag = mlir::emitRemark(kernel.getLoc());
          diag << "kernel-param-noalias: kernel @" << kernel.getName()
               << " arg " << argNo << " ("
               << (isAccessor ? "accessor" : "pointer") << " '" << field
               << "'): ";
          return diag;
        };
        // Already noalias (e.g. a restrict-qualified capture): leave it.
        if (kernel.getArgAttr(argNo, noAliasName)) {
          remark() << "already noalias";
          continue;
        }
        std::string note = noteFor(binding, field);
        if (!stamp || !binding.syclNoAliasFields.contains(field)) {
          remark() << "not stamped ("
                   << (notStamped.empty() ? llvm::StringRef(note) : notStamped)
                   << ")";
          continue;
        }
        kernel.setArgAttr(argNo, noAliasName, mlir::UnitAttr::get(ctx));
        did = true;
        mlir::InFlightDiagnostic diag = remark();
        diag << "noalias";
        if (!note.empty())
          diag << " (" << note << ")";
      }
    }
    return did;
  };

  unsigned specTag = 0;
  for (const auto &entry : table) {
    const cir::KernelBinding &binding = entry.second;
    if (binding.kind != cir::BindingKind::SYCLName)
      continue;

    // An empty field set means host-side recovery could not prove any pair
    // distinct at the launches this TU sees.
    if (binding.launchSites.empty() || binding.syclNoAliasFields.empty()) {
      decide(binding.deviceKernels, binding, /*stamp=*/false, "");
      continue;
    }

    if (binding.isTULocalOptimizable) {
      // Every launch is visible and the kernel is TU-local: safe to annotate
      // the shared device kernel in place.
      changed |= decide(binding.deviceKernels, binding, /*stamp=*/true, "");
    } else {
      // Residual: the shared kernel is cross-TU / by-name reachable, so another
      // TU could launch it with aliasing pointers. Specialize by clone --
      // redirect this TU's launches to a fresh kernel and annotate only the
      // clone, leaving the generic original for unseen launches.
      auto spec = cir::specializeSYCLKernelForLaunches(
          container, binding, binding.launchSites,
          "noalias." + std::to_string(specTag++));
      if (spec) {
        if (factsCache)
          factsCache->recordSpecialization(spec->name, entry.first);
        decide(spec->deviceKernels, binding, /*stamp=*/true, "");
        changed = true; // The clone + redirect mutated the IR regardless.
      } else {
        decide(binding.deviceKernels, binding, /*stamp=*/false,
               "the kernel is visible to other TUs and its launches cannot be "
               "redirected to a clone");
      }
    }
  }

  if (!changed)
    markAllAnalysesPreserved();
  else
    // SOUNDNESS: see SYCLHostFactsCache: the stamps and the specializations
    // leave the host facts true.
    markAnalysesPreserved<cir::SYCLHostFactsCache>();
}

} // namespace

std::unique_ptr<Pass> mlir::createOffloadInferKernelParamNoAliasPass() {
  return std::make_unique<OffloadInferKernelParamNoAliasPass>();
}
