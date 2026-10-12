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
// KernelBinding::syclNoAliasFields, from SYCLHostFacts, the points-to analysis
// of the host module (--cir-offload-host-facts). This pass maps each device
// parameter to its field and stamps `llvm.noalias` on the parameters whose
// field is in that set. An accessor's other parameters (ranges, offset) and
// its `readonly` are left alone. One remark per candidate parameter (category
// cir-offload-noalias, see emitOffloadRemark) reports the decision and its
// reason.
//
// Where the facts go. The device kernel is one weak_odr definition for the
// whole program, and syclNoAliasFields ANDs the distinctness over this TU's
// launches only, so the shared kernel is annotated in place only when no
// other TU can launch it (KernelBinding::stampsInPlace: isTUUnique, which
// includes the kernel copies of internalized launch paths, T26; under
// -cir-offload-assume-closed-tu, any kernel that is not registered).
// Otherwise nothing is stamped (the remark says the kernel is visible to
// other TUs); the facts reach the kernel copy of an internalized launch path
// instead (cir-offload-internalize-launch-paths, which runs before this
// pass).
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
    return markAllAnalysesPreserved();
  std::optional<cir::HostFactsMode> mode = cir::parseHostFactsMode(hostFacts);
  if (!mode) {
    container.emitError() << "unknown host-facts mode '" << hostFacts
                          << "' (expected ipo or none)";
    return signalPassFailure();
  }
  if (*mode == cir::HostFactsMode::None)
    return markAllAnalysesPreserved();
  // The table and the host facts the container passes share
  // (SYCLHostFactsCache).
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
  cir::SYCLHostFactsCache &factsCache = getAnalysis<cir::SYCLHostFactsCache>();
  unsigned builtBefore = factsCache.getNumBuilt();
  unsigned tablesBefore = factsCache.getNumTablesBuilt();
  const cir::KernelBindingTable &table = factsCache.getTable(options);
  numHostFactsBuilt += factsCache.getNumBuilt() - builtBefore;
  numTablesBuilt += factsCache.getNumTablesBuilt() - tablesBefore;
  // The budgets that tripped while the analysis was built (once: the passes
  // share it).
  if (std::optional<std::string> trips =
          factsCache.get(options).takeBudgetReport())
    mlir::emitRemark(container.getLoc())
        << "host-facts budget trips: " << *trips << " "
        << cir::formatHostFactsOptions(options);
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
        std::string text;
        llvm::raw_string_ostream os(text);
        os << "kernel-param-noalias: kernel @" << kernel.getName() << " arg "
           << argNo << " (" << (isAccessor ? "accessor" : "pointer") << " '"
           << field << "'): ";
        auto remark = [&](cir::OffloadRemarkKind kind) {
          cir::emitOffloadRemark(
              kernel.getLoc(), kind, "InferKernelParamNoAlias",
              cir::kNoAliasRemarkCategory, kernel.getName(), text);
        };
        // Already noalias (e.g. a restrict-qualified capture): leave it.
        if (kernel.getArgAttr(argNo, noAliasName)) {
          os << "already noalias";
          remark(cir::OffloadRemarkKind::Analysis);
          continue;
        }
        std::string note = noteFor(binding, field);
        if (!stamp || !binding.syclNoAliasFields.contains(field)) {
          os << "not stamped ("
             << (notStamped.empty() ? llvm::StringRef(note) : notStamped)
             << ")";
          remark(cir::OffloadRemarkKind::Missed);
          continue;
        }
        kernel.setArgAttr(argNo, noAliasName, mlir::UnitAttr::get(ctx));
        cir::noteInternalizedSYCLKernelUse(kernel);
        did = true;
        os << "noalias";
        if (!note.empty())
          os << " (" << note << ")";
        os << cir::describeInternalizedSYCLKernel(kernel);
        remark(cir::OffloadRemarkKind::Passed);
      }
    }
    return did;
  };

  // Without closed-TU the remark names whether the launch paths were
  // internalized.
  std::string sharedNote = cir::getSharedSYCLKernelNote(hostFactsInternalize);
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

    if (binding.stampsInPlace(hostFactsClosedTU)) {
      // No other TU launches the kernel (or closed-TU is assumed): safe to
      // annotate the shared device kernel in place.
      changed |= decide(binding.deviceKernels, binding, /*stamp=*/true, "");
    } else {
      // Another TU can launch the kernel, with other pointers: this TU's
      // facts do not hold for it, but for the kernel copies of the
      // internalized paths. Under closed-TU, a registered kernel: the
      // runtime may launch it by name.
      decide(binding.deviceKernels, binding, /*stamp=*/false,
             hostFactsClosedTU ? "the kernel is registered: it may be "
                                 "launched by name"
                               : llvm::StringRef(sharedNote));
    }
  }

  if (!changed)
    markAllAnalysesPreserved();
  else
    // SOUNDNESS: see SYCLHostFactsCache: the stamps leave the host facts and
    // the table true.
    markAnalysesPreserved<cir::SYCLHostFactsCache>();
}

} // namespace

std::unique_ptr<Pass> mlir::createOffloadInferKernelParamNoAliasPass() {
  return std::make_unique<OffloadInferKernelParamNoAliasPass>();
}
