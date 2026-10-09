//===- InternalizeLaunchPaths.cpp - Internalize SYCL host launch paths ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// cir-offload-internalize-launch-paths and
// cir-offload-prune-internalized-launch-paths (T26, plan 2.8).
//
// The problem. A SYCL launch sits in functions another TU can call
// (queue::submit<L>, the CGF lambda's operator(), handler::parallel_for<K,KL>,
// usually the user's inline launch function too). Facts derived from this
// TU's callers of such a function (the points-to analysis' top-down
// contexts) do not hold when the linker binds another TU's call to our copy,
// with other objects; without -cir-offload-assume-closed-tu they are
// rejected (Incomplete formals, UnknownCaller), so open mode proves nothing.
//
// The fix. internalizeLaunchPaths (LaunchPathInternalization.h) copies every
// function another TU could call on a path from a closed root (main, the
// dynamic initializers) to a kernel.info marker (getDeviceKernelInfo<K>, the
// end of every launch path) as an internal copy, and redirects the path to
// the copies: direct calls, the CGF invoker the cgfo.ctor stores (vouched:
// the SYCL runtime only calls it), indirect calls (guarded direct calls) and
// std::thread state vtables. The copies have only callers in this TU, so the
// analysis (with PointsToAnalysisOptions::localFunctionsClosed) covers them
// from the roots.
//
// The kernel side. The device kernel K is one weak_odr definition for the
// whole program, launched by every TU; this TU's facts are an AND over this
// TU's launches only. So the markers of the copies name a new kernel K' =
// `K.int.<module id>` (copySYCLKernelLocally: a device clone, internal host
// name string, CompileTimeKernelInfo and getter). The module id is an MD5 of
// the names of the module's strong external definitions (the scheme of
// llvm::getUniqueModuleId): two TUs of a program cannot define the same
// strong symbol, so no other TU mints the same K'. The originals keep
// launching the unstamped K. A kernel no other TU can launch whose paths are
// all clean already needs no copy (isTUUniqueSYCLKernel lets the passes
// stamp it in place).
//
// Legacy host facts (T26b). Without closed-TU a kernel other TUs can launch
// gets no fact in place, in every host-facts mode, so this pass runs under
// legacy too. The legacy trace reads only the body of the function holding a
// launch (after inlining the launch shims and the command group), so its
// facts do not depend on that function's callers: a kernel with no launch
// that gives a fact is skipped, and the functions holding one are roots of
// their own paths (KernelBinding::syclLegacyFactSites), which recovers the
// launches of a library TU without main.
//
// Ordering. This runs before the host-facts passes: the analysis they share
// (SYCLHostFactsCache) is built on the internalized module, so its soundness
// argument is the analysis' own (the module is equivalent to the original)
// and the cache's invalidation argument is unchanged. The prune pass runs
// after them: a K' none of them used (noteInternalizedSYCLKernelUse) is
// removed with every path copy that only led to it, so the output keeps
// copies only for launches with a fact.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/LaunchPathInternalization.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLKernelParams.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLPointsToModels.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MD5.h"

namespace mlir {
#define GEN_PASS_DECL_OFFLOADINTERNALIZELAUNCHPATHS
#define GEN_PASS_DEF_OFFLOADINTERNALIZELAUNCHPATHS
#define GEN_PASS_DECL_OFFLOADPRUNEINTERNALIZEDLAUNCHPATHS
#define GEN_PASS_DEF_OFFLOADPRUNEINTERNALIZEDLAUNCHPATHS
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace cir;

namespace {

llvm::StringRef getRole(Operation *func) {
  auto role =
      func->getAttrOfType<StringAttr>(CIRDialect::getSYCLHostAPIAttrName());
  return role ? role.getValue() : llvm::StringRef();
}

/// An id no other TU of the program has: the MD5 of the sorted names of the
/// module's strong external definitions (llvm::getUniqueModuleId), or empty
/// if it has none.
std::string getUniqueModuleId(ModuleOp module) {
  llvm::SmallVector<llvm::StringRef> names;
  for (Operation &op : module.getBody()->getOperations()) {
    if (auto func = dyn_cast<cir::FuncOp>(op)) {
      if (!func.isDeclaration() &&
          func.getLinkage() == cir::GlobalLinkageKind::ExternalLinkage)
        names.push_back(func.getSymName());
    } else if (auto global = dyn_cast<cir::GlobalOp>(op)) {
      if (!global.isDeclaration() &&
          global.getLinkage() == cir::GlobalLinkageKind::ExternalLinkage)
        names.push_back(global.getSymName());
    }
  }
  if (names.empty())
    return "";
  llvm::sort(names);
  llvm::MD5 md5;
  for (llvm::StringRef name : names) {
    md5.update(name);
    md5.update(llvm::ArrayRef<uint8_t>{0});
  }
  llvm::MD5::MD5Result result;
  md5.final(result);
  return llvm::toHex(llvm::ArrayRef<uint8_t>(result.data(), 8),
                     /*LowerCase=*/true);
}

/// The vouched address-of edge: the CGF invoker a cgfo.ctor stores (its body
/// holds the get_global with the real headers; the reduced test API passes
/// it to the ctor). SPEC/RUNTIME: type_erased_cgfo_ty keeps the invoker only
/// to call it, from queue::submit_with_event_impl (the T14 submit model, PI
/// 14573-14574); nothing compares it.
bool isVouchedInvokerAddress(cir::GetGlobalOp addr, cir::FuncOp holder) {
  if (getRole(holder) == "cgfo.ctor")
    return true;
  for (Operation *user : addr->getUsers()) {
    auto call = dyn_cast<cir::CallOp>(user);
    if (!call || call.isIndirect())
      return false;
    auto callee = SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(
        call, call.getCalleeAttr());
    if (!callee || getRole(callee) != "cgfo.ctor")
      return false;
  }
  return !addr->use_empty();
}

/// Why no host fact is possible for a kernel with device kernels
/// \p kernels, or empty. A fact needs a noalias candidate (a captured USM
/// pointer or accessor, getSYCLNoAliasCandidates, and no poison), a scalar
/// capture (a constant), or an nd_range launch (a local size, for the
/// required work-group size), which only a kernel whose function object
/// takes an nd_item can have.
std::string noFactPossibleWhy(llvm::ArrayRef<cir::FuncOp> kernels) {
  cir::SYCLNoAliasCandidates candidates =
      cir::getSYCLNoAliasCandidates(kernels);
  if (!candidates.fields.empty() && candidates.poison.empty())
    return "";
  for (cir::FuncOp kernel : kernels) {
    for (unsigned i = 0, e = kernel.getNumArguments(); i != e; ++i)
      if (isa<cir::IntType, cir::FPTypeInterface, cir::BoolType>(
              kernel.getArgumentTypes()[i]) &&
          !cir::getSYCLCapturedField(kernel, i, /*wantPointer=*/false).empty())
        return "";
    cir::FuncOp op = cir::getSYCLClosureCallOperator(kernel);
    if (!op)
      return ""; // Unknown shape: do not rule anything out.
    for (Type type : op.getFunctionType().getInputs()) {
      if (auto ptr = dyn_cast<cir::PointerType>(type))
        type = ptr.getPointee();
      auto rec = dyn_cast<cir::RecordType>(type);
      if (rec && rec.getName() && rec.getName().getValue().contains("nd_item"))
        return "";
    }
  }
  return "no noalias candidate, scalar capture or nd_range launch";
}

struct OffloadInternalizeLaunchPathsPass
    : public impl::OffloadInternalizeLaunchPathsBase<
          OffloadInternalizeLaunchPathsPass> {
  void runOnOperation() override;
};

void OffloadInternalizeLaunchPathsPass::runOnOperation() {
  ModuleOp container = getOperation();
  std::optional<cir::HostFactsMode> mode = cir::parseHostFactsMode(hostFacts);
  // Every host-facts mode: the legacy trace reads the copies' launches as it
  // reads the originals' (it inlines the launch shims by name, and a copy
  // keeps its original's name with a suffix).
  if (!cir::isOffloadContainer(container) || !mode || hostFactsClosedTU)
    return markAllAnalysesPreserved();
  ModuleOp host = cir::getOffloadContainerHostModule(container);
  if (!host)
    return markAllAnalysesPreserved();

  // The sinks: the kernel.info getters that name a device kernel for which a
  // fact is possible (the paths of the others could only cost: no clone).
  llvm::StringMap<llvm::SmallVector<cir::FuncOp, 1>> deviceKernels;
  for (ModuleOp dm : cir::getOffloadContainerDeviceModules(container))
    for (auto kernel : dm.getOps<cir::FuncOp>())
      if (!kernel.isDeclaration() &&
          kernel.getCallingConv() == cir::CallingConv::SpirKernel)
        deviceKernels[kernel.getSymName()].push_back(kernel);
  llvm::StringSet<> possible;
  unsigned numNoFact = 0;
  for (auto &[name, kernels] : deviceKernels) {
    if (noFactPossibleWhy(kernels).empty())
      possible.insert(name);
    else
      ++numNoFact;
  }
  // Legacy host facts: the trace is intraprocedural (it reads the function
  // holding a launch, after inlining the launch shims and the command
  // group), so a launch in a copy gives exactly what the launch it was copied
  // from gives, whoever calls it. So:
  //  - a kernel none of whose launches gives a fact on its own gets none
  //    through a copy either: skip it;
  //  - the functions holding a launch that does are roots too, kept in place
  //    (an external one included): the kernel copy they then launch has
  //    facts that hold at each of its launches for any caller, and no other
  //    TU names it. (An external root that the linker replaces by another
  //    TU's copy launches the original kernel, unstamped.)
  //  - a kernel no other TU can launch (KernelBinding::isTUUnique) is
  //    stamped in place over all its launches, as before: no copy (no
  //    duplicate device kernel).
  // The legacy table is the shared analysis, usually already built.
  unsigned numNoLegacyFact = 0, numTUUnique = 0;
  llvm::StringSet<> legacyRoots;
  if (*mode == cir::HostFactsMode::Legacy) {
    const cir::KernelBindingTable &table =
        getAnalysis<cir::KernelBindingTable>();
    for (auto &[name, kernels] : deviceKernels) {
      if (!possible.contains(name))
        continue;
      const cir::KernelBinding *binding = table.lookup(name);
      if (binding && binding->isTUUnique) {
        possible.erase(name);
        ++numTUUnique;
        continue;
      }
      if (binding && !binding->syclLegacyFactSites.empty()) {
        for (const std::string &root : binding->syclLegacyFactSites)
          legacyRoots.insert(root);
        continue;
      }
      possible.erase(name);
      ++numNoLegacyFact;
    }
  }
  // The kernel each getter names (looked up once, with symbol tables).
  SymbolTableCollection symbols;
  llvm::DenseMap<Operation *, std::string> kernelOf;
  auto kernelName = [&](cir::FuncOp getter) -> llvm::StringRef {
    auto [it, inserted] = kernelOf.try_emplace(getter);
    if (inserted)
      it->second = cir::getSYCLKernelInfoName(getter, &symbols);
    return it->second;
  };
  llvm::SmallVector<cir::FuncOp> sinks;
  for (auto func : host.getOps<cir::FuncOp>())
    if (getRole(func) == "kernel.info" && !func.isDeclaration() &&
        possible.contains(kernelName(func)))
      sinks.push_back(func);
  if (numNoFact)
    emitRemark(container.getLoc())
        << "launch-path-internalization: " << numNoFact << " kernel"
        << (numNoFact == 1 ? "" : "s")
        << " skipped: no fact is possible (no noalias candidate, scalar "
           "capture or nd_range launch)";
  if (numNoLegacyFact)
    emitRemark(container.getLoc())
        << "launch-path-internalization: " << numNoLegacyFact << " kernel"
        << (numNoLegacyFact == 1 ? "" : "s")
        << " skipped: no launch gives a legacy host fact";
  if (numTUUnique)
    emitRemark(container.getLoc())
        << "launch-path-internalization: " << numTUUnique << " kernel"
        << (numTUUnique == 1 ? "" : "s")
        << " skipped: stamped in place (no other TU can launch "
        << (numTUUnique == 1 ? "it" : "them") << ")";
  if (sinks.empty())
    return markAllAnalysesPreserved();

  std::string moduleId = getUniqueModuleId(host);
  if (moduleId.empty()) {
    emitRemark(container.getLoc())
        << "launch-path-internalization: none (the host module has no "
           "strong external definition to derive a program-unique kernel "
           "name from)";
    return markAllAnalysesPreserved();
  }

  cir::InternalizationOptions options;
  options.vouchAddress = isVouchedInvokerAddress;
  options.maxFunctions = maxFunctions;
  options.maxOps = maxOps;
  if (!legacyRoots.empty())
    options.extraRoot = [&](cir::FuncOp func) {
      return legacyRoots.contains(func.getSymName());
    };
  cir::InternalizationResult result =
      cir::internalizeLaunchPaths(host, sinks, options);
  numCopies += result.numCloned;
  if (!result.skipped.empty()) {
    InFlightDiagnostic diag = emitRemark(container.getLoc());
    diag << "launch-path-internalization budget: " << result.budgetWhy
         << "; not internalized:";
    for (cir::FuncOp sink : result.skipped)
      diag << " " << kernelName(sink);
  }

  for (auto &[sink, why] : result.unreached)
    emitRemark(sink.getLoc())
        << "launch-path-internalization: kernel @" << kernelName(sink)
        << " not internalized (" << why << ")";

  // A kernel copy per marker reached through a copy.
  MLIRContext *ctx = &getContext();
  unsigned numKernels = 0;
  cir::LocalKernelCopier copier(container);
  for (cir::InternalizationResult::Sink &sink : result.sinks) {
    if (!sink.throughClone)
      continue;
    std::string kernel = kernelName(sink.sink).str();
    std::string name = kernel + ".int." + moduleId;
    std::optional<cir::LocalKernelCopy> copy =
        copier.copy(sink.sink, name, ".int");
    if (!copy) {
      emitRemark(sink.sink.getLoc())
          << "launch-path-internalization: kernel " << kernel
          << " not specialized (its kernel info has an unexpected shape)";
      continue;
    }
    for (Operation *op : sink.calls) {
      auto call = cast<cir::CallOp>(op);
      call.setCalleeAttr(FlatSymbolRefAttr::get(copy->getter.getSymNameAttr()));
      call->setAttr(cir::kInternalizedEdgeAttr, sink.sink.getSymNameAttr());
    }
    SmallVector<Attribute> roots;
    for (const std::string &root : sink.roots)
      roots.push_back(StringAttr::get(ctx, root));
    auto info = DictionaryAttr::get(
        ctx, {NamedAttribute(StringAttr::get(ctx, "original"),
                             StringAttr::get(ctx, kernel)),
              NamedAttribute(StringAttr::get(ctx, "roots"),
                             ArrayAttr::get(ctx, roots))});
    for (cir::FuncOp device : copy->deviceKernels)
      device->setAttr(cir::kInternalizedKernelAttr, info);
    ++numKernels;
    ++numKernelCopies;
    emitRemark(copy->deviceKernels.front().getLoc())
        << "launch-path-internalization: kernel @" << kernel << " -> @"
        << copy->name << " for " << sink.calls.size() << " launch marker"
        << (sink.calls.size() == 1 ? "" : "s")
        << cir::describeInternalizedSYCLKernel(copy->deviceKernels.front());
  }
  emitRemark(container.getLoc())
      << "launch-path-internalization: " << numKernels << " kernel"
      << (numKernels == 1 ? "" : "s") << " specialized, " << result.numCloned
      << " function copies (" << result.numOps << " ops), " << result.numInPlace
      << " clean in place, " << result.numGuards << " guarded calls, "
      << result.numVTables << " vtable copies; " << result.skipped.size()
      << " kernel" << (result.skipped.size() == 1 ? "" : "s") << " over budget";
}

struct OffloadPruneInternalizedLaunchPathsPass
    : public impl::OffloadPruneInternalizedLaunchPathsBase<
          OffloadPruneInternalizedLaunchPathsPass> {
  void runOnOperation() override;
};

void OffloadPruneInternalizedLaunchPathsPass::runOnOperation() {
  ModuleOp container = getOperation();
  if (!cir::isOffloadContainer(container))
    return markAllAnalysesPreserved();
  ModuleOp host = cir::getOffloadContainerHostModule(container);
  // The kernel copies, used or not.
  llvm::StringSet<> used, unused;
  SmallVector<cir::FuncOp> unusedKernels;
  for (ModuleOp dm : cir::getOffloadContainerDeviceModules(container))
    for (auto kernel : dm.getOps<cir::FuncOp>()) {
      if (!cir::isInternalizedSYCLKernel(kernel))
        continue;
      if (kernel->hasAttr(cir::kInternalizedUsedAttr)) {
        used.insert(kernel.getSymName());
      } else {
        unused.insert(kernel.getSymName());
        unusedKernels.push_back(kernel);
      }
    }
  if (used.empty() && unused.empty() &&
      llvm::none_of(host.getOps<cir::FuncOp>(), [](cir::FuncOp f) {
        return f->hasAttr(cir::kInternalizedAttr);
      }))
    return markAllAnalysesPreserved();
  // The getter copies of the used kernel copies are what the paths lead to.
  SymbolTableCollection symbols;
  llvm::DenseMap<Operation *, bool> keeps;
  unsigned erased = cir::pruneInternalizedPaths(host, [&](cir::FuncOp f) {
    auto [it, inserted] = keeps.try_emplace(f, false);
    if (inserted)
      it->second = f->hasAttr(cir::kInternalizedAttr) &&
                   getRole(f) == "kernel.info" &&
                   used.contains(cir::getSYCLKernelInfoName(f, &symbols));
    return it->second;
  });
  for (cir::FuncOp kernel : unusedKernels)
    kernel.erase();
  cir::stripInternalizationTags(host);
  for (ModuleOp dm : cir::getOffloadContainerDeviceModules(container))
    for (auto kernel : dm.getOps<cir::FuncOp>()) {
      kernel->removeAttr(cir::kInternalizedKernelAttr);
      kernel->removeAttr(cir::kInternalizedUsedAttr);
    }
  emitRemark(container.getLoc())
      << "launch-path-internalization: kept " << used.size() << " of "
      << used.size() + unused.size() << " kernel copies, erased " << erased
      << " host copies";
}

} // namespace

std::unique_ptr<Pass> mlir::createOffloadInternalizeLaunchPathsPass() {
  return std::make_unique<OffloadInternalizeLaunchPathsPass>();
}

std::unique_ptr<Pass> mlir::createOffloadPruneInternalizedLaunchPathsPass() {
  return std::make_unique<OffloadPruneInternalizedLaunchPathsPass>();
}
