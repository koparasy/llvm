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
// Open roots (T28). A kernel no closed root reaches (a TU without main) gets
// its paths internalized from the externally visible functions nothing in the
// TU uses, whose callers are all code we don't see: see
// internalizeFromOpenRoots for the rule and its soundness argument.
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
#include "llvm/ADT/DenseSet.h"
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

/// The symbols \p host uses outside their own bodies, or std::nullopt if
/// the uses cannot be enumerated.
std::optional<llvm::DenseSet<StringAttr>> getUsedSymbols(ModuleOp host) {
  std::optional<SymbolTable::UseRange> uses =
      SymbolTable::getSymbolUses(&host.getBodyRegion());
  if (!uses)
    return std::nullopt;
  llvm::DenseSet<StringAttr> used;
  for (const SymbolTable::SymbolUse &use : *uses) {
    StringAttr name = use.getSymbolRef().getRootReference();
    auto parent = use.getUser()->getParentOfType<cir::FuncOp>();
    if (!parent || parent.getSymNameAttr() != name)
      used.insert(name);
  }
  return used;
}

/// Whether \p func can be an open root: a definition another TU can call
/// (external, linkonce_odr, weak, weak_odr linkage; not
/// available_externally, whose body this TU does not provide) that nothing
/// in the module uses (no call, address or initializer names it, except its
/// own body), so its callers are all in code we don't see. \p used: the
/// symbols the module used outside their own bodies before any path was
/// internalized (the originals the closed roots' copies replace are not
/// roots).
bool isOpenRoot(cir::FuncOp func, const llvm::DenseSet<StringAttr> &used) {
  if (func.isDeclaration() || cir::isLocalLinkage(func.getLinkage()) ||
      func.getLinkage() == cir::GlobalLinkageKind::AvailableExternallyLinkage ||
      cir::isClosedRoot(func) || func->hasAttr(cir::kInternalizedAttr))
    return false;
  return !used.contains(func.getSymNameAttr());
}

/// Open roots (T28). A kernel no closed root reaches (a library TU without
/// main: s3d's S3D.cpp; a launch only an external entry point reaches) may
/// still get facts that hold for every caller: internalize its paths from
/// the open roots (isOpenRoot), kept in place like closed roots, whose
/// copies and kernel copy K.int then have the open root as their only
/// caller.
///
/// SOUNDNESS. The facts at K.int's launches are the points-to analysis'
/// facts at launches reached only from open roots R. R is externally
/// visible, so the analysis treats it as having callers we don't see: its
/// formals are Incomplete (UnknownCaller), and so is what they reach;
/// nameable globals are External; and the noalias, constant and geometry
/// rules (SYCLHostFacts.cpp) accept a fact only if it does not depend on
/// Incomplete or External memory or on a formal's targets. What survives
/// depends only on what R's analysed subtree does itself: allocations made
/// inside it, constants it stores, its own control flow. That holds for any
/// call of R:
///  - a call from code we don't see bound to our copy of R runs this body
///    (an inline R has the same body in every TU, C++20 [basic.def.odr]p14),
///    with unknown arguments and unknown global state, which the analysis
///    already assumed; it reaches K.int only through R's redirected edges;
///  - a call the linker binds to another TU's copy of R (linkonce, weak)
///    runs that copy, which launches the original K, unstamped;
///  - the copies of R's subtree are local, called only by R and by each
///    other (localFunctionsClosed): no other path reaches them. R itself is
///    analysed even if it is an unused ODR function (skipUnusedODRFunctions
///    keeps a body that names a local function: it can run code of this TU
///    that a foreign copy would not).
/// So the kernel K.int, which no other TU names, has at every launch the
/// facts the analysis proved.
///
/// Only the sinks no closed root reaches (phase 1) are planned here, with
/// copies of their own: a kernel a closed root reaches keeps the facts its
/// closed-root contexts give, which an open root's Incomplete context would
/// weaken (one K.int per kernel, ANDed over its launches). A kernel no other
/// TU can launch (KernelBinding::isTUUnique) is not planned: the passes
/// stamp it in place over all of its launches, which include those an open
/// root reaches.
void internalizeFromOpenRoots(
    ModuleOp container, ModuleOp host,
    const cir::InternalizationOptions &closedOptions,
    llvm::function_ref<llvm::StringRef(cir::FuncOp)> kernelName,
    const std::optional<llvm::DenseSet<StringAttr>> &used,
    cir::InternalizationResult &result) {
  if (result.unreached.empty())
    return;
  llvm::StringMap<bool> locality = cir::getSYCLKernelInfoLocality(container);
  llvm::StringMap<cir::FuncOp> deviceKernel;
  for (ModuleOp dm : cir::getOffloadContainerDeviceModules(container))
    for (auto kernel : dm.getOps<cir::FuncOp>())
      if (!kernel.isDeclaration() &&
          kernel.getCallingConv() == cir::CallingConv::SpirKernel)
        deviceKernel.try_emplace(kernel.getSymName(), kernel);
  SmallVector<cir::FuncOp> sinks;
  SmallVector<std::pair<cir::FuncOp, std::string>> unreached;
  for (auto &[sink, why] : result.unreached) {
    llvm::StringRef name = kernelName(sink);
    cir::FuncOp kernel = deviceKernel.lookup(name);
    if (kernel &&
        cir::isTUUniqueSYCLKernel(container, kernel, name, &locality) &&
        !cir::isRegisteredSYCLKernel(container, name)) {
      unreached.push_back({sink, why});
      continue;
    }
    sinks.push_back(sink);
  }
  if (sinks.empty() || !used)
    return;
  cir::InternalizationOptions options = closedOptions;
  options.closedRoots = false;
  options.extraRoot = [&](cir::FuncOp func) { return isOpenRoot(func, *used); };
  options.maxFunctions = closedOptions.maxFunctions > result.numCloned
                             ? closedOptions.maxFunctions - result.numCloned
                             : 0;
  options.maxOps = closedOptions.maxOps > result.numOps
                       ? closedOptions.maxOps - result.numOps
                       : 0;
  cir::InternalizationResult open =
      cir::internalizeLaunchPaths(host, sinks, options);
  // Why neither kind of root reaches a sink: the closed roots' reason, then
  // the open roots'.
  llvm::DenseMap<Operation *, std::string> closedWhy;
  for (auto &[sink, why] : result.unreached)
    closedWhy[sink] = why;
  for (auto &[sink, why] : open.unreached)
    unreached.push_back(
        {sink, closedWhy.lookup(sink) + "; " +
                   (StringRef(why).starts_with("no root reaches it")
                        ? std::string("no open root (an externally visible "
                                      "function nothing in this TU uses) "
                                      "reaches it either")
                        : why)});
  result.unreached = std::move(unreached);
  for (cir::InternalizationResult::Sink &sink : open.sinks)
    result.sinks.push_back(std::move(sink));
  for (cir::FuncOp sink : open.skipped)
    result.skipped.push_back(sink);
  if (result.budgetWhy.empty())
    result.budgetWhy = open.budgetWhy;
  for (auto &[original, copy] : open.copies)
    result.copies.try_emplace(original, copy);
  result.numCloned += open.numCloned;
  result.numInPlace += open.numInPlace;
  result.numGuards += open.numGuards;
  result.numVTables += open.numVTables;
  result.numRedirected += open.numRedirected;
  result.numOps += open.numOps;
  result.numUnvouchedAddresses += open.numUnvouchedAddresses;
}

struct OffloadInternalizeLaunchPathsPass
    : public impl::OffloadInternalizeLaunchPathsBase<
          OffloadInternalizeLaunchPathsPass> {
  void runOnOperation() override;
};

/// The analysis remark of the pass (category cir-offload-internalize).
void noteRemark(Location loc, const llvm::Twine &text) {
  cir::emitOffloadRemark(loc, cir::OffloadRemarkKind::Analysis,
                         "InternalizeLaunchPaths",
                         cir::kInternalizeRemarkCategory, "", text);
}

void OffloadInternalizeLaunchPathsPass::runOnOperation() {
  ModuleOp container = getOperation();
  std::optional<cir::HostFactsMode> mode = cir::parseHostFactsMode(hostFacts);
  if (!mode) {
    container.emitError() << "unknown host-facts mode '" << hostFacts
                          << "' (expected ipo or none)";
    return signalPassFailure();
  }
  if (!cir::isOffloadContainer(container) ||
      *mode == cir::HostFactsMode::None || hostFactsClosedTU)
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
    noteRemark(
        container.getLoc(),
        "launch-path-internalization: " + llvm::Twine(numNoFact) + " kernel" +
            (numNoFact == 1 ? "" : "s") +
            " skipped: no fact is possible (no noalias candidate, scalar "
            "capture or nd_range launch)");
  if (sinks.empty())
    return markAllAnalysesPreserved();

  std::string moduleId = getUniqueModuleId(host);
  if (moduleId.empty()) {
    noteRemark(container.getLoc(),
               "launch-path-internalization: none (the host module has no "
               "strong external definition to derive a program-unique kernel "
               "name from)");
    return markAllAnalysesPreserved();
  }

  cir::InternalizationOptions options;
  options.vouchAddress = isVouchedInvokerAddress;
  options.maxFunctions = maxFunctions;
  options.maxOps = maxOps;
  // The open roots are the functions nothing used before the closed roots'
  // paths were copied.
  std::optional<llvm::DenseSet<StringAttr>> used = getUsedSymbols(host);
  cir::InternalizationResult result =
      cir::internalizeLaunchPaths(host, sinks, options);
  // Open roots: the sinks no closed root reaches.
  internalizeFromOpenRoots(container, host, options, kernelName, used, result);
  numCopies += result.numCloned;
  if (!result.skipped.empty()) {
    std::string text =
        "launch-path-internalization budget: " + result.budgetWhy +
        "; not internalized:";
    for (cir::FuncOp sink : result.skipped)
      text += (" " + kernelName(sink)).str();
    noteRemark(container.getLoc(), text);
  }

  for (auto &[sink, why] : result.unreached)
    noteRemark(sink.getLoc(), "launch-path-internalization: kernel @" +
                                  kernelName(sink) + " not internalized (" +
                                  why + ")");

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
      noteRemark(
          sink.sink.getLoc(),
          "launch-path-internalization: kernel " + kernel +
              " not specialized (its kernel info has an unexpected shape)");
      continue;
    }
    for (Operation *op : sink.calls) {
      auto call = cast<cir::CallOp>(op);
      call.setCalleeAttr(FlatSymbolRefAttr::get(copy->getter.getSymNameAttr()));
      call->setAttr(cir::kInternalizedEdgeAttr, sink.sink.getSymNameAttr());
    }
    // The other reads of the kernel's name on the paths (the handler's
    // kernel name, T26c), checked.
    numNameReadViolations +=
        copier.redirectNameReads(*copy, sink.calls, ".int");
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
    noteRemark(
        copy->deviceKernels.front().getLoc(),
        "launch-path-internalization: kernel @" + kernel + " -> @" +
            copy->name + " for " + llvm::Twine(sink.calls.size()) +
            " launch marker" + (sink.calls.size() == 1 ? "" : "s") +
            cir::describeInternalizedSYCLKernel(copy->deviceKernels.front()));
  }
  noteRemark(container.getLoc(),
             "launch-path-internalization: " + llvm::Twine(numKernels) +
                 " kernel" + (numKernels == 1 ? "" : "s") + " specialized, " +
                 llvm::Twine(result.numCloned) + " function copies (" +
                 llvm::Twine(result.numOps) + " ops), " +
                 llvm::Twine(result.numInPlace) + " clean in place, " +
                 llvm::Twine(result.numGuards) + " guarded calls, " +
                 llvm::Twine(result.numVTables) + " vtable copies; " +
                 llvm::Twine(result.skipped.size()) + " kernel" +
                 (result.skipped.size() == 1 ? "" : "s") + " over budget");
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
  noteRemark(container.getLoc(), "launch-path-internalization: kept " +
                                     llvm::Twine(used.size()) + " of " +
                                     llvm::Twine(used.size() + unused.size()) +
                                     " kernel copies, erased " +
                                     llvm::Twine(erased) + " host copies");
}

} // namespace

std::unique_ptr<Pass> mlir::createOffloadInternalizeLaunchPathsPass() {
  return std::make_unique<OffloadInternalizeLaunchPathsPass>();
}

std::unique_ptr<Pass> mlir::createOffloadPruneInternalizedLaunchPathsPass() {
  return std::make_unique<OffloadPruneInternalizedLaunchPathsPass>();
}
