//===- SYCLHostFacts.cpp - SYCL host facts from points-to -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The noalias rule (identityOfClosureField, noAliasFields).
//
// The memory of a candidate closure field at a launch is:
//  - an accessor: the `$storage` ghost node its handle in the closure links
//    to, plus the storage's `$host` memory if the buffer was built over host
//    memory (the runtime may use that memory as the storage);
//  - a pointer: the node the closure's pointer slot links to.
//
// Its identity is complete when:
//  (a) every node of it is made only by allocations of the right kind: at
//      least one origin, each of that kind, and no UnknownObj flag. A
//      storage must be made by buffer(range) or buffer(hostData, range)
//      (buffer.alloc, buffer.host_ptr), whose model gives every call a fresh
//      storage; buffer.other and accessor.other storage is not placed. A
//      pointer target must be a heap or USM allocation. Host memory may also
//      be a stack object or a global. A node with several origins (an
//      allocation function that allocates at one of two sites, such as
//      std::allocator's aligned and plain operator new) stands for objects
//      made at any of them;
//  (b) the pointer value the field holds can only be one the analysed code
//      computed. With unification, the target of a slot stands for every
//      value stored in any slot that links to it, so this holds if no slot
//      that links to it can be written by code we don't see: none of its
//      holders (the nodes with a link to it) is External or Incomplete, it is
//      not the target of a formal of a function with callers we don't see,
//      and, for a storage, none of the handles whose `$storage` reaches it is
//      External or Incomplete (code we don't see could assign the handle).
//      Status propagation makes everything reachable from an External or
//      Incomplete node External or Incomplete too, so checking the direct
//      holders covers values loaded through any chain of pointers.
//
// Two complete identities are distinct when they share no node (and no
// origin). Two fields are no-alias at a launch when both are complete and
// distinct; no-alias for the kernel when that holds at every launch.
//
// SOUNDNESS. Take two candidate fields f and g of a launch with complete,
// distinct identities. By (b), the pointer values in the closure at the
// launch are values the analysed code computed, and the analysis is sound
// for those: the runtime value of f points into an object allocated at one
// of the sites of f's node, which by (a) are all allocations, the set Sf;
// likewise Sg for g. The sets are disjoint (the nodes differ, and each site,
// an origin with its calling context, is in one node; sameMemory also checks
// the origins), so the objects are distinct allocations, whatever happened
// to their contents. In
// particular the device writes of other kernels, and of this one, do not
// matter: the device cannot create allocations or change which allocation a
// captured pointer value points into; the host fixed both values when it
// built the closure. That is why External does not reject a field here:
// applySYCLDeviceWrites makes every allocation a kernel reaches External,
// which says the device may store pointers to it and into it, not that the
// values the host captured are unknown. Contents written by code we don't see
// are what (b) excludes. For an accessor, the memory the device accesses is
// its buffer's storage or, zero-copy, the host memory the buffer was built
// over (buffer(hostData)); both are in the identity, and two buffers over the
// same host memory share it (the spec makes that UB while both live, but this
// rule does not rely on it). A summary node (an allocation in a loop) is
// still one site: two fields bound to one summary node share it and are not
// distinct.
//
// Every launch must be known. The launches of a kernel are the LaunchRecords
// that name it. A record without kernel names may launch any kernel, so it
// rejects every kernel; a record without a closure rejects the kernel it
// names; a kernel.info marker in an analysed function that does not feed a
// recorded launch is a launch path no model records, and rejects the kernel
// it names. This matches the legacy trace, which folds every launch it
// recovers in user code and rejects a kernel with a user-code launch whose
// closure it does not recover; like it, the launches are this TU's: the
// consumer specializes a kernel by clone unless it is TU-local
// (KernelBinding::isTULocalOptimizable). Facts that depend on the callers of
// an externally callable function are Incomplete unless closed-TU is
// assumed (D2), and (b) rejects them.
//
// The constant rule (constantField). A candidate is a scalar closure field
// a device kernel takes as a scalar parameter (cir.sycl_kernel_param kind
// "scalar", a top-level int, float or bool member: the legacy fold's
// candidates). At a launch, its value is what the stored-value queries say
// the field's cell of the launch's closure holds (StoredValueQuery::valuesAt
// over the launch record's cell: in BU mode the record in the graph of an
// uncovered caller whose copies cover every execution, as for noalias); it
// is a constant if that answer is complete with exactly one constant and no
// def. The kernel's constant is the one every launch gives: every launch
// known (as for noalias), each with a closure, all giving the same
// constant.
//
// SOUNDNESS. The runtime copies the closure when the launch call runs (PI
// 17660-17672), and the kernel's parameter is initialized from that copy;
// the launch models read the closure and write none of it (they declare
// their writes, SYCLPointsToModels.cpp). The stored-value answer holds for
// the cell at every point of the program (flow-insensitive, StoredValues.cpp),
// so in particular at the launch: a complete answer with one constant c
// means every value the field holds when the closure is copied is c. A
// covering record stands for the launch in every context that executes it
// (computeCoveringLaunches), and each such record is required to give c, so
// every execution of every launch passes c. Unknown writers (code we don't
// see, External or Incomplete nodes, callers we don't see) make the answer
// incomplete; open mode therefore proves only what the module alone fixes.
//
// The geometry rule (geometry). The launch models record the geometry
// operand of a launch (LaunchRecord::geometry): a range or nd_range the
// launch function takes by value, or the detail::nd_range_view the runtime
// glue takes over one. Like the legacy trace (traceNDRange), the global size
// is the range, or the nd_range's globalSize (the view's MGlobalSize), the
// local size the nd_range's localSize (the view's MLocalSize; none for a
// range, or a view whose MLocalSize is only ever null), and the offset is
// not read. Each component is a constant if the stored-value queries prove
// one for its cell; through a view, if the pointer slot only holds addresses
// of objects (never null) that the analysis sees completely, and the cell it
// points to holds one constant. The dimensions are the type's (the view's
// MDims, if constant). The kernel's geometry is the AND over its launches:
// every launch known, each with a geometry; a component is known if every
// launch gives the same constant, and a local size is known only if every
// launch has one. A launch that may run one of several kernels (the
// range-rounding __pf_kernel_wrapper of a range launch) gives no global
// size: the runtime may round it up. A launch whose geometry operand is one
// of several ranges (parallel_for_work_group) has none recorded.
//
// SOUNDNESS. The same argument as for the constants, for the geometry
// object, which the launch functions take by value (the caller's object is
// only read; the models declare so) and the runtime reads when the call
// runs: every execution of every launch uses the agreed constants.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLHostFacts.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/CIR/Dialect/Analysis/CIRPointerDecomposition.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToAnalysis.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToModels.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/StoredValues.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLKernelParams.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLPointsToModels.h"
#include "clang/CIR/Interfaces/CIROpInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace cir;
using namespace cir::pointsto;

namespace {

StringRef getRole(cir::FuncOp func) {
  if (!func)
    return {};
  auto role =
      func->getAttrOfType<StringAttr>(CIRDialect::getSYCLHostAPIAttrName());
  return role ? role.getValue() : StringRef();
}

cir::FuncOp getDirectCallee(Operation *op, SymbolTableCollection &symbols) {
  auto call = dyn_cast_if_present<CIRCallOpInterface>(op);
  if (!call || call.isIndirect())
    return {};
  auto sym = op->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!sym)
    return {};
  return symbols.lookupNearestSymbolFrom<cir::FuncOp>(op, sym);
}

/// "file:line" of \p op, or "in @func".
std::string formatLocation(Operation *op) {
  if (!op)
    return "?";
  if (auto loc = dyn_cast<FileLineColLoc>(op->getLoc()))
    return (llvm::sys::path::filename(loc.getFilename().getValue()) + ":" +
            Twine(loc.getLine()))
        .str();
  if (auto func = op->getParentOfType<cir::FuncOp>())
    return ("in @" + func.getSymName()).str();
  return "?";
}

/// The name of the alloca \p value addresses, through casts, or empty.
StringRef allocaName(Value value) {
  for (unsigned i = 0; i < 8 && value; ++i) {
    if (auto alloca = value.getDefiningOp<cir::AllocaOp>())
      return alloca.getName();
    auto cast = value.getDefiningOp<cir::CastOp>();
    if (!cast)
      break;
    value = cast.getSrc();
  }
  return {};
}

bool isUnknownWritable(const Node *node) {
  // MayBeShared (BU): a node that may be a global's object, which any
  // function may write through the global (only set on nodes that are not
  // shared themselves).
  return hasAny(node->getFlags(), NodeFlags::External | NodeFlags::Incomplete |
                                      NodeFlags::MayBeShared);
}

/// The External (else Incomplete, else MayBeShared) reason of \p node, as
/// explain() words it.
std::string explainStatus(const Node *node) {
  const Reason *why = node->getReason(NodeFlags::External);
  if (!why)
    why = node->getReason(NodeFlags::Incomplete);
  if (!why)
    why = node->getReason(NodeFlags::MayBeShared);
  return explain(why);
}

/// " [<n> objects merged into one node]" for a node with several origins
/// (N17: a merged node's first reason alone is arbitrary).
std::string mergedSuffix(const Node *node) {
  size_t n = node->getOrigins().size();
  return n > 1 ? (" [" + Twine(n) + " objects merged into one node]").str()
               : std::string();
}

} // namespace

SYCLHostFacts::SYCLHostFacts(ModuleOp container,
                             const HostFactsOptions &options) {
  ModuleOp host = getOffloadContainerHostModule(container);
  ownedRegistry = std::make_unique<PointsToModelRegistry>();
  registerGenericModels(*ownedRegistry);
  PointsToAnalysisOptions ptOptions;
  ptOptions.assumeClosedTU = options.assumeClosedTU;
  if (options.bottomUp)
    ptOptions.mode = pointsto::PointsToMode::BU;
  ptOptions.subobjects = options.subobjects;
  ptOptions.budgetScale = options.budgetScale;
  ptOptions.relevanceSlice = options.relevanceSlice;
  ptOptions.maxSeconds = options.maxSeconds;
  ptOptions.maxMemoryMB = options.maxMemoryMB;
  addSYCLPointsToPlugin(ptOptions);
  ownedAnalysis =
      std::make_unique<PointsToAnalysis>(host, ownedRegistry.get(), ptOptions);
  // The compile-time guard (T19): a bottom-up analysis over its time or
  // memory limit falls back to the context-insensitive one, which is
  // cheaper; one over the limit again gives no facts at all (every query
  // answers "not known", see unknownLaunchWhy).
  if (const pointsto::Reason *why = ownedAnalysis->getGaveUpReason()) {
    std::string detail = why->detail.str();
    if (ptOptions.mode == pointsto::PointsToMode::BU) {
      mlir::emitRemark(container.getLoc())
          << "host-facts: the bottom-up points-to analysis gave up: " << detail
          << "; using the context-insensitive analysis";
      ptOptions.mode = pointsto::PointsToMode::CI;
      ptOptions.relevanceSlice = false;
      ownedAnalysis = std::make_unique<PointsToAnalysis>(
          host, ownedRegistry.get(), ptOptions);
      why = ownedAnalysis->getGaveUpReason();
      detail = why ? why->detail.str() : "";
    }
    if (why) {
      gaveUp = "the points-to analysis gave up: " + detail;
      mlir::emitRemark(container.getLoc())
          << "host-facts: " << gaveUp << "; no host facts";
    }
  }
  analysis = ownedAnalysis.get();
  for (ModuleOp device : getOffloadContainerDeviceModules(container))
    collectDeviceKernels(device);
  buildIndex();
}

SYCLHostFacts::SYCLHostFacts(const PointsToAnalysis &analysis,
                             Operation *deviceScope)
    : analysis(&analysis) {
  collectDeviceKernels(deviceScope);
  buildIndex();
}

SYCLHostFacts::~SYCLHostFacts() = default;

std::optional<std::string> SYCLHostFacts::takeBudgetReport() const {
  std::string trips = analysis->formatBudgetTrips();
  if (trips.empty() || trips == reportedBudget)
    return std::nullopt;
  reportedBudget = trips;
  return trips;
}

//===----------------------------------------------------------------------===//
// SYCLHostFactsCache
//===----------------------------------------------------------------------===//

SYCLHostFactsCache::SYCLHostFactsCache(Operation *container)
    : container(cast<ModuleOp>(container)) {}

SYCLHostFactsCache::~SYCLHostFactsCache() = default;

const SYCLHostFacts &SYCLHostFactsCache::get(const HostFactsOptions &options) {
  std::unique_ptr<SYCLHostFacts> &entry =
      facts[getHostFactsAnalysisKey(options)];
  if (!entry) {
    entry = std::make_unique<SYCLHostFacts>(container, options);
    ++numBuilt;
  }
  return *entry;
}

void SYCLHostFactsCache::recordSpecialization(StringRef clone,
                                              StringRef original) {
  origins[clone] = getOriginalKernel(original).str();
}

StringRef SYCLHostFactsCache::getOriginalKernel(StringRef kernel) const {
  auto it = origins.find(kernel);
  return it == origins.end() ? kernel : StringRef(it->second);
}

void SYCLHostFacts::collectDeviceKernels(Operation *scope) {
  // The device kernels: functions with a body whose arguments carry the
  // closure-field mapping, or with the spir_kernel convention.
  scope->walk([&](cir::FuncOp func) {
    if (func.isDeclaration())
      return;
    bool isKernel = func.getCallingConv() == cir::CallingConv::SpirKernel;
    for (unsigned i = 0, e = func.getNumArguments(); i != e && !isKernel; ++i)
      isKernel = getSYCLKernelParam(func, i).has_value();
    if (isKernel)
      deviceKernels[func.getSymName()].push_back(func);
  });
}

void SYCLHostFacts::buildIndex() {
  ModuleOp module = analysis->getModule();
  layout = std::make_unique<DataLayout>(module);
  closureLayout = std::make_unique<SYCLClosureLayout>(module);

  // The holders of each node, and the handles of each storage.
  const Graph &graph = analysis->getGraph();
  GhostField storageField = getSYCLStorageField();
  for (const Node *node : graph.nodes()) {
    for (const auto &link : node->getLinks())
      holders[link.second.getNode()].push_back(node);
    for (const GhostLink &ghost : node->getGhostLinks())
      if (ghost.field == storageField)
        handles[ghost.target.getNode()].push_back(node);
  }

  // What callers we don't see may pass.
  for (const auto &fn : analysis->getFunctions()) {
    if (fn->formalsComplete)
      continue;
    Operation *func = fn->func.getOperation();
    for (const std::optional<Cell> &formal : fn->formals)
      if (formal)
        formalTargets.try_emplace(formal->getNode(), func);
    for (const std::optional<AggregateSlots> &slots : fn->formalSlots)
      if (slots)
        for (const auto &slot : *slots)
          formalTargets.try_emplace(slot.second.getNode(), func);
  }

  // The launches no model records.
  llvm::DenseSet<Operation *> launchCalls;
  for (const LaunchRecord &launch : analysis->getLaunches()) {
    launchCalls.insert(launch.call);
    if (launch.kernelNames.empty() && !unnamedLaunch)
      unnamedLaunch = launch.call;
  }
  module.walk([&](CIRCallOpInterface call) {
    cir::FuncOp callee = getDirectCallee(call, symbols);
    if (getRole(callee) != "kernel.info")
      return;
    auto func = call->getParentOfType<cir::FuncOp>();
    if (!func || !analysis->getFunction(func))
      return; // Not analysed: inside a body a model replaced.
    bool feedsLaunch = llvm::any_of(call->getUsers(), [&](Operation *user) {
      return launchCalls.contains(user);
    });
    if (feedsLaunch)
      return;
    std::string name = getSYCLKernelInfoName(callee, &symbols);
    if (!name.empty())
      unrecordedMarkers.try_emplace(name, call.getOperation());
  });
}

ArrayRef<cir::FuncOp> SYCLHostFacts::getDeviceKernels(StringRef kernel) const {
  auto it = deviceKernels.find(kernel);
  return it == deviceKernels.end() ? ArrayRef<cir::FuncOp>()
                                   : ArrayRef<cir::FuncOp>(it->second);
}

std::string SYCLHostFacts::describeAllocation(const Node *node) const {
  std::string id = ("n" + Twine(node->getId())).str();
  if (node->getOrigins().size() != 1)
    return "memory " + id + mergedSuffix(node);
  const Origin &origin = node->getOrigins().front();
  switch (origin.kind) {
  case Origin::Kind::GhostSite: {
    cir::FuncOp callee = getDirectCallee(origin.site, symbols);
    StringRef name;
    if (auto call = dyn_cast_if_present<CIRCallOpInterface>(origin.site)) {
      unsigned self = callee && callee.getNumArguments() &&
                              callee.getArgAttr(0, "llvm.sret")
                          ? 1
                          : 0;
      if (self < call.getNumArgOperands())
        name = allocaName(call.getArgOperand(self));
    }
    std::string buffer = name.empty() ? std::string("a buffer")
                                      : ("buffer '" + name + "'").str();
    return buffer + " (ghost " + id + ")";
  }
  case Origin::Kind::HeapCall:
    return (hasAny(node->getFlags(), NodeFlags::HasUSM)
                ? "a USM allocation ("
                : "a heap allocation (") +
           id + ", " + formatLocation(origin.site) + ")";
  case Origin::Kind::Alloca: {
    StringRef name;
    if (auto alloca = dyn_cast_if_present<cir::AllocaOp>(origin.site))
      name = alloca.getName();
    return ("local '" + name + "' (" + id + ")").str();
  }
  case Origin::Kind::GlobalSym:
    return ("global " +
            Twine(origin.sym ? origin.sym.getLeafReference().getValue()
                             : StringRef("?")) +
            " (" + id + ")")
        .str();
  case Origin::Kind::FuncSym:
    return "a function (" + id + ")";
  }
  return id;
}

std::optional<std::string> SYCLHostFacts::notOneAllocationWhy(const Node *node,
                                                              bool storage,
                                                              bool host) const {
  std::string id = ("n" + Twine(node->getId())).str();
  if (hasAny(node->getFlags(), NodeFlags::UnknownObj))
    return id + " may be memory that code we don't see allocated: " +
           explain(node->getReason(NodeFlags::External)
                       ? node->getReason(NodeFlags::External)
                       : node->getReason(NodeFlags::Incomplete)) +
           mergedSuffix(node);
  ArrayRef<Origin> origins = node->getOrigins();
  if (origins.empty())
    return id + " has no allocation site the analysis sees";
  // Each origin must be an allocation of the kind the field needs. Several
  // are fine (rule (a)): an allocation function that allocates at one of two
  // sites, for instance std::allocator's aligned and plain operator new.
  auto originWhy = [&](const Origin &origin) -> std::optional<std::string> {
    if (storage) {
      StringRef role = getRole(getDirectCallee(origin.site, symbols));
      if (origin.kind != Origin::Kind::GhostSite ||
          (role != "buffer.alloc" && role != "buffer.host_ptr"))
        return ("the storage " + id + " is made by " +
                (role.empty() ? StringRef("no buffer constructor") : role) +
                ", which the analysis cannot place")
            .str();
      return std::nullopt;
    }
    if (origin.kind == Origin::Kind::HeapCall)
      return std::nullopt;
    if (host && (origin.kind == Origin::Kind::Alloca ||
                 origin.kind == Origin::Kind::GlobalSym))
      return std::nullopt;
    return id + " is not a heap or USM allocation";
  };
  for (const Origin &origin : origins) {
    std::optional<std::string> why = originWhy(origin);
    if (!why)
      continue;
    if (origins.size() == 1)
      return why;
    std::string out =
        id + " merges " + std::to_string(origins.size()) + " allocations (";
    for (auto [i, origin] : llvm::enumerate(origins.take_front(3))) {
      if (i)
        out += ", ";
      out += formatLocation(origin.site);
    }
    if (origins.size() > 3)
      out += ", ...";
    return out + "), not all of which it can place: " + *why;
  }
  return std::nullopt;
}

std::optional<std::string> SYCLHostFacts::unknownValueWhy(const Node *node,
                                                          bool storage) const {
  std::string id = ("n" + Twine(node->getId())).str();
  if (node->hasFlags(NodeFlags::MayBeShared) &&
      !analysis->getGraph().isShared(node))
    return id + " may be a global's object: " +
           explain(node->getReason(NodeFlags::MayBeShared));
  if (auto it = formalTargets.find(node); it != formalTargets.end())
    return id + " may be passed in by a caller of @" +
           cast<cir::FuncOp>(it->second).getSymName().str() +
           " that is not analysed";
  if (auto it = holders.find(node); it != holders.end())
    for (const Node *holder : it->second)
      if (isUnknownWritable(holder))
        return "a pointer to " + id + " is held in n" +
               std::to_string(holder->getId()) + mergedSuffix(holder) +
               ", which code we don't see may write: " + explainStatus(holder);
  if (storage)
    if (auto it = handles.find(node); it != handles.end())
      for (const Node *handle : it->second)
        if (isUnknownWritable(handle))
          return "a handle of storage " + id + " (n" +
                 std::to_string(handle->getId()) + mergedSuffix(handle) +
                 ") may be assigned by code we don't see: " +
                 explainStatus(handle);
  return std::nullopt;
}

SYCLHostFacts::MemIdentity
SYCLHostFacts::identityOfClosureField(const LaunchRecord &launch,
                                      StringRef field) const {
  MemIdentity id;
  auto fail = [&](const Twine &why) {
    id.complete = false;
    id.what = why.str();
    return id;
  };
  if (!launch.closure || !launch.closureType)
    return fail("the launch's closure is not visible");
  MLIRContext *ctx = launch.call->getContext();
  ArrayAttr path = ArrayAttr::get(ctx, {StringAttr::get(ctx, field)});
  auto member = closureLayout->getField(launch.closureType, path, *layout);
  if (!member)
    return fail("'" + field + "' is not found in the launch's closure");
  const Graph &graph = analysis->getGraph();
  Cell at(launch.closure->getNode(),
          launch.closure->getOffset() + member->first);

  if (isa<cir::PointerType>(member->second)) {
    id.kind = MemIdentity::Kind::Pointer;
    std::optional<Cell> target = graph.lookupLink(at);
    if (!target)
      return fail("no pointer the analysis sees is stored in '" + field + "'");
    id.node = target->getNode();
    if (std::optional<std::string> why =
            notOneAllocationWhy(id.node, /*storage=*/false, /*host=*/false))
      return fail("pointer '" + field + "': " + *why);
    if (std::optional<std::string> why =
            unknownValueWhy(id.node, /*storage=*/false))
      return fail("pointer '" + field + "': " + *why);
    id.complete = true;
    id.what = describeAllocation(id.node);
    return id;
  }

  id.kind = MemIdentity::Kind::Storage;
  // N22: a handle without a `$storage` ghost was not built by a model; its
  // storage is unknown, never distinct.
  std::optional<Cell> storage =
      graph.lookupGhostLink(at, getSYCLStorageField());
  if (!storage)
    return fail("accessor '" + field + "' has no storage a model made");
  id.node = storage->getNode();
  if (std::optional<std::string> why =
          notOneAllocationWhy(id.node, /*storage=*/true, /*host=*/false))
    return fail("accessor '" + field + "': " + *why);
  if (std::optional<std::string> why =
          unknownValueWhy(id.node, /*storage=*/true))
    return fail("accessor '" + field + "': " + *why);
  StringRef role =
      getRole(getDirectCallee(id.node->getOrigins()[0].site, symbols));
  std::optional<Cell> host =
      graph.lookupGhostLink(*storage, getSYCLHostField());
  if (role == "buffer.host_ptr" && !host)
    return fail("accessor '" + field + "': the host memory of " +
                describeAllocation(id.node) + " is not known");
  id.what = describeAllocation(id.node);
  if (host) {
    id.host = host->getNode();
    if (std::optional<std::string> why =
            notOneAllocationWhy(id.host, /*storage=*/false, /*host=*/true))
      return fail("accessor '" + field + "': the host memory of " + id.what +
                  ": " + *why);
    if (std::optional<std::string> why =
            unknownValueWhy(id.host, /*storage=*/false))
      return fail("accessor '" + field + "': the host memory of " + id.what +
                  ": " + *why);
    id.what += " over " + describeAllocation(id.host);
  }
  id.complete = true;
  return id;
}

bool SYCLHostFacts::sameMemory(const MemIdentity &a, const MemIdentity &b) {
  for (const Node *x : {a.node, a.host})
    for (const Node *y : {b.node, b.host}) {
      if (!x || !y)
        continue;
      if (x == y)
        return true;
      // Complete identities: a shared origin is a shared site.
      for (const Origin &origin : x->getOrigins())
        if (llvm::is_contained(y->getOrigins(), origin))
          return true;
    }
  return false;
}

SYCLHostFacts::LaunchFacts
SYCLHostFacts::factsOfLaunch(const LaunchRecord &launch,
                             const SYCLNoAliasCandidates &candidates) const {
  LaunchFacts facts;
  SmallVector<StringRef, 4> names;
  for (const auto &c : candidates.fields)
    names.push_back(c.first());
  llvm::sort(names);
  for (StringRef name : names) {
    MemIdentity id = identityOfClosureField(launch, name);
    bool wantStorage = candidates.fields.lookup(name);
    if (id.complete && (id.kind == MemIdentity::Kind::Storage) != wantStorage) {
      id.complete = false;
      id.what = ("closure member '" + name +
                 "' is not the kind its kernel parameter says")
                    .str();
    }
    facts.fields.emplace_back(name.str(), std::move(id));
  }
  for (auto [i, entry] : llvm::enumerate(facts.fields)) {
    const auto &[name, id] = entry;
    NoAliasDecision &d = facts.decisions[name];
    if (!candidates.poison.empty()) {
      d.note = candidates.poison;
      continue;
    }
    if (!id.complete) {
      d.note = id.what;
      continue;
    }
    d.ok = true;
    d.note = id.what;
    for (auto [j, other] : llvm::enumerate(facts.fields)) {
      if (i == j)
        continue;
      if (!other.second.complete) {
        d = {false, "'" + other.first + "' may alias it: " + other.second.what};
        break;
      }
      if (sameMemory(id, other.second)) {
        d = {false, "'" + name + "' and '" + other.first +
                        "' use the same memory, " + id.what};
        break;
      }
    }
  }
  return facts;
}

std::optional<std::string>
SYCLHostFacts::unknownLaunchWhy(StringRef kernel) const {
  if (!gaveUp.empty())
    return gaveUp;
  if (unnamedLaunch)
    return "the launch at " + formatLocation(unnamedLaunch) +
           " may launch any kernel: its kernel is not known";
  if (auto it = unrecordedMarkers.find(kernel); it != unrecordedMarkers.end())
    return "the kernel is named at " + formatLocation(it->second) +
           " by a launch path the points-to models do not record";
  return std::nullopt;
}

llvm::StringMap<NoAliasDecision>
SYCLHostFacts::noAliasFields(StringRef kernel) const {
  return noAliasFields(kernel,
                       getSYCLNoAliasCandidates(getDeviceKernels(kernel)));
}

llvm::StringMap<NoAliasDecision>
SYCLHostFacts::noAliasFields(StringRef kernel,
                             const SYCLNoAliasCandidates &candidates) const {
  llvm::StringMap<NoAliasDecision> decisions;
  auto rejectAll = [&](const std::string &why) {
    for (const auto &c : candidates.fields)
      decisions[c.first()] = {false, why};
    return decisions;
  };
  if (candidates.fields.empty())
    return decisions;
  if (std::optional<std::string> why = unknownLaunchWhy(kernel))
    return rejectAll(*why);
  bool seen = false;
  std::optional<std::string> unrecovered;
  for (const LaunchRecord &launch : analysis->getLaunches()) {
    if (!llvm::any_of(launch.kernelNames, [&](StringAttr name) {
          return name.getValue() == kernel;
        }))
      continue;
    if (!launch.closure) {
      if (!unrecovered)
        unrecovered = "the launch at " + formatLocation(launch.call) +
                      " has no closure the analysis sees";
      continue;
    }
    andNoAliasDecisions(factsOfLaunch(launch, candidates).decisions, seen,
                        decisions);
  }
  if (!seen)
    return rejectAll("no launch of the kernel is recorded by the points-to "
                     "models");
  if (unrecovered)
    for (auto &d : decisions)
      if (d.second.ok)
        d.second = {false, *unrecovered};
  return decisions;
}

//===----------------------------------------------------------------------===//
// Constants and geometry (the stored-value queries)
//===----------------------------------------------------------------------===//

namespace {

bool isScalarType(Type type) {
  return isa_and_present<cir::IntType, cir::FPTypeInterface, cir::BoolType>(
      type);
}

/// The value of an integer constant, as a signed number.
int64_t asInt64(cir::IntAttr value) {
  auto type = dyn_cast<cir::IntType>(value.getType());
  return type && !type.isSigned()
             ? static_cast<int64_t>(value.getValue().getZExtValue())
             : value.getValue().getSExtValue();
}

/// A constant for the remarks.
std::string formatConst(TypedAttr value) {
  if (!value)
    return "none";
  std::string out;
  llvm::raw_string_ostream os(out);
  if (auto i = dyn_cast<cir::IntAttr>(value))
    os << asInt64(i);
  else
    value.print(os);
  return out;
}

/// Whether \p value is the address of an object (an alloca, a global, a
/// member of one): a pointer that is never null, through bit casts and
/// array decay.
bool isNonNullAddress(Value value) {
  for (unsigned i = 0; i < 16 && value; ++i) {
    Operation *def = value.getDefiningOp();
    if (isa_and_present<cir::AllocaOp, cir::GetGlobalOp, cir::GetMemberOp>(def))
      return true;
    auto cast = dyn_cast_if_present<cir::CastOp>(def);
    if (!cast || (cast.getKind() != cir::CastKind::bitcast &&
                  cast.getKind() != cir::CastKind::array_to_ptrdecay))
      return false;
    value = cast.getSrc();
  }
  return false;
}

/// The shape of a sycl::range<D> or sycl::id<D> record: D, and the type and
/// size of its elements (a detail::array<D> holding an array of D
/// integers).
struct RangeShape {
  unsigned dims = 0;
  Type elem;
  uint64_t elemSize = 0;
};

std::optional<RangeShape> getRangeShape(Type type, const DataLayout &layout) {
  for (unsigned depth = 0; depth < 4; ++depth) {
    if (auto array = dyn_cast_if_present<cir::ArrayType>(type)) {
      auto elem = dyn_cast<cir::IntType>(array.getElementType());
      if (!elem || array.getSize() < 1 || array.getSize() > 3)
        return std::nullopt;
      return RangeShape{static_cast<unsigned>(array.getSize()), elem,
                        layout.getTypeSize(elem).getFixedValue()};
    }
    auto rec = dyn_cast_if_present<cir::RecordType>(type);
    if (!rec || rec.isIncomplete() || rec.isUnion() ||
        rec.getNumElements() != 1)
      return std::nullopt;
    type = rec.getElementType(0);
  }
  return std::nullopt;
}

} // namespace

bool SYCLHostFacts::isClosureOf(const LaunchRecord &launch,
                                ArrayRef<cir::FuncOp> kernels) const {
  if (!launch.closureType || kernels.empty())
    return false;
  MLIRContext *ctx = launch.call->getContext();
  for (cir::FuncOp device : kernels)
    for (unsigned i = 0, e = device.getNumArguments(); i != e; ++i) {
      std::optional<SYCLKernelParam> param = getSYCLKernelParam(device, i);
      if (!param)
        continue;
      ArrayAttr path =
          ArrayAttr::get(ctx, {StringAttr::get(ctx, param->field)});
      if (!closureLayout->getField(launch.closureType, path, *layout))
        return false;
    }
  return true;
}

llvm::StringMap<Type>
SYCLHostFacts::getConstCandidates(StringRef kernel) const {
  llvm::StringMap<Type> out;
  for (cir::FuncOp device : getDeviceKernels(kernel))
    for (unsigned i = 0, e = device.getNumArguments(); i != e; ++i) {
      Type type = device.getArgumentTypes()[i];
      if (!isScalarType(type))
        continue;
      StringRef field = getSYCLCapturedField(device, i, /*wantPointer=*/false);
      if (!field.empty())
        out.try_emplace(field, type);
    }
  return out;
}

SYCLHostFacts::ConstFact SYCLHostFacts::constantAt(Cell cell, uint64_t size,
                                                   Type type) const {
  StoredValues values =
      analysis->getStoredValueQuery().valuesAt(cell, size, nullptr, type);
  if (!values.complete)
    return {nullptr, "it may hold a value the analysis does not know: " +
                         explain(values.why)};
  if (!values.defs.empty()) {
    Value def = values.defs.front();
    std::string why =
        "it holds a value computed at run time (" +
        formatLocation(def.getDefiningOp()
                           ? def.getDefiningOp()
                           : def.getParentRegion()->getParentOp()) +
        ")";
    if (values.defWhy)
      why += ": " + explain(values.defWhy);
    return {nullptr, why};
  }
  if (values.consts.empty())
    return {nullptr, "it is never written"};
  if (values.consts.size() > 1)
    return {nullptr, "it may hold " + std::to_string(values.consts.size()) +
                         " constants (" + formatConst(values.consts[0]) + ", " +
                         formatConst(values.consts[1]) + ")"};
  TypedAttr c = values.consts.front();
  if (!isa<cir::IntAttr, cir::FPAttr, cir::BoolAttr>(c))
    return {nullptr, "it holds a constant that is not a scalar"};
  return {c, "constant " + formatConst(c)};
}

SYCLHostFacts::ConstFact
SYCLHostFacts::constantOfLaunch(const LaunchRecord &launch, StringRef field,
                                Type type) const {
  if (!launch.closure || !launch.closureType)
    return {nullptr, "the launch's closure is not visible"};
  MLIRContext *ctx = launch.call->getContext();
  ArrayAttr path = ArrayAttr::get(ctx, {StringAttr::get(ctx, field)});
  auto member = closureLayout->getField(launch.closureType, path, *layout);
  if (!member)
    return {nullptr,
            ("'" + field + "' is not found in the launch's closure").str()};
  if (!isScalarType(member->second))
    return {nullptr, ("'" + field + "' is not a scalar member").str()};
  std::optional<int64_t> size = getTypeSizeInBytes(member->second, *layout);
  if (!size || *size <= 0)
    return {nullptr, ("'" + field + "' has no size").str()};
  Cell at(launch.closure->getNode(),
          launch.closure->getOffset() + member->first);
  ConstFact fact = constantAt(at, static_cast<uint64_t>(*size),
                              type ? type : member->second);
  if (!fact.value)
    fact.note = ("'" + field + "' at the launch at " +
                 formatLocation(launch.call) + ": " + fact.note)
                    .str();
  return fact;
}

SmallVector<const LaunchRecord *, 2>
SYCLHostFacts::launchesOf(StringRef kernel,
                          std::optional<std::string> &why) const {
  SmallVector<const LaunchRecord *, 2> out;
  why = unknownLaunchWhy(kernel);
  for (const LaunchRecord &launch : analysis->getLaunches()) {
    if (!llvm::any_of(launch.kernelNames, [&](StringAttr name) {
          return name.getValue() == kernel;
        }))
      continue;
    if (!launch.closure && !why)
      why = "the launch at " + formatLocation(launch.call) +
            " has no closure the analysis sees";
    out.push_back(&launch);
  }
  if (out.empty() && !why)
    why = "no launch of the kernel is recorded by the points-to models";
  return out;
}

SYCLHostFacts::ConstFact SYCLHostFacts::constantField(StringRef kernel,
                                                      StringRef field) const {
  llvm::StringMap<Type> candidates = getConstCandidates(kernel);
  auto it = candidates.find(field);
  if (it == candidates.end())
    return {nullptr,
            ("'" + field + "' is not a scalar kernel parameter").str()};
  std::optional<std::string> why;
  SmallVector<const LaunchRecord *, 2> launches = launchesOf(kernel, why);
  if (why)
    return {nullptr, *why};
  ConstFact result;
  const LaunchRecord *first = nullptr;
  for (const LaunchRecord *launch : launches) {
    // A launch that may run one of several kernels (a range launch, which may
    // run the range-rounding wrapper, whose closure wraps the user's): the
    // closure the record names must be this kernel's, every top-level field
    // its parameters name a member of it.
    if (launch->kernelNames.size() > 1 &&
        !isClosureOf(*launch, getDeviceKernels(kernel)))
      return {nullptr, "the launch at " + formatLocation(launch->call) +
                           " may run one of several kernels, and its "
                           "closure is not this kernel's"};
    ConstFact here = constantOfLaunch(*launch, field, it->second);
    if (!here.value)
      return here;
    if (!first) {
      result = here;
      first = launch;
      continue;
    }
    if (here.value != result.value)
      return {nullptr,
              ("the launches disagree: " + formatConst(result.value) + " at " +
               formatLocation(first->call) + ", " + formatConst(here.value) +
               " at " + formatLocation(launch->call))};
  }
  result.note = "constant " + formatConst(result.value) + " at " +
                std::to_string(launches.size()) +
                (launches.size() == 1 ? " launch" : " launches");
  return result;
}

llvm::StringMap<SYCLHostFacts::ConstFact>
SYCLHostFacts::constantFields(StringRef kernel) const {
  llvm::StringMap<ConstFact> out;
  for (const auto &candidate : getConstCandidates(kernel))
    out[candidate.first()] = constantField(kernel, candidate.first());
  return out;
}

SYCLHostFacts::GeometryFact
SYCLHostFacts::geometryOfLaunch(const LaunchRecord &launch) const {
  GeometryFact fact;
  SyclGeometry &g = fact.geometry;
  auto fail = [&](const Twine &why) {
    fact.geometry = SyclGeometry();
    fact.note = why.str();
    return fact;
  };
  auto rec = dyn_cast_if_present<cir::RecordType>(launch.geometryType);
  if (!launch.geometry || !rec || rec.isIncomplete())
    return fail("the launch at " + formatLocation(launch.call) +
                " passes no geometry object the models record");
  StringRef name = rec.getName() ? rec.getName().getValue() : StringRef();
  Cell base(launch.geometry->getNode(), launch.geometry->getOffset());
  auto note = [&](const std::string &why) {
    if (fact.note.empty())
      fact.note = why;
  };
  // The components of a range at \p cell.
  auto readRange = [&](Cell cell, const RangeShape &shape,
                       SmallVectorImpl<std::optional<int64_t>> &out) {
    for (unsigned i = 0; i < shape.dims; ++i) {
      Cell at(cell.getNode(),
              cell.getOffset() + static_cast<int64_t>(i * shape.elemSize));
      ConstFact c = constantAt(at, shape.elemSize, shape.elem);
      auto value = dyn_cast_if_present<cir::IntAttr>(c.value);
      if (!value)
        note("component " + std::to_string(i) + ": " + c.note);
      out.push_back(value ? std::optional<int64_t>(asInt64(value))
                          : std::nullopt);
    }
  };
  auto memberShape = [&](unsigned index) -> std::optional<RangeShape> {
    if (index >= rec.getNumElements())
      return std::nullopt;
    return getRangeShape(rec.getElementType(index), *layout);
  };
  auto memberCell = [&](unsigned index) {
    return Cell(base.getNode(),
                base.getOffset() +
                    static_cast<int64_t>(rec.getElementOffset(*layout, index)));
  };

  if (name.contains("::nd_range_view")) {
    // {const size_t *MGlobalSize, *MLocalSize, *MOffset; size_t MDims}.
    if (rec.getNumElements() != 4 ||
        !isa<cir::PointerType>(rec.getElementType(0)) ||
        !isa<cir::PointerType>(rec.getElementType(1)) ||
        !isa<cir::IntType>(rec.getElementType(3)))
      return fail("the nd_range_view has an unexpected layout");
    auto dimsTy = cast<cir::IntType>(rec.getElementType(3));
    ConstFact dims =
        constantAt(memberCell(3), dimsTy.getWidth() / 8, rec.getElementType(3));
    auto d = dyn_cast_if_present<cir::IntAttr>(dims.value);
    if (!d || asInt64(d) < 1 || asInt64(d) > 3)
      return fail("the nd_range_view's MDims: " +
                  (d ? std::string("out of range") : dims.note));
    g.dims = static_cast<unsigned>(asInt64(d));
    // The sizes a pointer slot points to: false if it holds only null.
    auto readThrough = [&](unsigned index,
                           SmallVectorImpl<std::optional<int64_t>> &out,
                           StringRef what) -> bool {
      auto ptrTy = cast<cir::PointerType>(rec.getElementType(index));
      auto elem = dyn_cast<cir::IntType>(ptrTy.getPointee());
      Cell slot = memberCell(index);
      StoredValues values = analysis->getStoredValueQuery().valuesAt(
          slot, layout->getTypeSize(ptrTy).getFixedValue(), nullptr, ptrTy);
      bool onlyNull = values.complete && values.defs.empty() &&
                      !values.consts.empty() &&
                      llvm::all_of(values.consts, [](TypedAttr c) {
                        auto p = dyn_cast<cir::ConstPtrAttr>(c);
                        return p && p.isNullValue();
                      });
      if (onlyNull)
        return false;
      auto unknown = [&](const std::string &why) {
        note((what + ": " + why).str());
        out.assign(g.dims, std::nullopt);
        return true;
      };
      if (!elem)
        return unknown("not a pointer to an integer");
      if (!values.complete)
        return unknown("may hold a pointer the analysis does not know: " +
                       explain(values.why));
      if (!values.consts.empty() ||
          !llvm::all_of(values.defs, isNonNullAddress))
        return unknown("may be null");
      ObjectSet targets = analysis->objectsAt(slot);
      if (!targets.complete || targets.cells.size() != 1)
        return unknown("does not point to one object the analysis sees");
      RangeShape shape{g.dims, elem, layout->getTypeSize(elem).getFixedValue()};
      readRange(targets.cells.front(), shape, out);
      return true;
    };
    if (!readThrough(0, g.global, "MGlobalSize"))
      return fail("the nd_range_view's MGlobalSize is null");
    if (!readThrough(1, g.local, "MLocalSize"))
      g.local.clear();
    g.valid = true;
  } else if (name.contains("::nd_range<")) {
    // {range<D> globalSize, range<D> localSize, id<D> offset}.
    std::optional<RangeShape> global = memberShape(0), local = memberShape(1);
    if (rec.getNumElements() != 3 || !global || !local ||
        global->dims != local->dims)
      return fail("the nd_range has an unexpected layout");
    g.dims = global->dims;
    readRange(memberCell(0), *global, g.global);
    readRange(memberCell(1), *local, g.local);
    g.valid = true;
  } else if (name.contains("::range<")) {
    std::optional<RangeShape> shape = getRangeShape(rec, *layout);
    if (!shape)
      return fail("the range has an unexpected layout");
    g.dims = shape->dims;
    readRange(base, *shape, g.global);
    g.valid = true;
  } else {
    return fail("the geometry is not a range, nd_range or nd_range_view");
  }
  // SOUNDNESS: a range launch may run the range-rounding wrapper kernel
  // over a global range rounded up from the one passed.
  if (launch.kernelNames.size() > 1 &&
      llvm::any_of(g.global, [](auto c) { return c.has_value(); })) {
    g.global.assign(g.dims, std::nullopt);
    note("the launch may run one of " +
         std::to_string(launch.kernelNames.size()) +
         " kernels, which may round the global range up");
  }
  return fact;
}

SYCLHostFacts::GeometryFact SYCLHostFacts::geometry(StringRef kernel) const {
  GeometryFact result;
  auto fail = [&](const std::string &why) {
    result.geometry = SyclGeometry();
    result.note = why;
    return result;
  };
  std::optional<std::string> why;
  SmallVector<const LaunchRecord *, 2> launches = launchesOf(kernel, why);
  if (why)
    return fail(*why);
  bool first = true;
  for (const LaunchRecord *launch : launches) {
    GeometryFact here = geometryOfLaunch(*launch);
    if (!here.geometry.valid)
      return fail(here.note);
    if (result.note.empty())
      result.note = here.note;
    if (first) {
      result.geometry = here.geometry;
      first = false;
      continue;
    }
    SyclGeometry &g = result.geometry;
    if (g.dims != here.geometry.dims)
      return fail("the launches use different dimensions");
    auto meet = [&](SmallVectorImpl<std::optional<int64_t>> &into,
                    ArrayRef<std::optional<int64_t>> other, StringRef what) {
      for (auto [a, b] : llvm::zip(into, other))
        if (a != b) {
          if (a && b && result.note.empty())
            result.note = ("the launches disagree on the " + what +
                           " size at " + formatLocation(launch->call))
                              .str();
          a = std::nullopt;
        }
    };
    meet(g.global, here.geometry.global, "global");
    if (g.local.empty() != here.geometry.local.empty()) {
      // nd_range and range launches: no local size every launch has.
      if (result.note.empty())
        result.note = "some launches have no local size";
      g.local.assign(g.dims, std::nullopt);
    } else {
      meet(g.local, here.geometry.local, "local");
    }
  }
  return result;
}
