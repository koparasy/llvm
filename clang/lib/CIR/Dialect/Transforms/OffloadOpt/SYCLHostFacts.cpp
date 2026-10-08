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
//  (a) every node of it is ONE allocation: exactly one origin, no UnknownObj
//      flag. A storage must be made by buffer(range) or buffer(hostData,
//      range) (buffer.alloc, buffer.host_ptr), whose model gives every call
//      a fresh storage; buffer.other and accessor.other storage is not
//      placed. A pointer target must be a heap or USM allocation. Host
//      memory may also be a stack object or a global;
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
// for those: the runtime value of f points into an object whose allocation
// site is in f's node, and by (a) that is one site, Sf; likewise Sg for g.
// The sites differ (the nodes differ, each site is one node), so the objects
// are distinct allocations, whatever happened to their contents. In
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
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLHostFacts.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToAnalysis.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToModels.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLKernelParams.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLPointsToModels.h"
#include "clang/CIR/Interfaces/CIROpInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Path.h"

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

cir::FuncOp getDirectCallee(Operation *op) {
  auto call = dyn_cast_if_present<CIRCallOpInterface>(op);
  if (!call || call.isIndirect())
    return {};
  auto sym = op->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!sym)
    return {};
  return SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(op, sym);
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
  return hasAny(node->getFlags(), NodeFlags::External | NodeFlags::Incomplete);
}

/// The External (else Incomplete) reason of \p node, as explain() words it.
std::string explainStatus(const Node *node) {
  const Reason *why = node->getReason(NodeFlags::External);
  if (!why)
    why = node->getReason(NodeFlags::Incomplete);
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
  addSYCLPointsToPlugin(ptOptions);
  ownedAnalysis =
      std::make_unique<PointsToAnalysis>(host, ownedRegistry.get(), ptOptions);
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
    cir::FuncOp callee = getDirectCallee(call);
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
    std::string name = getSYCLKernelInfoName(callee);
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
    cir::FuncOp callee = getDirectCallee(origin.site);
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
  if (origins.size() > 1) {
    std::string out =
        id + " merges " + std::to_string(origins.size()) + " allocations (";
    for (auto [i, origin] : llvm::enumerate(origins.take_front(3))) {
      if (i)
        out += ", ";
      out += formatLocation(origin.site);
    }
    if (origins.size() > 3)
      out += ", ...";
    return out + ")";
  }
  const Origin &origin = origins.front();
  if (storage) {
    StringRef role = getRole(getDirectCallee(origin.site));
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
}

std::optional<std::string> SYCLHostFacts::unknownValueWhy(const Node *node,
                                                          bool storage) const {
  std::string id = ("n" + Twine(node->getId())).str();
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
  StringRef role = getRole(getDirectCallee(id.node->getOrigins()[0].site));
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
      // One origin each (complete identities): equal origins are one site.
      if (x->getOrigins().size() == 1 && y->getOrigins().size() == 1 &&
          x->getOrigins()[0] == y->getOrigins()[0])
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
