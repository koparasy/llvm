//===- SYCLPointsToModels.cpp - SYCL models for CIR points-to -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Trust rule (plan D1): a model attaches to a spec-defined public API
// function and relies only on what the SYCL 2020 specification says it does.
// Citations are to SYCL-Docs/adoc/chapters/programming_interface.adoc (PI)
// and architecture.adoc (ARCH). Every model replaces the function's body
// (replacesDefinition): the bodies only reach the opaque runtime ABI
// (buffer_plain, AccessorBaseHost, the handler setters, the
// submit_*_impl entry points), which the analysis could only treat as
// unknown code. Internal glue roles that only these bodies reach get no
// model; the few the test inputs reach from analysed code (hostkernel.ctor,
// launch.direct_impl, kernel.info, the handler setters and kernel.wrapper)
// get the semantics of the public API they implement, cited there.
//
// Operands are found structurally, never by mangled name: CIR call operands
// are ABI-lowered (an `sret` result slot first, then `this` for a method; a
// by-value range may be split in two, an empty class dropped), so the
// source-level `cir.sycl_host_api_arg` index is used only where nothing
// before it can change shape. A call whose operands do not have the shape a
// model expects gets the unknown-call rule.
//
// Effects common to every model:
//  - nothing escapes: the runtime keeps no pointer to user memory that the
//    analysis does not model (the storage it may use is the `$host` ghost);
//  - the handle objects a model constructs or assigns get Mod, the others
//    Ref; the pointer slots of a constructed handle (its impl pointers) point
//    to fresh unknown objects, External: the runtime owns what they point to;
//  - a model whose function returns a runtime object (an event) gives its
//    result slot the same treatment.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLPointsToModels.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToModels.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Interfaces/CIROpInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace cir;
using namespace cir::pointsto;

GhostField cir::getSYCLStorageField() {
  static GhostField field = GhostField::get("$storage");
  return field;
}

GhostField cir::getSYCLHostField() {
  static GhostField field = GhostField::get("$host");
  return field;
}

GhostField cir::getSYCLKernelField() {
  static GhostField field = GhostField::get("$kernel");
  return field;
}

//===----------------------------------------------------------------------===//
// Structural helpers
//===----------------------------------------------------------------------===//

static StringRef getRole(cir::FuncOp func) {
  if (!func)
    return {};
  auto role =
      func->getAttrOfType<StringAttr>(CIRDialect::getSYCLHostAPIAttrName());
  return role ? role.getValue() : StringRef();
}

static cir::FuncOp lookupCallee(Operation *at, FlatSymbolRefAttr sym) {
  if (!at || !sym)
    return {};
  return SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(at, sym);
}

/// Whether the callee returns its result through an `sret` slot (operand 0).
static bool hasSRet(cir::FuncOp callee) {
  return callee && callee.getNumArguments() &&
         callee.getArgAttr(0, LLVM::LLVMDialect::getStructRetAttrName());
}

/// The type of call operand \p index, or null.
static Type getOperandType(const CallSiteRecord &site, unsigned index) {
  auto call = dyn_cast_if_present<CIRCallOpInterface>(site.call);
  if (site.synthetic || !call || index >= call.getNumArgOperands())
    return {};
  return call.getArgOperand(index).getType();
}

/// The record a pointer of type \p type points to, or null.
static cir::RecordType getPointeeRecord(Type type) {
  auto ptr = dyn_cast_if_present<cir::PointerType>(type);
  return ptr ? dyn_cast<cir::RecordType>(ptr.getPointee()) : cir::RecordType();
}

static Cell shiftCell(Cell cell, int64_t delta) {
  Cell r = cell.resolve();
  return Cell(r.getRawNode(), r.getRawOffset() + delta);
}

static void forEachPointerOperand(const CallSiteRecord &site,
                                  function_ref<void(unsigned, Cell)> fn) {
  for (auto [i, arg] : llvm::enumerate(site.args))
    if (arg)
      fn(i, *arg);
}

/// Ref on the objects the pointer operands point to (not what they reach),
/// but \p except.
static void refOperands(const CallSiteRecord &site, ModelContext &ctx,
                        ArrayRef<unsigned> except = {}) {
  forEachPointerOperand(site, [&](unsigned i, Cell cell) {
    if (!llvm::is_contained(except, i))
      ctx.modRef(cell, /*mod=*/false, /*ref=*/true);
  });
}

/// Mod and Ref on everything the pointer operands of a launch reach, but
/// \p except: the runtime may write through them when the command runs (a
/// reduction's result variable, for instance). Not an escape: the runtime
/// keeps none of these pointers for code we don't see.
static void modRefReachableOperands(const CallSiteRecord &site,
                                    ModelContext &ctx,
                                    ArrayRef<unsigned> except) {
  forEachPointerOperand(site, [&](unsigned i, Cell cell) {
    if (!llvm::is_contained(except, i))
      ctx.defer(DeferredEffect::Kind::Reachable, cell,
                NodeFlags::Mod | NodeFlags::Ref, nullptr);
  });
}

/// What the runtime keeps in a handle it constructs: each pointer slot of the
/// object at \p handle (of the type operand \p index points to) points to a
/// fresh unknown object owned by the runtime. The handle itself does not
/// escape: no code we don't see learns where it is.
static void setRuntimeInternals(const CallSiteRecord &site, ModelContext &ctx,
                                unsigned index, Cell handle) {
  auto ptr = dyn_cast_if_present<cir::PointerType>(getOperandType(site, index));
  if (!ptr)
    return;
  Graph &graph = ctx.getGraph();
  const Reason *why = nullptr;
  for (int64_t off :
       getPointerSlotOffsets(ptr.getPointee(), ctx.getDataLayout())) {
    if (!why)
      why = ctx.makeReason(site, Reason::Kind::EscapedTo,
                           "kept by the SYCL runtime");
    Cell fresh(graph.makeNode(NodeFlags::UnknownObj));
    graph.addFlags(fresh, NodeFlags::External | NodeFlags::Incomplete, why);
    graph.setLink(shiftCell(handle, off), fresh, why);
  }
}

/// A fresh Ghost node made at the call site.
static Cell makeGhost(const CallSiteRecord &site, ModelContext &ctx,
                      NodeFlags flags = NodeFlags::None) {
  return Cell(ctx.getGraph().makeNode(
      Origin{Origin::Kind::GhostSite, site.call}, NodeFlags::Ghost | flags));
}

/// The operand holding the kernel (or command-group) function object of a
/// launch API call: the last pointer operand, past `sret` and \p skip leading
/// operands, that points to a record not declared in namespace sycl. CIRGen
/// tags the role by the same rule on the declaration (the last lambda or
/// non-SYCL class parameter); the other parameters of these functions
/// (ranges, events, spans, properties, code locations) are SYCL classes or
/// scalars. A functor is passed by reference, so it is a pointer operand.
static std::optional<unsigned> findFunctorOperand(const CallSiteRecord &site,
                                                  unsigned first) {
  for (unsigned i = site.args.size(); i-- > first;) {
    if (!site.args[i])
      continue;
    cir::RecordType rec = getPointeeRecord(getOperandType(site, i));
    if (!rec)
      continue;
    StringAttr name = rec.getName();
    if (!name || !name.getValue().starts_with("sycl::"))
      return i;
  }
  return std::nullopt;
}

/// The operand holding the launch geometry: a pointer to a sycl::range,
/// sycl::nd_range or detail::nd_range_view, past \p first. A range passed by
/// value has no cell.
static std::optional<unsigned> findGeometryOperand(const CallSiteRecord &site,
                                                   unsigned first) {
  for (unsigned i = first; i < site.args.size(); ++i) {
    if (!site.args[i])
      continue;
    cir::RecordType rec = getPointeeRecord(getOperandType(site, i));
    StringAttr name = rec ? rec.getName() : StringAttr();
    if (!name || !name.getValue().starts_with("sycl::"))
      continue;
    StringRef n = name.getValue();
    if (n.contains("::range<") || n.contains("::nd_range<") ||
        n.contains("::nd_range_view"))
      return i;
  }
  return std::nullopt;
}

/// The cgfo object of a queue submission: the operand that points to a
/// record of two pointers, one to a function (the invoker) and one to the
/// command-group function object (detail::type_erased_cgfo_ty). Returns the
/// operand and the offsets of the object and invoker members.
struct CGFOperand {
  unsigned index;
  int64_t objectOffset;
  int64_t invokerOffset;
};

static std::optional<CGFOperand> findCGFOperand(const CallSiteRecord &site,
                                                const DataLayout &layout) {
  for (unsigned i = 0; i < site.args.size(); ++i) {
    if (!site.args[i])
      continue;
    cir::RecordType rec = getPointeeRecord(getOperandType(site, i));
    if (!rec || rec.isIncomplete() || rec.isUnion())
      continue;
    std::optional<unsigned> object, invoker;
    bool other = false;
    for (auto [idx, member] : llvm::enumerate(rec.getMembers())) {
      auto ptr = dyn_cast<cir::PointerType>(member);
      if (!ptr) {
        other = true;
        continue;
      }
      std::optional<unsigned> &slot =
          isa<cir::FuncType>(ptr.getPointee()) ? invoker : object;
      if (slot)
        other = true;
      slot = idx;
    }
    if (other || !object || !invoker)
      continue;
    return CGFOperand{
        i, static_cast<int64_t>(rec.getElementOffset(layout, *object)),
        static_cast<int64_t>(rec.getElementOffset(layout, *invoker))};
  }
  return std::nullopt;
}

/// The type of the kernel object the HostKernel(Ref) \p wrapper wraps: the
/// pointee of the kernel operand of a hostkernel.ctor call on the object
/// \p wrapper addresses (through base-class and cast steps), in the same
/// function. Null if there is none.
static Type findWrappedKernelType(Value wrapper) {
  for (unsigned i = 0; i < 8 && wrapper; ++i) {
    Operation *def = wrapper.getDefiningOp();
    if (auto base = dyn_cast_if_present<cir::BaseClassAddrOp>(def))
      wrapper = base.getSrcAddr();
    else if (auto cast = dyn_cast_if_present<cir::CastOp>(def);
             cast && cast.getKind() == cir::CastKind::bitcast)
      wrapper = cast.getSrc();
    else
      break;
  }
  for (Operation *user : wrapper.getUsers()) {
    auto call = dyn_cast<CIRCallOpInterface>(user);
    if (!call || call.isIndirect())
      continue;
    cir::FuncOp ctor =
        lookupCallee(call, call->getAttrOfType<FlatSymbolRefAttr>("callee"));
    if (getRole(ctor) != "hostkernel.ctor")
      continue;
    unsigned self = hasSRet(ctor) ? 1 : 0;
    if (self + 1 >= call.getNumArgOperands() ||
        call.getArgOperand(self) != wrapper)
      continue;
    if (auto ptr =
            dyn_cast<cir::PointerType>(call.getArgOperand(self + 1).getType()))
      return ptr.getPointee();
  }
  return {};
}

/// The kernels the launch function \p callee names: the kernel.info markers
/// its body (and the bodies it calls directly, transitively) calls.
static void collectKernelNames(cir::FuncOp callee,
                               SmallVectorImpl<StringAttr> &out,
                               llvm::DenseSet<Operation *> &visited,
                               unsigned depth = 0) {
  constexpr unsigned kMaxDepth = 16;
  if (!callee || callee.isDeclaration() || depth > kMaxDepth ||
      !visited.insert(callee).second)
    return;
  callee.walk([&](CIRCallOpInterface call) {
    if (call.isIndirect())
      return;
    cir::FuncOp target =
        lookupCallee(call, call->getAttrOfType<FlatSymbolRefAttr>("callee"));
    if (!target)
      return;
    if (getRole(target) == "kernel.info") {
      std::string name = getSYCLKernelInfoName(target);
      if (name.empty())
        return;
      auto attr = StringAttr::get(callee.getContext(), name);
      if (!llvm::is_contained(out, attr))
        out.push_back(attr);
      return;
    }
    collectKernelNames(target, out, visited, depth + 1);
  });
}

std::string cir::getSYCLKernelInfoName(cir::FuncOp kernelInfo) {
  std::string result;
  if (!kernelInfo || kernelInfo.isDeclaration())
    return result;
  kernelInfo.walk([&](cir::GetGlobalOp getGlobal) {
    auto global = SymbolTable::lookupNearestSymbolFrom<cir::GlobalOp>(
        getGlobal, getGlobal.getNameAttr());
    std::optional<Attribute> init =
        global ? global.getInitialValue() : std::nullopt;
    auto info = init ? dyn_cast<cir::ConstRecordAttr>(*init) : nullptr;
    if (!info || info.getMembers().empty())
      return WalkResult::advance();
    auto view = dyn_cast<cir::ConstRecordAttr>(info.getMembers()[0]);
    if (!view || view.getMembers().empty())
      return WalkResult::advance();
    auto str = dyn_cast<cir::GlobalViewAttr>(view.getMembers()[0]);
    auto strGlobal = str ? SymbolTable::lookupNearestSymbolFrom<cir::GlobalOp>(
                               global, str.getSymbol())
                         : cir::GlobalOp();
    std::optional<Attribute> strInit =
        strGlobal ? strGlobal.getInitialValue() : std::nullopt;
    auto chars = strInit ? dyn_cast<cir::ConstArrayAttr>(*strInit) : nullptr;
    auto text = chars ? dyn_cast<StringAttr>(chars.getElts()) : nullptr;
    if (!text)
      return WalkResult::advance();
    result = text.getValue().split('\0').first.str();
    return WalkResult::interrupt();
  });
  return result;
}

std::optional<int64_t>
cir::getSYCLClosureFieldOffset(ModuleOp module, Type closure, ArrayAttr path,
                               const DataLayout &layout) {
  // The member names of each record, from the cir.get_member ops on it.
  llvm::DenseMap<std::pair<Type, StringAttr>, unsigned> members;
  module.walk([&](cir::GetMemberOp op) {
    Type base = op.getAddrTy().getPointee();
    members.try_emplace({base, op.getNameAttr()}, op.getIndex());
  });
  int64_t offset = 0;
  Type type = closure;
  for (Attribute elem : path) {
    auto name = dyn_cast<StringAttr>(elem);
    auto rec = dyn_cast_if_present<cir::RecordType>(type);
    if (!name || !rec || rec.isIncomplete())
      return std::nullopt;
    auto it = members.find({type, name});
    if (it == members.end() || it->second >= rec.getNumElements())
      return std::nullopt;
    offset += rec.getElementOffset(layout, it->second);
    type = rec.getElementType(it->second);
  }
  return offset;
}

//===----------------------------------------------------------------------===//
// Models
//===----------------------------------------------------------------------===//

namespace {

/// A model keyed on a role. It replaces the body, and falls back to the
/// unknown-call rule if the operands are not the shape it expects.
class RoleModel : public PointsToModel {
public:
  RoleModel(StringRef role, StringRef citation)
      : role(role), citation(citation) {}

  bool matches(CIRCallOpInterface call, cir::FuncOp) const override {
    return static_cast<bool>(call);
  }
  bool replacesDefinition() const override { return true; }
  StringRef name() const override { return role; }
  StringRef specCitation() const override { return citation; }

  void apply(CallSiteRecord &site, ModelContext &ctx) const override {
    if (applyRole(site, ctx, lookupCallee(site.call, site.callee)))
      return;
    ctx.applyUnknownCall(
        site,
        ctx.makeReason(site, Reason::Kind::UnmodelledRole,
                       (role + ": operands of an unexpected shape").str()));
  }

protected:
  /// Give the call its effect; false (with nothing changed) if the operands
  /// are not the shape the model expects.
  virtual bool applyRole(CallSiteRecord &site, ModelContext &ctx,
                         cir::FuncOp callee) const = 0;

  StringRef role;
  StringRef citation;
};

/// buffer.alloc, buffer.host_ptr, buffer.other: a buffer constructor.
///
/// SPEC: buffer(range, ...) allocates new storage (PI 5392-5400, "Construct
/// with uninitialized memory"); buffer(hostData, range, ...) "assumes
/// exclusive access to this memory for the duration of its lifetime"
/// (PI 5443-5446): the runtime may use the host memory as the storage, so the
/// storage's `$host` is that memory. Any other constructor (interop, iterator
/// pairs, shared_ptr) gets storage the analysis cannot place: External.
class BufferCtorModel : public RoleModel {
public:
  enum class Kind { Alloc, HostPtr, Other };
  BufferCtorModel(StringRef role, StringRef citation, Kind kind)
      : RoleModel(role, citation), kind(kind) {}

protected:
  bool applyRole(CallSiteRecord &site, ModelContext &ctx,
                 cir::FuncOp callee) const override {
    unsigned self = hasSRet(callee) ? 1 : 0;
    std::optional<Cell> handle = ModelContext::getArg(site, self);
    if (!handle)
      return false;
    Graph &graph = ctx.getGraph();
    Cell storage = makeGhost(site, ctx);
    if (kind == Kind::Other)
      graph.addFlags(storage, NodeFlags::External,
                     ctx.makeReason(site, Reason::Kind::UnmodelledRole,
                                    "buffer.other: storage not known"));
    SmallVector<unsigned, 2> written{self};
    if (kind == Kind::HostPtr)
      if (std::optional<Cell> host = ModelContext::getArg(site, self + 1)) {
        graph.setGhostLink(storage, getSYCLHostField(), *host);
        // The runtime reads it, and writes it back on destruction.
        ctx.modRef(host, /*mod=*/true, /*ref=*/true);
        written.push_back(self + 1);
      }
    graph.setGhostLink(*handle, getSYCLStorageField(), storage);
    setRuntimeInternals(site, ctx, self, *handle);
    ctx.modRef(handle, /*mod=*/true, /*ref=*/false);
    refOperands(site, ctx, written);
    return true;
  }

private:
  Kind kind;
};

/// The roles that make a handle refer to another handle's storage.
///
/// SPEC (reference semantics, PI 295-330 `sec:reference-semantics`): a copy
/// (constructor or assignment) of a buffer or accessor "must behave as-if it
/// were the original instance". A sub-buffer is created "without allocation"
/// over "the buffer with the real data" (PI 5596-5604); a reinterpreted buffer
/// "is a copy of the original SYCL buffer only" (PI 5356-5357); an accessor
/// accesses the buffer it is constructed from (PI 7802ff,
/// `table.accessors.command.buffer.constructors`), and get_access(handler&)
/// is that constructor (PI 5703-5720). Sub-buffer ranges are not compared:
/// the sub-buffer shares its parent's storage (conservative v1).
class ShareStorageModel : public RoleModel {
public:
  enum class Shape {
    /// (this, src, ...): copy, move, assignment, sub-buffer.
    ThisFromNext,
    /// (this, ..., buffer at the role's source index, ...): accessor ctor.
    ThisFromRoleArg,
    /// (sret result, this source, ...): get_access, reinterpret.
    ResultFromThis,
  };
  ShareStorageModel(StringRef role, StringRef citation, Shape shape,
                    bool returnsThis = false)
      : RoleModel(role, citation), shape(shape), returnsThis(returnsThis) {}

protected:
  bool applyRole(CallSiteRecord &site, ModelContext &ctx,
                 cir::FuncOp callee) const override {
    bool sret = hasSRet(callee);
    unsigned dst = 0, src = 0;
    switch (shape) {
    case Shape::ThisFromNext:
      dst = sret ? 1 : 0;
      src = dst + 1;
      break;
    case Shape::ThisFromRoleArg: {
      dst = sret ? 1 : 0;
      unsigned index = 0;
      if (auto attr = callee ? callee->getAttrOfType<IntegerAttr>(
                                   CIRDialect::getSYCLHostAPIArgAttrName())
                             : IntegerAttr())
        index = attr.getInt();
      src = dst + 1 + index;
      break;
    }
    case Shape::ResultFromThis:
      if (!sret)
        return false; // The result is an SSA aggregate: no place for a ghost.
      dst = 0;
      src = 1;
      break;
    }
    std::optional<Cell> to = ModelContext::getArg(site, dst);
    std::optional<Cell> from = ModelContext::getArg(site, src);
    if (!to || !from)
      return false;
    Graph &graph = ctx.getGraph();
    const Reason *why = ctx.makeReason(site, Reason::Kind::MergedAt,
                                       (role + ": same storage").str());
    graph.setGhostLink(*to, getSYCLStorageField(),
                       graph.getGhostLink(*from, getSYCLStorageField(), why),
                       why);
    setRuntimeInternals(site, ctx, dst, *to);
    ctx.modRef(to, /*mod=*/true, /*ref=*/false);
    refOperands(site, ctx, {dst});
    if (returnsThis)
      ctx.setResult(site, *to);
    return true;
  }

private:
  Shape shape;
  bool returnsThis;
};

/// accessor.other: an accessor whose storage the analysis cannot place (a
/// default-constructed or placeholder-only accessor, a host or image one).
class OtherAccessorModel : public RoleModel {
public:
  using RoleModel::RoleModel;

protected:
  bool applyRole(CallSiteRecord &site, ModelContext &ctx,
                 cir::FuncOp callee) const override {
    unsigned self = hasSRet(callee) ? 1 : 0;
    std::optional<Cell> handle = ModelContext::getArg(site, self);
    if (!handle)
      return false;
    Cell storage = makeGhost(site, ctx);
    ctx.getGraph().addFlags(
        storage, NodeFlags::External,
        ctx.makeReason(site, Reason::Kind::UnmodelledRole,
                       "accessor.other: storage not known"));
    ctx.getGraph().setGhostLink(*handle, getSYCLStorageField(), storage);
    setRuntimeInternals(site, ctx, self, *handle);
    ctx.modRef(handle, /*mod=*/true, /*ref=*/false);
    refOperands(site, ctx, {self});
    return true;
  }
};

/// usm.alloc: SPEC (PI 11550-11560, and the malloc_shared/malloc_host/malloc
/// and aligned_alloc forms after it): "Returns a pointer to the newly
/// allocated memory": a fresh allocation, disjoint from every live object.
/// It is HasUSM: device code may access it.
class USMAllocModel : public RoleModel {
public:
  using RoleModel::RoleModel;

protected:
  bool applyRole(CallSiteRecord &site, ModelContext &ctx,
                 cir::FuncOp) const override {
    refOperands(site, ctx);
    if (!site.result)
      return !site.resultSlots;
    Cell object = ctx.makeHeapObject(site, std::nullopt);
    ctx.getGraph().addFlags(object, NodeFlags::HasUSM);
    ctx.setResult(site, object);
    return true;
  }
};

/// usm.free: SPEC (PI 12044-12060): frees the allocation; no pointer effect.
class USMFreeModel : public RoleModel {
public:
  using RoleModel::RoleModel;

protected:
  bool applyRole(CallSiteRecord &site, ModelContext &,
                 cir::FuncOp) const override {
    return !site.result && !site.resultSlots;
  }
};

/// queue.submit_impl: the private glue queue::submit (analysed) hands its
/// detail::type_erased_cgfo_ty to, which the runtime calls back.
///
/// Design B: queue::submit's body is analysed (it builds the cgfo from the
/// CGF argument, whatever the ABI made of it), and this glue is modelled as
/// a synchronous call of `cgfo.invoker(cgfo.object, handler)` with a fresh
/// handler object. D1: the model attaches to the function the public
/// queue::submit spends its CGF in, and claims only what the spec says
/// queue::submit does with it: "Host code within a command-group function
/// object ... is executed once, before the command group submit call
/// returns" (PI 14573-14574). The invoker and object are the cgfo's two
/// members, found by type (a function pointer and a data pointer), and the
/// invoker is resolved and bound like any indirect call (the cgfo
/// constructor's body stores it). Nothing the CGF captures escapes through
/// the submission; the handler is a fresh object.
class SubmitImplModel : public RoleModel {
public:
  using RoleModel::RoleModel;

protected:
  bool applyRole(CallSiteRecord &site, ModelContext &ctx,
                 cir::FuncOp callee) const override {
    if (!ctx.canAddCalls())
      return false;
    std::optional<CGFOperand> cgf = findCGFOperand(site, ctx.getDataLayout());
    if (!cgf)
      return false;
    Graph &graph = ctx.getGraph();
    Cell cgfo = *site.args[cgf->index];
    Cell object = graph.getLink(shiftCell(cgfo, cgf->objectOffset));
    Cell invoker = graph.getLink(shiftCell(cgfo, cgf->invokerOffset));
    Cell handler = makeGhost(site, ctx, NodeFlags::Mod | NodeFlags::Ref);
    ctx.addIndirectCall(site, invoker, {object, handler});
    SmallVector<unsigned, 1> written;
    if (hasSRet(callee))
      if (std::optional<Cell> event = ModelContext::getArg(site, 0)) {
        setRuntimeInternals(site, ctx, 0, *event);
        ctx.modRef(event, /*mod=*/true, /*ref=*/false);
        written.push_back(0);
      }
    refOperands(site, ctx, written);
    return true;
  }
};

/// handler.kernel, queue.kernel, launch.direct: a kernel launch, recorded.
///
/// SPEC: the variables a kernel lambda captures (the members a named
/// function object's operator() uses) "become parameters to the kernel"
/// (PI 17660-17672, `sec:kernel.parameter.passing`); a kernel parameter is
/// device copyable, and the implementation copies such data between the
/// host and the device (ARCH 1815-1826, `sec::device.copyable`). The
/// launch functions take the function object by const reference
/// (PI 14745ff, `table.members.handler.kernel`). So the runtime reads the
/// function object, copies it, and keeps no reference to it: Ref on
/// everything it reaches, nothing escapes. What the device does to USM
/// memory the closure reaches is covered by HasUSM in the call mod/ref
/// queries. The other operands (ranges, reductions, properties) get Mod and
/// Ref on what they reach: a reduction writes its result back.
class LaunchModel : public RoleModel {
public:
  LaunchModel(StringRef role, StringRef citation, bool isMethod)
      : RoleModel(role, citation), isMethod(isMethod) {}

protected:
  bool applyRole(CallSiteRecord &site, ModelContext &ctx,
                 cir::FuncOp callee) const override {
    bool sret = hasSRet(callee);
    unsigned first = (sret ? 1 : 0) + (isMethod ? 1 : 0);
    LaunchRecord launch;
    launch.call = site.call;
    launch.model = role;
    llvm::DenseSet<Operation *> visited;
    collectKernelNames(callee, launch.kernelNames, visited);
    std::optional<unsigned> functor = findFunctorOperand(site, first);
    if (!functor) {
      ctx.recordLaunch(std::move(launch));
      return false;
    }
    launch.closure = site.args[*functor];
    if (auto ptr = dyn_cast_if_present<cir::PointerType>(
            getOperandType(site, *functor)))
      launch.closureType = ptr.getPointee();
    if (std::optional<unsigned> geometry = findGeometryOperand(site, first))
      if (*geometry != *functor)
        launch.geometry = site.args[*geometry];
    ctx.defer(DeferredEffect::Kind::Reachable, *launch.closure, NodeFlags::Ref,
              nullptr);
    SmallVector<unsigned, 2> done{*functor};
    if (sret)
      if (std::optional<Cell> event = ModelContext::getArg(site, 0)) {
        setRuntimeInternals(site, ctx, 0, *event);
        ctx.modRef(event, /*mod=*/true, /*ref=*/false);
        done.push_back(0);
      }
    modRefReachableOperands(site, ctx, done);
    ctx.recordLaunch(std::move(launch));
    return true;
  }

private:
  bool isMethod;
};

/// launch.direct_impl: the runtime entry point of launch.direct (the queue
/// shortcuts), reached from analysed code only when that body is analysed.
/// Same semantics as launch.direct (see LaunchModel): the kernel object is
/// the one the HostKernelRef operand wraps (its `$kernel` ghost, set by
/// hostkernel.ctor), and the kernel is the one the kernel.info result passed
/// along names.
class LaunchDirectImplModel : public RoleModel {
public:
  using RoleModel::RoleModel;

protected:
  bool applyRole(CallSiteRecord &site, ModelContext &ctx,
                 cir::FuncOp callee) const override {
    LaunchRecord launch;
    launch.call = site.call;
    launch.model = role;
    auto call = cast<CIRCallOpInterface>(site.call);
    for (unsigned i = 0, e = call.getNumArgOperands(); i != e; ++i)
      if (auto marker =
              call.getArgOperand(i).getDefiningOp<CIRCallOpInterface>())
        if (!marker.isIndirect()) {
          cir::FuncOp info = lookupCallee(
              marker, marker->getAttrOfType<FlatSymbolRefAttr>("callee"));
          if (getRole(info) != "kernel.info")
            continue;
          std::string name = getSYCLKernelInfoName(info);
          auto attr = StringAttr::get(call.getContext(), name);
          if (!name.empty() && !llvm::is_contained(launch.kernelNames, attr))
            launch.kernelNames.push_back(attr);
        }
    std::optional<unsigned> wrapper;
    forEachPointerOperand(site, [&](unsigned i, Cell cell) {
      if (launch.closure)
        return;
      if (std::optional<Cell> kernel =
              ctx.getGraph().lookupGhostLink(cell, getSYCLKernelField())) {
        launch.closure = *kernel;
        launch.closureType = findWrappedKernelType(call.getArgOperand(i));
        wrapper = i;
      }
    });
    unsigned first = hasSRet(callee) ? 1 : 0;
    if (std::optional<unsigned> geometry = findGeometryOperand(site, first))
      launch.geometry = site.args[*geometry];
    if (!launch.closure) {
      ctx.recordLaunch(std::move(launch));
      return false;
    }
    ctx.defer(DeferredEffect::Kind::Reachable, *launch.closure, NodeFlags::Ref,
              nullptr);
    SmallVector<unsigned, 2> done{*wrapper};
    if (hasSRet(callee))
      if (std::optional<Cell> event = ModelContext::getArg(site, 0)) {
        setRuntimeInternals(site, ctx, 0, *event);
        ctx.modRef(event, /*mod=*/true, /*ref=*/false);
        done.push_back(0);
      }
    modRefReachableOperands(site, ctx, done);
    ctx.recordLaunch(std::move(launch));
    return true;
  }
};

/// hostkernel.ctor: detail::HostKernel / HostKernelRef wrapping the kernel
/// object for launch.direct and handler.kernel, whose semantics it
/// implements (see LaunchModel): it reads the kernel object and records it
/// as the wrapper's `$kernel`.
class HostKernelCtorModel : public RoleModel {
public:
  using RoleModel::RoleModel;

protected:
  bool applyRole(CallSiteRecord &site, ModelContext &ctx,
                 cir::FuncOp callee) const override {
    unsigned self = hasSRet(callee) ? 1 : 0;
    std::optional<Cell> wrapper = ModelContext::getArg(site, self);
    std::optional<Cell> kernel = ModelContext::getArg(site, self + 1);
    if (!wrapper || !kernel)
      return false;
    ctx.getGraph().setGhostLink(*wrapper, getSYCLKernelField(), *kernel);
    ctx.modRef(wrapper, /*mod=*/true, /*ref=*/false);
    refOperands(site, ctx, {self});
    return true;
  }
};

/// kernel.info, handler.set_kernel_func, handler.set_kernel_info,
/// handler.set_ndrange, kernel.wrapper: the launch glue of handler.kernel and
/// launch.direct (see LaunchModel): it reads its operands and hands them to
/// the runtime, which keeps no reference to user memory. A pointer result
/// (the kernel.info record) is runtime memory: unknown.
class LaunchGlueModel : public RoleModel {
public:
  using RoleModel::RoleModel;

protected:
  bool applyRole(CallSiteRecord &site, ModelContext &ctx,
                 cir::FuncOp) const override {
    refOperands(site, ctx);
    if (site.result || site.resultSlots)
      ctx.setResultUnknown(site, NodeFlags::External | NodeFlags::Incomplete,
                           ctx.makeReason(site, Reason::Kind::EscapedTo,
                                          "result of the SYCL runtime"));
    return true;
  }
};

} // namespace

void cir::registerSYCLPointsToModels(PointsToModelRegistry &registry) {
  using Shape = ShareStorageModel::Shape;
  using BufferKind = BufferCtorModel::Kind;
  constexpr StringLiteral refSemantics =
      "SYCL 2020 sec:reference-semantics (PI 295-330)";

  // Storage.
  registry.addRoleModel("buffer.alloc",
                        std::make_unique<BufferCtorModel>(
                            "buffer.alloc",
                            "SYCL 2020 sec:buffer-ctors (PI 5392-5400)",
                            BufferKind::Alloc));
  registry.addRoleModel("buffer.host_ptr",
                        std::make_unique<BufferCtorModel>(
                            "buffer.host_ptr",
                            "SYCL 2020 sec:buffer-ctors (PI 5443-5446)",
                            BufferKind::HostPtr));
  registry.addRoleModel(
      "buffer.other",
      std::make_unique<BufferCtorModel>(
          "buffer.other", "SYCL 2020 sec:buffer-ctors", BufferKind::Other));
  registry.addRoleModel("buffer.copy",
                        std::make_unique<ShareStorageModel>(
                            "buffer.copy", refSemantics, Shape::ThisFromNext));
  registry.addRoleModel("buffer.assign",
                        std::make_unique<ShareStorageModel>(
                            "buffer.assign", refSemantics, Shape::ThisFromNext,
                            /*returnsThis=*/true));
  registry.addRoleModel(
      "buffer.sub",
      std::make_unique<ShareStorageModel>(
          "buffer.sub",
          "SYCL 2020 buffer sub-buffer constructor (PI 5596-5604)",
          Shape::ThisFromNext));
  registry.addRoleModel("buffer.reinterpret",
                        std::make_unique<ShareStorageModel>(
                            "buffer.reinterpret",
                            "SYCL 2020 buffer::reinterpret (PI 5340-5357)",
                            Shape::ResultFromThis));
  registry.addRoleModel("buffer.get_access",
                        std::make_unique<ShareStorageModel>(
                            "buffer.get_access",
                            "SYCL 2020 api:buffer-get-access (PI 5703-5720)",
                            Shape::ResultFromThis));
  registry.addRoleModel(
      "accessor.from_buffer",
      std::make_unique<ShareStorageModel>(
          "accessor.from_buffer",
          "SYCL 2020 table.accessors.command.buffer.constructors (PI 7802ff)",
          Shape::ThisFromRoleArg));
  registry.addRoleModel(
      "accessor.copy", std::make_unique<ShareStorageModel>(
                           "accessor.copy", refSemantics, Shape::ThisFromNext));
  registry.addRoleModel("accessor.assign", std::make_unique<ShareStorageModel>(
                                               "accessor.assign", refSemantics,
                                               Shape::ThisFromNext,
                                               /*returnsThis=*/true));
  registry.addRoleModel(
      "accessor.other",
      std::make_unique<OtherAccessorModel>(
          "accessor.other",
          "SYCL 2020 table.accessors.command.buffer.constructors (PI 7802ff)"));
  // vector.data is not modelled: its body is analysed, so a host pointer
  // reaches the vector's storage.

  // USM.
  registry.addRoleModel(
      "usm.alloc",
      std::make_unique<USMAllocModel>(
          "usm.alloc", "SYCL 2020 sec:usm, malloc_* (PI 11550-11560)"));
  registry.addRoleModel(
      "usm.free", std::make_unique<USMFreeModel>(
                      "usm.free", "SYCL 2020 sec:usm, free (PI 12044-12060)"));

  // Submission and launches.
  constexpr StringLiteral cgScope =
      "SYCL 2020 sec:command.group.scope (PI 14573-14574)";
  constexpr StringLiteral kernelParams =
      "SYCL 2020 sec:kernel.parameter.passing (PI 17660-17672), "
      "sec::device.copyable (ARCH 1815-1826)";
  registry.addRoleModel("queue.submit_impl", std::make_unique<SubmitImplModel>(
                                                 "queue.submit_impl", cgScope));
  registry.addRoleModel("handler.kernel",
                        std::make_unique<LaunchModel>(
                            "handler.kernel", kernelParams, /*isMethod=*/true));
  registry.addRoleModel("queue.kernel",
                        std::make_unique<LaunchModel>(
                            "queue.kernel", kernelParams, /*isMethod=*/true));
  registry.addRoleModel("launch.direct",
                        std::make_unique<LaunchModel>(
                            "launch.direct", kernelParams, /*isMethod=*/false));
  registry.addRoleModel("launch.direct_impl",
                        std::make_unique<LaunchDirectImplModel>(
                            "launch.direct_impl", kernelParams));
  registry.addRoleModel(
      "hostkernel.ctor",
      std::make_unique<HostKernelCtorModel>("hostkernel.ctor", kernelParams));
  for (StringLiteral glue :
       {StringLiteral("kernel.info"), StringLiteral("handler.set_kernel_func"),
        StringLiteral("handler.set_kernel_info"),
        StringLiteral("handler.set_ndrange"), StringLiteral("kernel.wrapper")})
    registry.addRoleModel(
        glue, std::make_unique<LaunchGlueModel>(glue, kernelParams));
  // Not modelled: queue.submit (its body is analysed, see SubmitImplModel),
  // cgfo.ctor (its body stores the object and invoker the model reads), the
  // geometry constructors (their bodies hold the sizes), and the internal
  // glue only modelled bodies reach (queue.kernel_impl, handler.kernel_impl,
  // handler.wrap_kernel, handler.store_lambda).
}
