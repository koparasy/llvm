//===- KernelBindingTable.cpp - Host<->device kernel bindings -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Remarks.h"
#include "mlir/IR/SymbolTable.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/LaunchPathInternalization.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLHostFacts.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLKernelParams.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLPointsToModels.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "kernel-binding-table"

using namespace cir;

// The symbol tables of the container while a KernelBindingTable is built
// (the IR of the container does not change then): MLIR's
// lookupNearestSymbolFrom scans the module on every call, which on a big TU
// was most of the table's cost. Lookups outside a table's construction (the
// kernel copies of LocalKernelCopier change the container) are not cached.
namespace {
struct SymbolCacheScope {
  static thread_local SymbolCacheScope *active;
  mlir::Operation *container;
  mlir::SymbolTableCollection symbols;
  SymbolCacheScope *previous;
  explicit SymbolCacheScope(mlir::Operation *container)
      : container(container), previous(active) {
    active = this;
  }
  ~SymbolCacheScope() { active = previous; }
};
thread_local SymbolCacheScope *SymbolCacheScope::active = nullptr;
} // namespace

template <typename T>
static T lookupSymbol(mlir::Operation *from, mlir::SymbolRefAttr sym) {
  SymbolCacheScope *scope = SymbolCacheScope::active;
  if (scope && from && scope->container->isProperAncestor(from))
    return scope->symbols.lookupNearestSymbolFrom<T>(from, sym);
  return mlir::SymbolTable::lookupNearestSymbolFrom<T>(from, sym);
}

// The constant `v` holds, or null if it is not a constant.
static mlir::TypedAttr asConst(mlir::Value v) {
  if (!v)
    return {};
  auto cst = v.getDefiningOp<cir::ConstantOp>();
  return cst ? cst.getValue() : mlir::TypedAttr{};
}

// Sema picks the launch configuration function from the language mode, in
// SemaCUDA::getConfigureFuncName. Of the five it can pick, CIRGen emits a stub
// body only for the two below: `-foffload-via-llvm` reports NYI when the CUDA
// runtime is constructed, and the legacy `cudaConfigureCall` /
// `hipConfigureCall` reach errorNYI("Emit Stub Body Legacy").
//
// The argument count is part of the test because the geometry accessors index
// the operands directly; without it, only the name stands between a same-named
// declaration and a read past the end.
static bool isPushCallConfiguration(cir::CallOp call) {
  std::optional<llvm::StringRef> callee = call.getCallee();
  if (!callee)
    return false;
  assert(*callee != "__llvmPushCallConfiguration" &&
         *callee != "cudaConfigureCall" && *callee != "hipConfigureCall" &&
         "NYI: launch configuration form CIRGen does not emit");
  if (*callee != "__cudaPushCallConfiguration" &&
      *callee != "__hipPushCallConfiguration")
    return false;
  return call.getArgOperands().size() == 4;
}

// CIRGen guards a launch with the result of the push-call-configuration call:
//
//   %c = cir.call @__cudaPushCallConfiguration(...)
//   %b = cir.cast int_to_bool %c
//   cir.if %b { } else { cir.call @stub(...) }
//
// Walk that back from the stub call. Returns null for any other shape,
// including a launch whose CFG has already been flattened.
static cir::CallOp tracePushConfiguration(cir::CallOp stubCall) {
  auto ifOp = stubCall->getParentOfType<cir::IfOp>();
  if (!ifOp)
    return {};
  auto cast = ifOp.getCondition().getDefiningOp<cir::CastOp>();
  if (!cast)
    return {};
  auto call = cast.getSrc().getDefiningOp<cir::CallOp>();
  if (!call || !isPushCallConfiguration(call))
    return {};
  return call;
}

// Whether `call` runs a constructor of `recordType`. CIR marks a constructor
// with a cxx_ctor attribute naming the record it builds, so the call is
// identified by that rather than by a mangled name: dim3 has more than one
// constructor symbol, and a name check would also accept an unrelated function
// with a matching signature.
static bool constructsRecord(cir::CallOp call, mlir::Type recordType) {
  mlir::FlatSymbolRefAttr callee = call.getCalleeAttr();
  if (!callee)
    return false;
  auto fn = lookupSymbol<cir::FuncOp>(call, callee);
  if (!fn)
    return false;
  auto ctor = mlir::dyn_cast_if_present<cir::CXXCtorAttr>(fn.getFuncInfoAttr());
  return ctor && ctor.getType() == recordType;
}

// A dim3 argument of the push call is a load of a stack temporary that a
// constructor filled in:
//
//   %t = cir.alloca "agg.tmp0" : !cir.ptr<!rec_dim3>
//   cir.call @_ZN4dim3C1Ejjj(%t, %x, %y, %z)
//   %d = cir.load %t
//
// The constructor is found through the slot, so the components belong to the
// temporary this launch reads rather than to any dim3 in the function.
static cir::LaunchSite::Dim3 traceDim3(mlir::Value dim) {
  auto load = dim.getDefiningOp<cir::LoadOp>();
  if (!load)
    return {};
  mlir::Value slot = load.getAddr();
  auto slotType = mlir::dyn_cast<cir::PointerType>(slot.getType());
  if (!slotType)
    return {};

  cir::CallOp ctor;
  for (mlir::Operation *user : slot.getUsers()) {
    auto call = mlir::dyn_cast<cir::CallOp>(user);
    if (!call || !constructsRecord(call, slotType.getPointee()))
      continue;
    // Users are unordered, so a second construction of the same slot leaves
    // the one reaching the load undecided; report no geometry instead.
    if (ctor)
      return {};
    ctor = call;
  }
  if (!ctor || ctor->getBlock() != load->getBlock() ||
      !ctor->isBeforeInBlock(load))
    return {};

  mlir::OperandRange args = ctor.getArgOperands();
  if (args.size() != 4 || args[0] != slot)
    return {};
  return {args[1], args[2], args[3]};
}

llvm::StringRef cir::LaunchSite::getKernelName() const {
  if (kind == BindingKind::SYCLName)
    return syclKernelName;
  cir::CallOp call = stubCall;
  return cir::getLaunchedKernel(call).getKernelName();
}

unsigned cir::LaunchSite::getNumArgs() const {
  return cir::CallOp(stubCall).getArgOperands().size();
}

mlir::Value cir::LaunchSite::getArg(unsigned i) const {
  mlir::OperandRange args = cir::CallOp(stubCall).getArgOperands();
  return i < args.size() ? args[i] : mlir::Value{};
}

mlir::TypedAttr cir::LaunchSite::getConstArg(unsigned i) const {
  return asConst(getArg(i));
}

mlir::TypedAttr cir::LaunchSite::Dim3::constX() const { return asConst(x); }
mlir::TypedAttr cir::LaunchSite::Dim3::constY() const { return asConst(y); }
mlir::TypedAttr cir::LaunchSite::Dim3::constZ() const { return asConst(z); }

bool cir::LaunchSite::Dim3::isTraced() const { return x && y && z; }

bool cir::LaunchSite::Dim3::isFullyConstant() const {
  return constX() && constY() && constZ();
}

// Push call operands: grid, block, shared memory, stream.
cir::LaunchSite::Dim3 cir::LaunchSite::getGridDim() const {
  if (!hasGeometry())
    return {};
  return traceDim3(cir::CallOp(pushConfigCall).getArgOperands()[0]);
}

cir::LaunchSite::Dim3 cir::LaunchSite::getBlockDim() const {
  if (!hasGeometry())
    return {};
  return traceDim3(cir::CallOp(pushConfigCall).getArgOperands()[1]);
}

mlir::Value cir::LaunchSite::getSharedMemBytes() const {
  if (!hasGeometry())
    return {};
  return cir::CallOp(pushConfigCall).getArgOperands()[2];
}

mlir::TypedAttr cir::LaunchSite::getConstSharedMem() const {
  return asConst(getSharedMemBytes());
}

mlir::Value cir::LaunchSite::getStream() const {
  if (!hasGeometry())
    return {};
  return cir::CallOp(pushConfigCall).getArgOperands()[3];
}

bool cir::LaunchSite::isDefaultStream() const {
  auto ptr = mlir::dyn_cast_if_present<cir::ConstPtrAttr>(asConst(getStream()));
  return ptr && ptr.isNullValue();
}

cir::CUDAKernelNameAttr cir::getLaunchedKernel(mlir::Operation *op) {
  if (!mlir::isa<cir::CallOp>(op))
    return {};
  return op->getAttrOfType<cir::CUDAKernelNameAttr>(
      cir::CUDAKernelNameAttr::getMnemonic());
}

// The character-array value a `cir.global` string constant holds, or empty if
// it is not a plain char-array constant. Trailing NUL is not part of the value.
static llvm::StringRef getGlobalStringValue(cir::GlobalOp global) {
  std::optional<mlir::Attribute> init = global.getInitialValue();
  if (!init)
    return {};
  auto arr = mlir::dyn_cast<cir::ConstArrayAttr>(*init);
  if (!arr)
    return {};
  auto str = mlir::dyn_cast<mlir::StringAttr>(arr.getElts());
  return str ? str.getValue() : llvm::StringRef{};
}

// The call of the closure's call operator in the SYCL device kernel `kernel`:
// the call whose receiver is the kernel object, the one named record alloca
// reaching a call in receiver position (the SYCL item/id temporaries are
// passed by value, never as a pointer receiver).
static cir::CallOp findClosureCall(cir::FuncOp kernel) {
  cir::CallOp found;
  kernel.walk([&](cir::AllocaOp alloca) {
    if (found)
      return;
    auto rec = mlir::dyn_cast<cir::RecordType>(alloca.getAllocaType());
    if (!rec || !rec.getName())
      return;
    llvm::SmallVector<mlir::Value, 4> worklist{alloca.getResult()};
    llvm::SmallPtrSet<mlir::Operation *, 8> seen;
    while (!worklist.empty() && !found) {
      mlir::Value cur = worklist.pop_back_val();
      for (mlir::Operation *user : cur.getUsers()) {
        if (auto call = mlir::dyn_cast<cir::CallOp>(user)) {
          mlir::OperandRange args = call.getArgOperands();
          if (!args.empty() && args[0] == cur && !call.isIndirect()) {
            found = call;
            break;
          }
          continue;
        }
        if (mlir::isa<cir::CastOp, cir::BaseClassAddrOp>(user))
          if (seen.insert(user).second)
            worklist.push_back(user->getResult(0));
      }
    }
  });
  return found;
}

cir::FuncOp cir::getSYCLClosureCallOperator(cir::FuncOp kernel) {
  cir::CallOp call = findClosureCall(kernel);
  return call ? lookupSymbol<cir::FuncOp>(call, call.getCalleeAttr())
              : cir::FuncOp();
}

bool cir::isInternalizedSYCLKernel(cir::FuncOp kernel) {
  return kernel && kernel->hasAttr(kInternalizedKernelAttr);
}

void cir::noteInternalizedSYCLKernelUse(cir::FuncOp kernel) {
  if (isInternalizedSYCLKernel(kernel))
    kernel->setAttr(kInternalizedUsedAttr,
                    mlir::UnitAttr::get(kernel.getContext()));
}

std::string cir::describeInternalizedSYCLKernel(cir::FuncOp kernel) {
  auto info =
      kernel
          ? kernel->getAttrOfType<mlir::DictionaryAttr>(kInternalizedKernelAttr)
          : nullptr;
  if (!info)
    return "";
  std::string out = " through the internalized launch path from root";
  auto roots = info.getAs<mlir::ArrayAttr>("roots");
  if (roots)
    for (auto [i, root] : llvm::enumerate(roots))
      if (auto name = mlir::dyn_cast<mlir::StringAttr>(root))
        out += (i ? ", @" : " @") + name.getValue().str();
  return out;
}

std::string cir::getSharedSYCLKernelNote(bool internalized) {
  std::string note = "the kernel is visible to other TUs; facts this TU "
                     "derives hold only through an internalized launch path";
  if (!internalized)
    note += ", and launch-path internalization is off";
  return note;
}

llvm::StringMap<bool> cir::getSYCLKernelInfoLocality(mlir::ModuleOp container) {
  llvm::StringMap<bool> local;
  mlir::ModuleOp host = cir::getOffloadContainerHostModule(container);
  mlir::SymbolTableCollection symbols;
  for (auto func : host.getOps<cir::FuncOp>()) {
    auto role = func->getAttrOfType<mlir::StringAttr>(
        cir::CIRDialect::getSYCLHostAPIAttrName());
    if (!role || role.getValue() != "kernel.info" || func.isDeclaration())
      continue;
    std::string name = cir::getSYCLKernelInfoName(func, &symbols);
    if (name.empty())
      continue;
    auto [it, inserted] =
        local.try_emplace(name, cir::isLocalLinkage(func.getLinkage()));
    if (!inserted)
      it->second &= cir::isLocalLinkage(func.getLinkage());
  }
  return local;
}

bool cir::isTUUniqueSYCLKernel(mlir::ModuleOp container, cir::FuncOp kernel,
                               llvm::StringRef name,
                               const llvm::StringMap<bool> *getterLocality) {
  if (isInternalizedSYCLKernel(kernel))
    return true;
  // The closure type: a local call operator means another TU cannot build
  // the kernel's function object.
  if (cir::CallOp call = findClosureCall(kernel))
    if (auto op = lookupSymbol<cir::FuncOp>(call, call.getCalleeAttr()))
      if (cir::isLocalLinkage(op.getLinkage()))
        return true;
  // The kernel-name type: every getDeviceKernelInfo<K> getter is local.
  if (getterLocality)
    return getterLocality->lookup(name);
  mlir::ModuleOp host = cir::getOffloadContainerHostModule(container);
  mlir::SymbolTableCollection symbols;
  bool any = false;
  for (auto func : host.getOps<cir::FuncOp>()) {
    auto role = func->getAttrOfType<mlir::StringAttr>(
        cir::CIRDialect::getSYCLHostAPIAttrName());
    if (!role || role.getValue() != "kernel.info" || func.isDeclaration() ||
        cir::getSYCLKernelInfoName(func, &symbols) != name)
      continue;
    if (!cir::isLocalLinkage(func.getLinkage()))
      return false;
    any = true;
  }
  return any;
}

// Whether `name` appears as a registered (by-name / bundle-exposed) kernel in
// any module's `cir.sycl_registered_kernels` attribute -- an array of
// [user-name, mangled-name] string pairs. Such a kernel is reachable by name
// across TUs, so it can never be treated as TU-local.
bool cir::isRegisteredSYCLKernel(mlir::ModuleOp container,
                                 llvm::StringRef name) {
  auto inModule = [&](mlir::ModuleOp mod) {
    auto arr = mod->getAttrOfType<mlir::ArrayAttr>(
        cir::CIRDialect::getSYCLRegisteredKernelsAttrName());
    if (!arr)
      return false;
    for (mlir::Attribute e : arr) {
      auto pair = mlir::dyn_cast<mlir::ArrayAttr>(e);
      if (pair && pair.size() == 2)
        if (auto mangled = mlir::dyn_cast<mlir::StringAttr>(pair[1]))
          if (mangled.getValue() == name)
            return true;
    }
    return false;
  };
  if (inModule(cir::getOffloadContainerHostModule(container)))
    return true;
  for (mlir::ModuleOp deviceMod :
       cir::getOffloadContainerDeviceModules(container))
    if (inModule(deviceMod))
      return true;
  return false;
}

bool cir::hasUseOutsideSelf(cir::FuncOp fn, mlir::Operation *scope) {
  auto uses = mlir::SymbolTable::getSymbolUses(fn, scope);
  if (!uses)
    return true; // Uses could not be enumerated: conservatively assume one.
  return llvm::any_of(*uses, [&](const mlir::SymbolTable::SymbolUse &use) {
    return !fn->isProperAncestor(use.getUser());
  });
}

// The launch sites recorded for a stub are all of them when no other TU can
// call it and every reference to it is one the table recorded. A reference that
// is not a launch site of this kernel -- an escaping address, or a call the
// table did not collect -- may reach the kernel in a way this table cannot
// describe. Uses inside the stub's own body are its self-reference.
bool KernelBindingTable::allLaunchSitesVisible(
    llvm::StringRef kernelName) const {
  const KernelBinding *binding = lookup(kernelName);
  // Only a CUDA/HIP binding has a host stub whose uses describe the launches;
  // a SYCL binding is launched by name and reports its own visibility.
  if (!binding || !binding->hostStub)
    return false;
  cir::FuncOp stub = binding->hostStub;
  if (!cir::isLocalLinkage(stub.getLinkage()))
    return false;
  auto uses = mlir::SymbolTable::getSymbolUses(stub, hostModule);
  if (!uses)
    return false;
  return llvm::all_of(*uses, [&](const mlir::SymbolTable::SymbolUse &use) {
    if (stub->isProperAncestor(use.getUser()))
      return true;
    cir::CUDAKernelNameAttr launched = cir::getLaunchedKernel(use.getUser());
    return launched && launched.getKernelName() == kernelName;
  });
}

void cir::emitOffloadRemark(mlir::Location loc, OffloadRemarkKind kind,
                            llvm::StringRef pass, llvm::StringRef category,
                            llvm::StringRef function, const llvm::Twine &text) {
  mlir::MLIRContext *ctx = loc->getContext();
  if (!ctx->getRemarkEngine())
    return;
  auto opts = mlir::remark::RemarkOpts::name(pass).category(category).function(
      function);
  std::string message = text.str();
  switch (kind) {
  case OffloadRemarkKind::Passed:
    mlir::remark::passed(loc, opts) << llvm::StringRef(message);
    return;
  case OffloadRemarkKind::Missed:
    mlir::remark::missed(loc, opts) << llvm::StringRef(message);
    return;
  case OffloadRemarkKind::Analysis:
    mlir::remark::analysis(loc, opts) << llvm::StringRef(message);
    return;
  }
}

llvm::StringRef cir::getHostFactsModeName(HostFactsMode mode) {
  switch (mode) {
  case HostFactsMode::None:
    return "none";
  case HostFactsMode::IPO:
    return "ipo";
  }
  llvm_unreachable("unknown host-facts mode");
}

std::optional<HostFactsMode> cir::parseHostFactsMode(llvm::StringRef name) {
  return llvm::StringSwitch<std::optional<HostFactsMode>>(name)
      .Case("none", HostFactsMode::None)
      .Case("ipo", HostFactsMode::IPO)
      .Default(std::nullopt);
}

std::string cir::formatSyclGeometry(const SyclGeometry &geometry) {
  if (!geometry.valid)
    return "none";
  std::string out;
  llvm::raw_string_ostream os(out);
  auto emit = [&](llvm::StringRef tag,
                  llvm::ArrayRef<std::optional<int64_t>> sizes) {
    os << tag << "=[";
    for (auto [i, s] : llvm::enumerate(sizes)) {
      if (i)
        os << ",";
      if (s)
        os << *s;
      else
        os << "?";
    }
    os << "]";
  };
  emit("global", geometry.global);
  if (!geometry.local.empty()) {
    os << " ";
    emit("local", geometry.local);
  }
  return out;
}

std::string cir::formatHostFactsOptions(const HostFactsOptions &options) {
  std::string out = "(";
  out += options.assumeClosedTU ? "closed-tu" : "open";
  out += options.bottomUp ? ", bu" : ", ci";
  if (options.subobjects)
    out += ", subobjects";
  if (options.internalizedOpenMode())
    out += ", internalized";
  return out + ")";
}

std::string cir::getHostFactsAnalysisKey(const HostFactsOptions &options) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << options.assumeClosedTU << options.bottomUp << options.subobjects
     << options.relevanceSlice << " " << options.budgetScale << " "
     << options.maxSeconds << " " << options.maxMemoryMB << " "
     << options.assumeSpecHostPtrUB;
  if (options.internalizedOpenMode())
    os << " internalized";
  return out;
}

KernelBindingTable::KernelBindingTable(mlir::ModuleOp container,
                                       const HostFactsOptions &options,
                                       SYCLHostFactsCache *cache)
    : hostFactsOptions(options), hostFactsCache(cache) {
  SymbolCacheScope symbolCache(container);
  // `container` must be an offload container; the helpers assert the unit attr
  // and recover the nested host and device modules.
  mlir::ModuleOp hostModule = cir::getOffloadContainerHostModule(container);
  this->hostModule = hostModule;

  hostModule.walk([&](cir::FuncOp hostFn) {
    auto kernelNameAttr = hostFn->getAttrOfType<cir::CUDAKernelNameAttr>(
        cir::CUDAKernelNameAttr::getMnemonic());
    if (kernelNameAttr) {
      assert(!lookup(kernelNameAttr.getKernelName()) &&
             "Duplicate stub found for Host TU");
      bindings[kernelNameAttr.getKernelName()].hostStub = hostFn;
    }
  });

  // Walked after the stubs, so every binding already has its host stub. A
  // launch site naming a kernel with no stub here launches nothing this table
  // describes, and recording it would leave a binding without a stub.
  hostModule.walk([&](mlir::Operation *op) {
    cir::CUDAKernelNameAttr kernel = cir::getLaunchedKernel(op);
    if (!kernel)
      return;
    auto it = bindings.find(kernel.getKernelName());
    if (it == bindings.end())
      return;
    cir::CallOp stubCall = mlir::cast<cir::CallOp>(op);
    LaunchSite site;
    site.stubCall = stubCall;
    site.pushConfigCall = tracePushConfiguration(stubCall);
    it->second.launchSites.push_back(site);
  });

  for (mlir::ModuleOp deviceMod :
       cir::getOffloadContainerDeviceModules(container)) {
    for (auto &binding : bindings) {
      if (cir::FuncOp kernel = llvm::dyn_cast_if_present<cir::FuncOp>(
              deviceMod.lookupSymbol(binding.first))) {
        binding.second.deviceKernels.push_back(kernel);
      }
    }
  }

  bindSYCLKernels(container);
  LLVM_DEBUG(print(llvm::dbgs()));
}

//===----------------------------------------------------------------------===//
// SYCL launch recognition (host-API roles)
//===----------------------------------------------------------------------===//
//
// CIRGen tags the SYCL launch API by declaration (cir.sycl_host_api, see
// CIRGenModule::getSYCLHostAPIRole), so launches are recognized by role, not
// by mangled name:
//  - the marker: detail::getDeviceKernelInfo<K>() (kernel.info), which names
//    the kernel and whose DeviceKernelInfo* result the launch passes on;
//  - the direct path (the queue shortcuts): the runtime launch
//    submit_kernel_direct_{with,without}_event_impl (launch.direct_impl),
//    whose HostKernelRefBase operand wraps the closure;
//  - the handler path (`q.submit([&](handler &h) { h.parallel_for(...); })`):
//    handler::setDeviceKernelInfoPtr(this, dki) (handler.set_kernel_info),
//    whose handler's MHostKernel is reset to a detail::HostKernel built
//    (hostkernel.ctor) from the closure.

/// The cir.sycl_host_api role of the direct callee of \p call, or empty.
static llvm::StringRef calleeRole(cir::CallOp call) {
  if (call.isIndirect())
    return {};
  auto fn = lookupSymbol<cir::FuncOp>(call, call.getCalleeAttr());
  auto role = fn ? fn->getAttrOfType<mlir::StringAttr>(
                       cir::CIRDialect::getSYCLHostAPIAttrName())
                 : nullptr;
  return role ? role.getValue() : llvm::StringRef();
}

// Whether `call` is the templated `sycl::detail::getDeviceKernelInfo<Kernel>()`
// launch marker (kernel.info): it takes no operands and returns the
// DeviceKernelInfo*.
static bool isGetDeviceKernelInfoCall(cir::CallOp call) {
  return call.getArgOperands().empty() && call->getNumResults() == 1 &&
         calleeRole(call) == "kernel.info";
}

// The operands of a recovered SYCL launch, from either launch shape.
namespace {
struct RecoveredLaunch {
  cir::CallOp anchor; // stored as LaunchSite::stubCall (launch-in @caller)
  mlir::Value
      closure; // HostKernelRef* (direct) / lambda-bearing value (handler)
};
} // namespace

// Defined below; follows `v = load(slot)` back to the pointer stored into the
// slot, so a value reloaded before use resolves to the object it names.
static mlir::Value stripToPointerObject(mlir::Value v);

// Whether `call` is `std::unique_ptr<HostKernelBase>::reset(HostKernelBase*)`
// -- the write that installs the wrapped kernel into handler.MHostKernel. Its
// arg1 is the freshly-built HostKernel object (as a HostKernelBase* base
// pointer). A standard library method, which carries no SYCL role; used only
// to recover the closure operand (the closure-layout check), never to
// recognize a launch.
static bool isHostKernelReset(cir::CallOp call) {
  std::optional<llvm::StringRef> callee = call.getCallee();
  return callee && callee->contains("HostKernelBase") &&
         callee->contains("5reset");
}

// The user lambda a handler-path launch captures, recovered from the handler.
// StoreLambda installs the wrapped kernel with
// `handler.MHostKernel.reset(new HostKernel<KernelType,...>(lambda))`, so the
// launch's lambda is reached structurally: the reset writing THIS handler's
// MHostKernel member gives the freshly-built HostKernel object, and that
// object's HostKernel constructor (hostkernel.ctor) took the user lambda as
// arg1. Keying off the handler's own member is what disambiguates the launch
// when several q.submit(handler) sites share one function: each has its own
// handler and its own MHostKernel.
static mlir::Value findHandlerClosure(mlir::Value handler) {
  for (mlir::Operation *hu : handler.getUsers()) {
    auto member = mlir::dyn_cast<cir::GetMemberOp>(hu);
    if (!member || member.getName() != "MHostKernel")
      continue;
    for (mlir::Operation *mu : member.getResult().getUsers()) {
      auto reset = mlir::dyn_cast<cir::CallOp>(mu);
      if (!reset || !isHostKernelReset(reset))
        continue;
      mlir::OperandRange resetArgs = reset.getArgOperands();
      if (resetArgs.size() < 2)
        continue;
      // The new'd HostKernel<KernelType,...> object. Peel the HostKernelBase
      // base-class view off the base pointer the reset consumes, then follow
      // the spill slot the pointer was stored into and reloaded from, back to
      // the operator-new result the constructor ran on.
      mlir::Value hostKernel = resetArgs[1];
      while (auto bca = hostKernel.getDefiningOp<cir::BaseClassAddrOp>())
        hostKernel = bca.getSrcAddr();
      hostKernel = stripToPointerObject(hostKernel);
      for (mlir::Operation *hku : hostKernel.getUsers()) {
        auto ctor = mlir::dyn_cast<cir::CallOp>(hku);
        if (!ctor || calleeRole(ctor) != "hostkernel.ctor")
          continue;
        mlir::OperandRange ctorArgs = ctor.getArgOperands();
        if (ctorArgs.size() >= 2 && ctorArgs[0] == hostKernel)
          return ctorArgs[1];
      }
    }
  }
  return {};
}

// The closure operand of a launch.direct_impl call: the operand that is (a
// base-class or reinterpret view of) the detail::HostKernelRef object a
// hostkernel.ctor call constructed, or null. Found by role, not by position.
static mlir::Value findDirectClosure(cir::CallOp call) {
  for (mlir::Value arg : call.getArgOperands()) {
    mlir::Value obj = stripToPointerObject(arg);
    for (unsigned i = 0; i < 8; ++i) {
      if (auto bca = obj.getDefiningOp<cir::BaseClassAddrOp>()) {
        obj = bca.getSrcAddr();
        continue;
      }
      if (auto cast = obj.getDefiningOp<cir::CastOp>())
        if (cast.getKind() == cir::CastKind::bitcast ||
            cast.getKind() == cir::CastKind::address_space) {
          obj = cast.getSrc();
          continue;
        }
      break;
    }
    for (mlir::Operation *user : obj.getUsers()) {
      auto ctor = mlir::dyn_cast<cir::CallOp>(user);
      if (ctor && calleeRole(ctor) == "hostkernel.ctor" &&
          !ctor.getArgOperands().empty() && ctor.getArgOperands()[0] == obj)
        return arg;
    }
  }
  return {};
}

// Follow the DeviceKernelInfo* result of a kernel.info marker to the launch it
// drives, recognizing both launch shapes. CIRGen spills the pointer into a
// stack slot and reloads it before the consuming call, so hop
// store->slot->load once.
static std::optional<RecoveredLaunch>
recoverLaunch(mlir::Value deviceKernelInfo) {
  llvm::SmallVector<mlir::Value, 2> reachable{deviceKernelInfo};
  for (mlir::Operation *user : deviceKernelInfo.getUsers()) {
    auto store = mlir::dyn_cast<cir::StoreOp>(user);
    if (!store || store.getValue() != deviceKernelInfo)
      continue;
    for (mlir::Operation *slotUser : store.getAddr().getUsers())
      if (auto load = mlir::dyn_cast<cir::LoadOp>(slotUser))
        reachable.push_back(load.getResult());
  }
  for (mlir::Value v : reachable)
    for (mlir::Operation *user : v.getUsers()) {
      auto call = mlir::dyn_cast<cir::CallOp>(user);
      if (!call)
        continue;
      llvm::StringRef role = calleeRole(call);
      if (role == "launch.direct_impl")
        return RecoveredLaunch{call, findDirectClosure(call)};
      if (role == "handler.set_kernel_info") {
        mlir::OperandRange args = call.getArgOperands();
        if (args.empty())
          return std::nullopt;
        return RecoveredLaunch{call, findHandlerClosure(args[0])};
      }
    }
  return std::nullopt;
}

// The `.str` char-array GlobalOp a CompileTimeKernelInfo record names as its
// kernel-name string_view -- its first member `{global_view<@.str>, len}`.
static cir::GlobalOp getCompileTimeKernelName(cir::GlobalOp info) {
  std::optional<mlir::Attribute> init = info.getInitialValue();
  if (!init)
    return {};
  auto rec = mlir::dyn_cast<cir::ConstRecordAttr>(*init);
  if (!rec || rec.getMembers().empty())
    return {};
  auto view = mlir::dyn_cast<cir::ConstRecordAttr>(rec.getMembers()[0]);
  if (!view || view.getMembers().empty())
    return {};
  auto strView = mlir::dyn_cast<cir::GlobalViewAttr>(view.getMembers()[0]);
  if (!strView)
    return {};
  return lookupSymbol<cir::GlobalOp>(info, strView.getSymbol());
}

// The _ZTS name-string GlobalOp a `getDeviceKernelInfo<Kernel>()` call targets.
// Its callee body references a `CompileTimeKernelInfo<Kernel>` global whose
// first field is the kernel-name string_view -- purely an IR/symbol walk, no
// AST and no callee-name substring matching.
static cir::GlobalOp resolveLaunchedNameString(cir::CallOp call) {
  mlir::FlatSymbolRefAttr callee = call.getCalleeAttr();
  if (!callee)
    return {};
  auto fn = lookupSymbol<cir::FuncOp>(call, callee);
  if (!fn)
    return {};
  cir::GlobalOp nameString;
  fn.walk([&](cir::GetGlobalOp getGlobal) {
    auto global =
        lookupSymbol<cir::GlobalOp>(getGlobal, getGlobal.getNameAttr());
    if (!global || !global.getName().contains("CompileTimeKernelInfo"))
      return;
    if (cir::GlobalOp name = getCompileTimeKernelName(global))
      nameString = name;
  });
  return nameString;
}

// Follow a spill slot: `v = load(slot)` where `slot` was `store(ptr, slot)`.
// Returns the stored pointer, so a value reloaded before the launch resolves to
// the object it names. CIRGen spills the nd_range_view pointer into a
// "RangeView" slot and reloads it at the call.
static mlir::Value stripToPointerObject(mlir::Value v) {
  for (unsigned i = 0; i < 8 && v; ++i) {
    auto load = v.getDefiningOp<cir::LoadOp>();
    if (!load)
      break;
    mlir::Value slot = load.getAddr();
    mlir::Value found;
    for (mlir::Operation *u : slot.getUsers())
      if (auto st = mlir::dyn_cast<cir::StoreOp>(u))
        if (st.getAddr() == slot &&
            mlir::isa<cir::PointerType>(st.getValue().getType()))
          found = st.getValue();
    if (!found)
      break;
    v = found;
  }
  return v;
}

// The noalias candidates of a kernel: see SYCLNoAliasCandidates.
using NoAliasCandidates = SYCLNoAliasCandidates;

static bool containsPointer(mlir::Type type) {
  if (mlir::isa<cir::PointerType>(type))
    return true;
  if (auto rec = mlir::dyn_cast<cir::RecordType>(type))
    return llvm::any_of(rec.getMembers(), containsPointer);
  if (auto arr = mlir::dyn_cast<cir::ArrayType>(type))
    return containsPointer(arr.getElementType());
  return false;
}

/// Whether a kernel pointer parameter points into global memory: anything but
/// the work-group local address space.
static bool isLocalPointer(mlir::Type type) {
  auto ptrTy = mlir::dyn_cast<cir::PointerType>(type);
  if (!ptrTy)
    return false;
  auto as = mlir::dyn_cast_if_present<cir::TargetAddressSpaceAttr>(
      ptrTy.getAddrSpace());
  return as && as.getValue() == 3;
}

void cir::classifySYCLKernelParams(cir::FuncOp kernel,
                                   SYCLNoAliasCandidates &out) {
  auto poison = [&](llvm::Twine why) {
    if (out.poison.empty())
      out.poison = why.str();
  };
  for (unsigned i = 0, e = kernel.getNumArguments(); i != e; ++i) {
    mlir::Type type = kernel.getArgumentTypes()[i];
    std::optional<cir::SYCLKernelParam> param =
        cir::getSYCLKernelParam(kernel, i);
    if (!param) {
      if (mlir::isa<cir::PointerType>(type) && !isLocalPointer(type))
        poison("kernel argument " + llvm::Twine(i) +
               " has no recorded closure field");
      continue;
    }
    if (param->kind == "scalar" || param->kind == "local_accessor")
      continue;
    if (param->kind == "std_layout") {
      mlir::Type contents = type;
      if (auto byval = kernel.getArgAttrOfType<mlir::TypeAttr>(
              i, mlir::LLVM::LLVMDialect::getByValAttrName()))
        contents = byval.getValue();
      if (containsPointer(contents))
        poison("captured struct '" + param->field + "' holds a pointer");
      continue;
    }
    bool isAccessor = param->kind == "accessor";
    if (isAccessor && param->accessorPart != "ptr")
      continue; // a range or offset
    if (!mlir::isa<cir::PointerType>(type) || isLocalPointer(type))
      continue;
    if (param->kind != "pointer" && !isAccessor) {
      poison("captured '" + param->field + "' is a " + param->kind);
      continue;
    }
    if (!param->isTopLevel()) {
      poison("captured '" + param->field + "' holds a nested " + param->kind);
      continue;
    }
    out.fields[param->field] = isAccessor;
  }
}

SYCLNoAliasCandidates
cir::getSYCLNoAliasCandidates(llvm::ArrayRef<cir::FuncOp> deviceKernels) {
  SYCLNoAliasCandidates merged;
  for (auto [i, kernel] : llvm::enumerate(deviceKernels)) {
    SYCLNoAliasCandidates c;
    classifySYCLKernelParams(kernel, c);
    if (i == 0) {
      merged = std::move(c);
      continue;
    }
    if (!c.poison.empty() && merged.poison.empty())
      merged.poison = c.poison;
    if (c.fields.size() != merged.fields.size() ||
        llvm::any_of(c.fields, [&](const auto &f) {
          auto it = merged.fields.find(f.first());
          return it == merged.fields.end() || it->second != f.second;
        }))
      if (merged.poison.empty())
        merged.poison = "the kernel's device copies disagree on its captures";
  }
  return merged;
}

void cir::andNoAliasDecisions(llvm::StringMap<NoAliasDecision> here,
                              bool &seenAnySite,
                              llvm::StringMap<NoAliasDecision> &decisions) {
  if (!seenAnySite) {
    decisions = std::move(here);
    seenAnySite = true;
    return;
  }
  // AND across sites: a field must be eligible here and previously, and any
  // field missing from either side drops out.
  llvm::StringMap<NoAliasDecision> merged;
  for (auto &e : decisions) {
    NoAliasDecision d = e.second;
    auto it = here.find(e.first());
    if (it == here.end())
      d = {false, "not captured at every launch"};
    else if (d.ok && !it->second.ok)
      d = it->second;
    merged[e.first()] = std::move(d);
  }
  decisions = std::move(merged);
}

// SYCL binding: there is no host stub and no cu.kernel_name. Bind each device
// spir_kernel to the host string constant equal to its mangled name, counting
// it only if that string is actually referenced by host code. Classify each
// binding's eligibility for TU-local optimization.
void KernelBindingTable::bindSYCLKernels(mlir::ModuleOp container) {
  mlir::ModuleOp hostModule = cir::getOffloadContainerHostModule(container);

  // Host string constants indexed by their char-array value, so a device
  // kernel's mangled name can be matched by string equality. A false positive
  // is impossible: only a string equal to a real device-kernel symbol matches.
  llvm::StringMap<cir::GlobalOp> nameStrings;
  hostModule.walk([&](cir::GlobalOp global) {
    llvm::StringRef value = getGlobalStringValue(global);
    if (!value.empty())
      nameStrings.try_emplace(value, global);
  });

  // The symbols the host module uses, collected once (getSymbolUses per
  // kernel walked the whole module for each).
  std::optional<llvm::DenseSet<mlir::StringAttr>> hostUses;
  auto isUsedInHost = [&](cir::GlobalOp global) {
    if (!hostUses) {
      hostUses.emplace();
      if (auto uses =
              mlir::SymbolTable::getSymbolUses(&hostModule.getBodyRegion()))
        for (const mlir::SymbolTable::SymbolUse &use : *uses)
          hostUses->insert(use.getSymbolRef().getRootReference());
    }
    return hostUses->contains(global.getSymNameAttr());
  };

  // Which kernel names only local getters name (isTUUniqueSYCLKernel),
  // computed once, on the first SYCL binding.
  std::optional<llvm::StringMap<bool>> getterLocality;
  for (mlir::ModuleOp deviceMod :
       cir::getOffloadContainerDeviceModules(container)) {
    deviceMod.walk([&](cir::FuncOp kernel) {
      if (kernel.getCallingConv() != cir::CallingConv::SpirKernel)
        return;
      llvm::StringRef name = kernel.getName();
      // CUDA/HIP already own this name -- do not also bind it as SYCL.
      if (bindings.count(name))
        return;
      auto it = nameStrings.find(name);
      if (it == nameStrings.end())
        return;
      cir::GlobalOp nameString = it->second;
      // A device kernel the host never names by string launches nothing this
      // TU describes; skip it rather than record a host-less binding.
      if (!isUsedInHost(nameString))
        return;

      KernelBinding &binding = bindings[name];
      // First device module to reach this name owns the SYCL classification;
      // later modules (multi-arch) only append their kernel copy.
      if (binding.deviceKernels.empty()) {
        binding.kind = BindingKind::SYCLName;
        binding.hostNameString = nameString;
        binding.isRegistered = isRegisteredSYCLKernel(container, name);
        // The open rule: no other TU can launch the kernel (T26b).
        if (!getterLocality)
          getterLocality = getSYCLKernelInfoLocality(container);
        binding.isTUUnique =
            isTUUniqueSYCLKernel(container, kernel, name, &*getterLocality) &&
            !binding.isRegistered;
      }
      binding.deviceKernels.push_back(kernel);
    });
  }

  recordSYCLLaunchSites(container);
}

// Records one launch site per host kernel.info marker, resolving the kernel it
// names through the CompileTimeKernelInfo chain and the launch call it feeds.
// Bindings already carry their name-strings.
void KernelBindingTable::recordSYCLLaunchSites(mlir::ModuleOp container) {
  mlir::ModuleOp hostModule = cir::getOffloadContainerHostModule(container);

  // Index SYCL bindings by their name-string global, so a marker's resolved
  // string maps straight to the kernel it launches.
  llvm::DenseMap<mlir::Operation *, llvm::StringRef> byNameString;
  for (auto &binding : bindings)
    if (binding.second.kind == BindingKind::SYCLName &&
        binding.second.hostNameString)
      byNameString[binding.second.hostNameString.getOperation()] =
          binding.first;
  if (byNameString.empty())
    return;

  // The noalias candidates of each kernel, from its device parameters. Every
  // device copy (multi-arch) must agree on them.
  llvm::StringMap<NoAliasCandidates> noAliasCandidates;
  for (auto &binding : bindings) {
    if (binding.second.kind != BindingKind::SYCLName)
      continue;
    noAliasCandidates[binding.first] =
        getSYCLNoAliasCandidates(binding.second.deviceKernels);
    binding.second.syclNoAliasBlocked = noAliasCandidates[binding.first].poison;
  }

  hostModule.walk([&](cir::CallOp call) {
    if (!isGetDeviceKernelInfoCall(call))
      return;
    cir::GlobalOp nameString = resolveLaunchedNameString(call);
    if (!nameString)
      return;
    auto it = byNameString.find(nameString.getOperation());
    if (it == byNameString.end())
      return;
    // The launch consumes the DeviceKernelInfo* this marker produced. A
    // marker whose result never reaches a launch records nothing rather than
    // a site with no call.
    std::optional<RecoveredLaunch> launch = recoverLaunch(call->getResult(0));
    if (!launch)
      return;

    KernelBinding &binding = bindings[it->second];
    LaunchSite site;
    site.kind = BindingKind::SYCLName;
    site.stubCall = launch->anchor;
    site.syclMarkerCall = call;
    site.syclKernelName = it->second;
    site.syclClosure = launch->closure;
    binding.launchSites.push_back(site);
  });

  if (hostFactsOptions.mode == HostFactsMode::IPO)
    applyHostFacts(container, noAliasCandidates);
}

// --cir-offload-host-facts=ipo: the noalias verdicts, the scalar-capture
// constants and the geometry of SYCLHostFacts.
void KernelBindingTable::applyHostFacts(
    mlir::ModuleOp container,
    const llvm::StringMap<SYCLNoAliasCandidates> &candidates) {
  std::optional<SYCLHostFacts> ownFacts;
  if (!hostFactsCache)
    ownFacts.emplace(container, hostFactsOptions);
  const SYCLHostFacts &facts =
      hostFactsCache ? hostFactsCache->get(hostFactsOptions) : *ownFacts;
  for (auto &entry : bindings) {
    KernelBinding &binding = entry.second;
    if (binding.kind != BindingKind::SYCLName)
      continue;

    // The scalar-capture constants.
    for (auto &c : facts.constantFields(entry.first)) {
      if (c.second.value)
        binding.syclConstArgs[c.first()] = c.second.value;
      binding.syclConstNotes[c.first()] = c.second.note;
    }

    // The geometry, the same for every launch site.
    SYCLHostFacts::GeometryFact geometry = facts.geometry(entry.first);
    for (LaunchSite &site : binding.launchSites)
      site.syclNDRange = geometry.geometry;
    binding.syclGeometryNote = geometry.note;

    auto candIt = candidates.find(entry.first);
    if (candIt == candidates.end() || candIt->second.fields.empty())
      continue;
    for (auto &d : facts.noAliasFields(entry.first, candIt->second)) {
      if (d.second.ok)
        binding.syclNoAliasFields.insert(d.first());
      binding.syclNoAliasNotes[d.first()] = d.second.note;
    }
  }
}

const KernelBinding *
KernelBindingTable::lookup(llvm::StringRef kernelName) const {
  auto it = bindings.find(kernelName);
  return it == bindings.end() ? nullptr : &it->second;
}

llvm::ArrayRef<cir::FuncOp>
KernelBindingTable::getDeviceFuncs(cir::FuncOp hostFn) const {
  auto kernelNameAttr = hostFn->getAttrOfType<cir::CUDAKernelNameAttr>(
      cir::CUDAKernelNameAttr::getMnemonic());
  if (!kernelNameAttr)
    return {};
  const KernelBinding *binding = lookup(kernelNameAttr.getKernelName());
  return binding ? binding->deviceKernels : llvm::ArrayRef<cir::FuncOp>{};
}

llvm::ArrayRef<cir::LaunchSite>
KernelBindingTable::getLaunchSites(llvm::StringRef kernelName) const {
  const KernelBinding *binding = lookup(kernelName);
  return binding ? binding->launchSites : llvm::ArrayRef<cir::LaunchSite>{};
}

void KernelBindingTable::print(llvm::raw_ostream &os) const {
  os << "// ---- KernelBindingTable ----\n";
  os << "// size " << size() << " empty " << (empty() ? "true" : "false")
     << "\n";
  for (const auto &binding : bindings) {
    os << "// - kernel " << binding.first << "\n";
    if (binding.second.kind == BindingKind::SYCLName) {
      cir::GlobalOp nameString = binding.second.hostNameString;
      os << "//   host-name @" << nameString.getName() << " (sycl, "
         << (binding.second.isTUUnique ? "tu-unique" : "shared")
         << (binding.second.isRegistered ? ", registered" : "") << ")\n";
      if (!binding.second.syclConstArgs.empty()) {
        // Sorted for deterministic output (StringMap iteration is unordered).
        llvm::SmallVector<llvm::StringRef, 4> names;
        for (auto &c : binding.second.syclConstArgs)
          names.push_back(c.first());
        llvm::sort(names);
        os << "//   const-args";
        for (llvm::StringRef n : names) {
          os << " " << n << "=";
          binding.second.syclConstArgs.lookup(n).print(os);
        }
        os << "\n";
      }
      if (!binding.second.syclNoAliasFields.empty()) {
        // Sorted for deterministic output (StringSet iteration is unordered).
        llvm::SmallVector<llvm::StringRef, 4> names;
        for (const auto &f : binding.second.syclNoAliasFields)
          names.push_back(f.getKey());
        llvm::sort(names);
        os << "//   no-alias";
        for (llvm::StringRef n : names)
          os << " " << n;
        os << "\n";
      }
      if (!binding.second.syclNoAliasNotes.empty()) {
        llvm::SmallVector<llvm::StringRef, 4> names;
        for (const auto &n : binding.second.syclNoAliasNotes)
          names.push_back(n.getKey());
        llvm::sort(names);
        for (llvm::StringRef n : names)
          os << "//   no-alias-note " << n << ": "
             << binding.second.syclNoAliasNotes.lookup(n) << "\n";
      }
      if (!binding.second.syclNoAliasBlocked.empty() &&
          !binding.second.launchSites.empty())
        os << "//   no-alias-blocked: " << binding.second.syclNoAliasBlocked
           << "\n";
      if (hostFactsOptions.mode == HostFactsMode::IPO) {
        // Every scalar-capture candidate with its constant or why none.
        llvm::SmallVector<llvm::StringRef, 4> names;
        for (const auto &n : binding.second.syclConstNotes)
          names.push_back(n.getKey());
        llvm::sort(names);
        for (llvm::StringRef n : names) {
          os << "//   const-arg " << n << ": ";
          if (mlir::TypedAttr c = binding.second.syclConstArgs.lookup(n))
            c.print(os);
          else
            os << "none";
          std::string note = binding.second.syclConstNotes.lookup(n);
          if (!binding.second.syclConstArgs.lookup(n) && !note.empty())
            os << " (" << note << ")";
          os << "\n";
        }
        if (!binding.second.launchSites.empty()) {
          os << "//   geometry: "
             << formatSyclGeometry(
                    binding.second.launchSites.front().syclNDRange);
          if (!binding.second.syclGeometryNote.empty())
            os << " (" << binding.second.syclGeometryNote << ")";
          os << "\n";
        }
      }
    } else {
      cir::FuncOp stub = binding.second.hostStub;
      os << "//   host-stub @" << stub.getName() << "\n";
    }
    if (binding.second.deviceKernels.empty())
      os << "//   <no device kernel>\n";
    for (cir::FuncOp kernel : binding.second.deviceKernels) {
      auto mod = kernel->getParentOfType<mlir::ModuleOp>();
      os << "//   device @" << mod.getSymName().value_or("<unnamed>") << " : @"
         << kernel.getName() << "\n";
    }
    if (binding.second.launchSites.empty())
      os << "//   <no launch site>\n";
    for (const cir::LaunchSite &launch : binding.second.launchSites) {
      cir::CallOp stubCall = launch.stubCall;
      auto caller = stubCall->getParentOfType<cir::FuncOp>();
      os << "//   launch in @" << caller.getName();
      if (launch.kind == BindingKind::SYCLName) {
        os << " (sycl), closure "
           << (launch.getSYCLClosure() ? "traced" : "none");
        // Constant nd_range sizes, per dimension. "?" is a dimension that is
        // not a compile-time constant; an empty local list is a plain range
        // launch with no explicit work-group size.
        const cir::SyclGeometry &geom = launch.getSYCLNDRange();
        auto emit = [&](llvm::StringRef tag,
                        llvm::ArrayRef<std::optional<int64_t>> sizes) {
          os << ", " << tag << "=[";
          for (auto [i, s] : llvm::enumerate(sizes)) {
            if (i)
              os << ",";
            if (s)
              os << *s;
            else
              os << "?";
          }
          os << "]";
        };
        if (!geom.valid) {
          os << ", geometry none";
        } else {
          emit("global", geom.global);
          if (!geom.local.empty())
            emit("local", geom.local);
        }
        os << "\n";
        continue;
      }
      os << " : " << launch.getNumArgs() << " args";
      if (!launch.hasGeometry()) {
        os << ", <no geometry>\n";
        continue;
      }
      // "untraced" means the dimension was not recovered; "dynamic" means it
      // was, and holds a runtime value.
      auto label = [](const cir::LaunchSite::Dim3 &dim) -> llvm::StringRef {
        if (!dim.isTraced())
          return "untraced";
        return dim.isFullyConstant() ? "const" : "dynamic";
      };
      os << ", grid " << label(launch.getGridDim()) << ", block "
         << label(launch.getBlockDim()) << ", stream "
         << (launch.isDefaultStream() ? "default" : "explicit") << "\n";
    }
    os << "//\n";
  }
  os << "// ----------------------------\n";
}

//===----------------------------------------------------------------------===//
// Kernel copies (launch-path internalization)
//===----------------------------------------------------------------------===//

// The CompileTimeKernelInfo global a getDeviceKernelInfo getter reads to name
// its kernel: the (only) global it `cir.get_global`s whose symbol carries the
// CompileTimeKernelInfo tag. Mirrors resolveLaunchedNameString's walk.
static cir::GlobalOp findGetterCompileTimeInfo(cir::FuncOp getter) {
  cir::GlobalOp info;
  getter.walk([&](cir::GetGlobalOp gg) {
    auto g = lookupSymbol<cir::GlobalOp>(gg, gg.getNameAttr());
    if (g && g.getName().contains("CompileTimeKernelInfo"))
      info = g;
  });
  return info;
}

// Rebuild a CompileTimeKernelInfo initializer so its kernel-name string_view
// (the first member's `{global_view<@name>, len}`) points at `nameGlobal` and
// reports `nameLen`. Every other member -- notably the kernel_param_desc_t
// table -- is reused verbatim, since the clone has the original's signature.
static mlir::Attribute repointCompileTimeInfo(mlir::Attribute init,
                                              mlir::StringAttr nameGlobal,
                                              uint64_t nameLen,
                                              mlir::MLIRContext *ctx) {
  auto outer = mlir::dyn_cast<cir::ConstRecordAttr>(init);
  if (!outer || outer.getMembers().empty())
    return {};
  llvm::SmallVector<mlir::Attribute> members(outer.getMembers().begin(),
                                             outer.getMembers().end());
  auto sv = mlir::dyn_cast<cir::ConstRecordAttr>(members[0]);
  if (!sv || sv.getMembers().size() < 2)
    return {};
  auto view = mlir::dyn_cast<cir::GlobalViewAttr>(sv.getMembers()[0]);
  auto len = mlir::dyn_cast<cir::IntAttr>(sv.getMembers()[1]);
  if (!view || !len)
    return {};

  llvm::SmallVector<mlir::Attribute> svMembers(sv.getMembers().begin(),
                                               sv.getMembers().end());
  svMembers[0] = cir::GlobalViewAttr::get(
      view.getType(), mlir::FlatSymbolRefAttr::get(nameGlobal),
      view.getIndices());
  svMembers[1] = cir::IntAttr::get(len.getType(), nameLen);
  members[0] =
      cir::ConstRecordAttr::get(mlir::cast<cir::RecordType>(sv.getType()),
                                mlir::ArrayAttr::get(ctx, svMembers));
  return cir::ConstRecordAttr::get(mlir::cast<cir::RecordType>(outer.getType()),
                                   mlir::ArrayAttr::get(ctx, members));
}

namespace {
/// The kernel copy mintSYCLKernelCopy made: the device clones and the getter
/// copy (H3) the launches are to call.
struct MintedKernelCopy {
  struct {
    std::string name;
    llvm::SmallVector<cir::FuncOp, 2> deviceKernels;
  } spec;
  cir::FuncOp getter;
  // The originals the copy replaces (the name string, the
  // CompileTimeKernelInfo) and their copies H1, H2.
  cir::GlobalOp nameString, compileTimeInfo, h1, h2;
  cir::FuncOp originalGetter;
};
} // namespace

// The kernel copy of LocalKernelCopier::copy (see LocalKernelCopier): the
// device clones of `deviceKernels` under `cloneName`, the name string (H1),
// the CompileTimeKernelInfo (H2), the getter's static-local cluster and the
// getter copy (H3). `getter` may be null (then nothing past H1 is made).
// With `local`, the host
// artifacts are internal, named `<original><suffix>` and tagged with
// cir.offload.internalized (launch-path internalization, T26).
// `hostTable` (optional) is a symbol table of the host module to insert
// into, and `onlyInGetter` (optional) tells whether every use of a global is
// in the getter (both computed once by a caller that copies many kernels;
// by default they are computed here).
static std::optional<MintedKernelCopy> mintSYCLKernelCopy(
    mlir::ModuleOp container, llvm::ArrayRef<cir::FuncOp> deviceKernels,
    cir::GlobalOp nameStr, cir::FuncOp getter, llvm::StringRef cloneName,
    bool local, llvm::StringRef suffix, mlir::SymbolTable *hostTable = nullptr,
    llvm::function_ref<bool(cir::GlobalOp, cir::FuncOp)> onlyInGetter = {}) {
  mlir::ModuleOp hostModule = cir::getOffloadContainerHostModule(container);
  mlir::MLIRContext *ctx = container.getContext();
  MintedKernelCopy minted;
  auto &spec = minted.spec;
  spec.name = cloneName.str();
  // A host artifact of a local copy: internal, renamed, tagged.
  auto localize = [&](mlir::Operation *op) {
    if (!local)
      return;
    mlir::StringAttr original = mlir::SymbolTable::getSymbolName(op);
    mlir::SymbolTable::setSymbolName(op, (original.getValue() + suffix).str());
    if (auto func = mlir::dyn_cast<cir::FuncOp>(op)) {
      func.setLinkage(cir::GlobalLinkageKind::InternalLinkage);
      func.removeComdatAttr();
    } else if (auto global = mlir::dyn_cast<cir::GlobalOp>(op)) {
      global.setLinkage(cir::GlobalLinkageKind::InternalLinkage);
      global.setComdat(false);
    }
    mlir::SymbolTable::setSymbolVisibility(
        op, mlir::SymbolTable::Visibility::Private);
    op->setAttr("cir.offload.internalized", original);
  };

  // 1. Device clone(s): one per arch module, external linkage inherited from
  // the original (an internal clone is dead-stripped at sycl-post-link), tagged
  // with the combine-private single-caller provenance marker.
  auto specPrivate = mlir::UnitAttr::get(ctx);
  for (cir::FuncOp kernel : deviceKernels) {
    auto clone = mlir::cast<cir::FuncOp>(kernel->clone());
    mlir::SymbolTable::setSymbolName(clone, cloneName);
    clone->setAttr("cir.specialization_private", specPrivate);
    mlir::OpBuilder b(kernel.getOperation());
    b.setInsertionPointAfter(kernel.getOperation());
    b.insert(clone);
    spec.deviceKernels.push_back(clone);
  }

  std::optional<mlir::SymbolTable> ownHostSyms;
  if (!hostTable)
    ownHostSyms.emplace(hostModule);
  mlir::SymbolTable &hostSyms = hostTable ? *hostTable : *ownHostSyms;

  // 2. H1: a mangled-name string global holding the clone name. Cloned from the
  // original name-string so linkage/visibility match, then retyped and
  // re-inited for the (different-length) clone name.
  cir::GlobalOp h1 = mlir::cast<cir::GlobalOp>(nameStr->clone());
  auto origArrTy = mlir::cast<cir::ArrayType>(nameStr.getSymType());
  mlir::Type eltTy = origArrTy.getElementType();
  uint64_t nameLen = cloneName.size();
  auto fullTy = cir::ArrayType::get(eltTy, nameLen + 1);
  auto strAttr =
      mlir::StringAttr::get(cloneName, cir::ArrayType::get(eltTy, nameLen));
  h1.setSymType(fullTy);
  h1.setInitialValueAttr(cir::ConstArrayAttr::get(fullTy, strAttr));
  localize(h1);
  mlir::StringAttr h1Name = hostSyms.insert(h1);
  minted.nameString = nameStr;
  minted.h1 = h1;

  // 3. The getter this launch marker calls and the CompileTimeKernelInfo it
  // names.
  if (!getter)
    return std::nullopt;
  cir::GlobalOp ctki = findGetterCompileTimeInfo(getter);
  if (!ctki || !ctki.getInitialValue())
    return std::nullopt;

  // 4. H2: a CompileTimeKernelInfo clone repointed at H1.
  cir::GlobalOp h2 = mlir::cast<cir::GlobalOp>(ctki->clone());
  mlir::Attribute h2Init =
      repointCompileTimeInfo(*ctki.getInitialValue(), h1Name, nameLen, ctx);
  if (!h2Init)
    return std::nullopt;
  h2.setInitialValueAttr(h2Init);
  localize(h2);
  mlir::StringAttr h2Name = hostSyms.insert(h2);
  minted.compileTimeInfo = ctki;
  minted.h2 = h2;

  // 5. Clone the getter's per-instantiation private cluster (the static-local
  // DeviceKernelInfo* singleton and its guard) so the clone lazily registers
  // its OWN DeviceKernelInfo keyed to H2/H1 -- sharing the original's would
  // cache the wrong (original-named) kernel. A cluster global is one the getter
  // references by get_global, that is not the CompileTimeKernelInfo, and whose
  // every use in the host module is inside the getter (i.e. a function-local
  // static / guard).
  llvm::DenseMap<mlir::StringAttr, mlir::StringAttr> remap;
  remap[ctki.getSymNameAttr()] = h2Name;
  llvm::SmallVector<cir::GlobalOp, 2> clonedCluster;
  llvm::SmallPtrSet<mlir::Operation *, 4> seen;
  getter.walk([&](cir::GetGlobalOp gg) {
    auto g = hostTable
                 ? hostTable->lookup<cir::GlobalOp>(gg.getNameAttr().getAttr())
                 : lookupSymbol<cir::GlobalOp>(gg, gg.getNameAttr());
    if (!g || g == ctki || !seen.insert(g).second)
      return;
    if (onlyInGetter) {
      if (!onlyInGetter(g, getter))
        return; // Referenced outside the getter: shared, do not clone.
    } else {
      auto uses = mlir::SymbolTable::getSymbolUses(g, hostModule);
      if (!uses)
        return;
      for (const mlir::SymbolTable::SymbolUse &u : *uses)
        if (!getter->isAncestor(u.getUser()))
          return; // Referenced outside the getter: shared, do not clone.
    }
    auto c = mlir::cast<cir::GlobalOp>(g->clone());
    localize(c);
    mlir::StringAttr cName = hostSyms.insert(c);
    remap[g.getSymNameAttr()] = cName;
    clonedCluster.push_back(c);
  });

  // Repoint each cloned static-local's guard association to the cloned guard.
  for (cir::GlobalOp c : clonedCluster) {
    if (auto guard = c.getStaticLocalGuard()) {
      auto it = remap.find(guard->getName());
      if (it != remap.end())
        c.setStaticLocalGuardAttr(
            cir::StaticLocalGuardAttr::get(ctx, it->second));
    }
  }

  // 6. H3: a getter clone whose get_global references are remapped onto H2 and
  // the cloned cluster.
  minted.originalGetter = getter;
  cir::FuncOp h3 = mlir::cast<cir::FuncOp>(getter->clone());
  localize(h3);
  mlir::StringAttr h3Name = hostSyms.insert(h3);
  h3.walk([&](cir::GetGlobalOp gg) {
    auto it = remap.find(gg.getNameAttr().getAttr());
    if (it != remap.end())
      gg.setNameAttr(mlir::FlatSymbolRefAttr::get(it->second));
  });
  (void)h3Name;
  minted.getter = h3;
  return minted;
}

//===----------------------------------------------------------------------===//
// The kernel-name reads of a specialized launch path (T26c)
//===----------------------------------------------------------------------===//
//
// The SYCL runtime names a launch twice: by the DeviceKernelInfo the marker
// returns, and by the name the launch implementation stores in the handler
// (handler::setDeviceKernelInfo: `constexpr auto Info =
// CompileTimeKernelInfo<K>; MKernelName = Info.Name;`), and
// handler::finalize() asserts that the two agree. CIRGen materializes the
// constexpr local as a private constant `__const.<function>.Info` whose
// initializer is a copy of the CompileTimeKernelInfo value: a third copy of
// the kernel name, besides the CompileTimeKernelInfo (H2) and the string
// (H1). Redirecting the marker alone leaves it, and every other such copy,
// naming the original kernel.
//
// So after the markers of a kernel copy are redirected, every reference to
// the kernel's naming artifacts in the SYCL launch implementation on the
// paths to those markers is redirected too:
//  - the name string, the CompileTimeKernelInfo and the kernel.info getter:
//    to H1, H2, H3;
//  - every constant global whose initializer refers (through
//    #cir.global_view or a symbol reference, transitively) to one of them:
//    to a copy whose initializer refers to the replacements, with the length
//    of a string_view {name, length} updated (the clone's name is longer);
//  - the same in the attributes of the functions' own ops (a constant
//    record stored directly instead of copied from a `__const` global).
// The originals are untouched: other launches (this TU's other paths,
// other TUs) keep using them.
//
// Which functions: the functions holding the markers and, transitively,
// their callers that belong to the SYCL implementation (in namespace sycl,
// or with a SYCL host-API role), up to the user's code: the SYCL library
// reads CompileTimeKernelInfo<K> only to launch K (the handler's name, the
// checks of the kernel's size and parameters). User code above the launch
// API is not rewritten: a name it reads (e.g. get_kernel_id<K>() for a
// kernel bundle) is not the launch's, and a by-name launch of the copy
// would launch it with arguments the analysis did not see.
//
// The post-condition, checked after the rewrite: no such function refers to
// an original naming artifact, to a global carrying one, or calls a
// function outside the set that does (a remark for each; the lit tests
// check that none is emitted).

namespace {
/// The top-level ops of a host module that refer to each symbol.
using SymbolHolders =
    llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<mlir::Operation *, 1>>;
} // namespace

static SymbolHolders collectSymbolHolders(mlir::ModuleOp host) {
  SymbolHolders holders;
  if (std::optional<mlir::SymbolTable::UseRange> uses =
          mlir::SymbolTable::getSymbolUses(&host.getBodyRegion()))
    for (const mlir::SymbolTable::SymbolUse &use : *uses) {
      mlir::Operation *holder = use.getUser();
      while (holder && holder->getParentOp() != host.getOperation())
        holder = holder->getParentOp();
      auto &list = holders[use.getSymbolRef().getRootReference()];
      if (holder && !llvm::is_contained(list, holder))
        list.push_back(holder);
    }
  return holders;
}

/// Whether \p func is part of the SYCL implementation (namespace sycl, a
/// lambda in a function of it, or a SYCL host-API role); a copy keeps its
/// original's name with a suffix.
static bool isSYCLImplementation(cir::FuncOp func) {
  if (func->hasAttr(cir::CIRDialect::getSYCLHostAPIAttrName()))
    return true;
  llvm::StringRef name = func.getSymName();
  return name.starts_with("_ZN4sycl") || name.starts_with("_ZNK4sycl") ||
         name.starts_with("_ZZN4sycl") || name.starts_with("_ZZNK4sycl");
}

/// Redirect the kernel-name reads of the launch paths to the markers
/// \p markers of the kernel copy \p minted (see above). With \p local, the
/// global copies are internal, named `<original><suffix>` and tagged
/// cir.offload.internalized, and every rewritten op is tagged
/// kInternalizedNameReadAttr (so that pruneInternalizedPaths can undo it
/// with the markers when the kernel copy is not used). Returns the number of
/// post-condition violations (each remarked).
static unsigned
redirectKernelNameReads(mlir::ModuleOp host, mlir::SymbolTable &hostSyms,
                        const SymbolHolders &holders, MintedKernelCopy minted,
                        llvm::ArrayRef<mlir::Operation *> markers, bool local,
                        llvm::StringRef suffix) {
  mlir::MLIRContext *ctx = host.getContext();
  cir::GlobalOp nameStr = minted.nameString;
  mlir::StringAttr nameSym = nameStr.getSymNameAttr();
  uint64_t oldLen = getGlobalStringValue(nameStr).size();
  uint64_t newLen = minted.spec.name.size();
  mlir::StringAttr getterCopy = minted.getter.getSymNameAttr();

  // The originals, and the globals that carry one (backward over the
  // initializers that refer to them).
  llvm::SmallVector<mlir::StringAttr, 3> originals{
      nameSym, minted.compileTimeInfo.getSymNameAttr(),
      minted.originalGetter.getSymNameAttr()};
  llvm::DenseMap<mlir::StringAttr, mlir::StringAttr> replacement;
  replacement[originals[0]] = minted.h1.getSymNameAttr();
  replacement[originals[1]] = minted.h2.getSymNameAttr();
  replacement[originals[2]] = getterCopy;
  llvm::DenseMap<mlir::StringAttr, cir::GlobalOp> carriers;
  {
    llvm::SmallVector<mlir::StringAttr> work(originals.begin(),
                                             originals.end());
    while (!work.empty()) {
      mlir::StringAttr sym = work.pop_back_val();
      auto it = holders.find(sym);
      if (it == holders.end())
        continue;
      for (mlir::Operation *holder : it->second)
        if (auto global = mlir::dyn_cast<cir::GlobalOp>(holder))
          if (carriers.try_emplace(global.getSymNameAttr(), global).second)
            work.push_back(global.getSymNameAttr());
    }
  }
  auto isOriginal = [&](mlir::StringAttr sym) {
    return llvm::is_contained(originals, sym) || carriers.count(sym);
  };

  // The functions: the marker holders and their SYCL-implementation
  // callers.
  llvm::SmallSetVector<mlir::Operation *, 8> scope;
  for (mlir::Operation *marker : markers)
    if (auto holder = marker->getParentOfType<cir::FuncOp>())
      scope.insert(holder);
  for (unsigned i = 0; i != scope.size(); ++i) {
    auto func = mlir::cast<cir::FuncOp>(scope[i]);
    auto it = holders.find(func.getSymNameAttr());
    if (it == holders.end())
      continue;
    for (mlir::Operation *holder : it->second)
      if (auto caller = mlir::dyn_cast<cir::FuncOp>(holder))
        if (!caller.isDeclaration() && isSYCLImplementation(caller))
          scope.insert(caller);
  }

  // The replacement of a symbol: H1-H3, or the copy of a carrier (made on
  // first use; null for a carrier that is not a constant). `oldLength`
  // receives the length a string_view {name, length} had, if one was
  // updated.
  std::function<mlir::Attribute(mlir::Attribute, std::optional<uint64_t> &)>
      rewrite;
  std::function<mlir::StringAttr(mlir::StringAttr)> replace =
      [&](mlir::StringAttr sym) -> mlir::StringAttr {
    if (mlir::StringAttr r = replacement.lookup(sym))
      return r;
    cir::GlobalOp global = carriers.lookup(sym);
    if (!global || !global.getConstant() || !global.getInitialValue())
      return {};
    auto copy = mlir::cast<cir::GlobalOp>(global->clone());
    copy.setLinkage(cir::GlobalLinkageKind::InternalLinkage);
    copy.setComdat(false);
    mlir::SymbolTable::setSymbolVisibility(
        copy, mlir::SymbolTable::Visibility::Private);
    if (local) {
      copy.setSymName((global.getSymName() + suffix).str());
      copy->setAttr(cir::kInternalizedAttr, sym);
    }
    mlir::StringAttr name =
        hostSyms.insert(copy, std::next(mlir::Block::iterator(global)));
    replacement[sym] = name;
    std::optional<uint64_t> ignored;
    copy.setInitialValueAttr(rewrite(*global.getInitialValue(), ignored));
    return name;
  };
  rewrite = [&](mlir::Attribute attr,
                std::optional<uint64_t> &oldLength) -> mlir::Attribute {
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement(
        [&](mlir::FlatSymbolRefAttr ref) -> std::optional<mlir::Attribute> {
          if (!isOriginal(ref.getAttr()))
            return std::nullopt;
          mlir::StringAttr to = replace(ref.getAttr());
          if (!to)
            return std::nullopt;
          return mlir::FlatSymbolRefAttr::get(to);
        });
    // A string_view {name, length} (a two-member record, or a length that
    // is the name's): the length of the clone's name.
    replacer.addReplacement(
        [&](cir::ConstRecordAttr rec)
            -> std::optional<std::pair<mlir::Attribute, mlir::WalkResult>> {
          auto members = llvm::to_vector(rec.getMembers());
          bool changed = false;
          for (unsigned i = 0; i + 1 < members.size(); ++i) {
            auto view = mlir::dyn_cast<cir::GlobalViewAttr>(members[i]);
            auto len = mlir::dyn_cast<cir::IntAttr>(members[i + 1]);
            if (!view || !len ||
                view.getSymbol().getRootReference() != nameSym ||
                (members.size() != 2 && len.getUInt() != oldLen) ||
                len.getUInt() == newLen)
              continue;
            if (mlir::ArrayAttr indices = view.getIndices())
              if (llvm::any_of(indices, [](mlir::Attribute index) {
                    auto i = mlir::dyn_cast<cir::IntAttr>(index);
                    return !i || i.getSInt() != 0;
                  }))
                continue;
            oldLength = len.getUInt();
            members[i + 1] = cir::IntAttr::get(len.getType(), newLen);
            changed = true;
          }
          if (!changed)
            return std::nullopt;
          return std::make_pair(cir::ConstRecordAttr::get(
                                    mlir::cast<cir::RecordType>(rec.getType()),
                                    mlir::ArrayAttr::get(ctx, members)),
                                mlir::WalkResult::advance());
        });
    return replacer.replace(attr);
  };

  // Rewrite the functions' references. In local mode each rewritten op is
  // tagged with what undoes it (no symbol reference: the tag must not be a
  // use of the originals): {getter = H3, refs = {copy = original, ...}[,
  // string = H1, lengths = [new, old]]}.
  for (mlir::Operation *op : scope) {
    auto func = mlir::cast<cir::FuncOp>(op);
    llvm::SmallSetVector<mlir::Operation *, 4> users;
    if (std::optional<mlir::SymbolTable::UseRange> uses =
            mlir::SymbolTable::getSymbolUses(&func.getBody()))
      for (const mlir::SymbolTable::SymbolUse &use : *uses)
        if (isOriginal(use.getSymbolRef().getRootReference()))
          users.insert(use.getUser());
    for (mlir::Operation *user : users) {
      llvm::SmallVector<mlir::NamedAttribute, 2> refs;
      std::optional<uint64_t> oldLength;
      if (auto addr = mlir::dyn_cast<cir::GetGlobalOp>(user)) {
        mlir::StringAttr from = addr.getNameAttr().getAttr();
        mlir::StringAttr to = replace(from);
        if (!to)
          continue;
        addr.setNameAttr(mlir::FlatSymbolRefAttr::get(to));
        refs.push_back(mlir::NamedAttribute(to, from));
        // H1 has the clone name's type: retype, cast back for the users.
        auto global = hostSyms.lookup<cir::GlobalOp>(to);
        auto ptrTy = mlir::cast<cir::PointerType>(addr.getType());
        if (global && global.getSymType() != ptrTy.getPointee()) {
          addr.getAddr().setType(
              cir::PointerType::get(global.getSymType(), ptrTy.getAddrSpace()));
          mlir::OpBuilder b(ctx);
          b.setInsertionPointAfter(addr);
          auto cast = cir::CastOp::create(b, addr.getLoc(), ptrTy,
                                          cir::CastKind::bitcast, addr);
          addr.getAddr().replaceAllUsesExcept(cast, cast);
        }
      } else {
        llvm::SmallVector<mlir::StringAttr, 2> before;
        if (std::optional<mlir::SymbolTable::UseRange> uses =
                mlir::SymbolTable::getSymbolUses(user))
          for (const mlir::SymbolTable::SymbolUse &use : *uses)
            if (isOriginal(use.getSymbolRef().getRootReference()))
              before.push_back(use.getSymbolRef().getRootReference());
        mlir::NamedAttrList attrs(user->getAttrDictionary());
        bool changed = false;
        for (mlir::NamedAttribute &attr : attrs) {
          mlir::Attribute value = rewrite(attr.getValue(), oldLength);
          if (value != attr.getValue()) {
            attr.setValue(value);
            changed = true;
          }
        }
        if (!changed)
          continue;
        user->setAttrs(attrs.getDictionary(ctx));
        for (mlir::StringAttr from : before)
          if (mlir::StringAttr to = replacement.lookup(from))
            refs.push_back(mlir::NamedAttribute(to, from));
      }
      if (!local)
        continue;
      mlir::Builder b(ctx);
      llvm::sort(refs, [](mlir::NamedAttribute x, mlir::NamedAttribute y) {
        return x.getName().strref() < y.getName().strref();
      });
      refs.erase(
          std::unique(refs.begin(), refs.end(),
                      [](mlir::NamedAttribute x, mlir::NamedAttribute y) {
                        return x.getName() == y.getName();
                      }),
          refs.end());
      llvm::SmallVector<mlir::NamedAttribute, 4> tag{
          b.getNamedAttr("getter", getterCopy),
          b.getNamedAttr("refs", b.getDictionaryAttr(refs))};
      if (oldLength) {
        tag.push_back(b.getNamedAttr("string", minted.h1.getSymNameAttr()));
        tag.push_back(b.getNamedAttr(
            "lengths",
            b.getI64ArrayAttr({(int64_t)newLen, (int64_t)*oldLength})));
      }
      user->setAttr(cir::kInternalizedNameReadAttr, b.getDictionaryAttr(tag));
    }
  }

  // The post-condition. A function outside the set "names" the kernel if it
  // refers to an original or a carrier, or calls one that does.
  llvm::DenseMap<mlir::Operation *, mlir::StringAttr> namesVia;
  llvm::DenseSet<mlir::Operation *> visiting;
  std::function<mlir::StringAttr(cir::FuncOp)> names =
      [&](cir::FuncOp func) -> mlir::StringAttr {
    auto it = namesVia.find(func);
    if (it != namesVia.end())
      return it->second;
    if (func.isDeclaration() || !visiting.insert(func).second)
      return {};
    mlir::StringAttr found;
    if (std::optional<mlir::SymbolTable::UseRange> uses =
            mlir::SymbolTable::getSymbolUses(&func.getBody()))
      for (const mlir::SymbolTable::SymbolUse &use : *uses) {
        mlir::StringAttr sym = use.getSymbolRef().getRootReference();
        if (isOriginal(sym)) {
          found = sym;
          break;
        }
        auto callee = hostSyms.lookup<cir::FuncOp>(sym);
        if (callee && callee != func && !scope.contains(callee) &&
            names(callee)) {
          found = sym;
          break;
        }
      }
    visiting.erase(func);
    namesVia[func] = found;
    return found;
  };
  unsigned violations = 0;
  for (mlir::Operation *op : scope) {
    auto func = mlir::cast<cir::FuncOp>(op);
    std::optional<mlir::SymbolTable::UseRange> uses =
        mlir::SymbolTable::getSymbolUses(&func.getBody());
    if (!uses)
      continue;
    llvm::SmallSetVector<mlir::StringAttr, 2> direct, through;
    for (const mlir::SymbolTable::SymbolUse &use : *uses) {
      mlir::StringAttr sym = use.getSymbolRef().getRootReference();
      if (isOriginal(sym)) {
        direct.insert(sym);
        continue;
      }
      auto callee = hostSyms.lookup<cir::FuncOp>(sym);
      if (callee && !scope.contains(callee) && names(callee))
        through.insert(sym);
    }
    if (direct.empty() && through.empty())
      continue;
    ++violations;
    mlir::InFlightDiagnostic diag = mlir::emitRemark(func.getLoc());
    diag << "kernel-name-redirect: @" << func.getSymName()
         << " on a launch path of @" << minted.spec.name
         << " still names the original kernel @"
         << getGlobalStringValue(nameStr);
    for (mlir::StringAttr sym : direct)
      diag << "; refers to @" << sym.getValue();
    for (mlir::StringAttr sym : through)
      diag << "; calls @" << sym.getValue() << ", which names it"
           << " (through @"
           << names(hostSyms.lookup<cir::FuncOp>(sym)).getValue() << ")";
  }
  return violations;
}

cir::LocalKernelCopier::LocalKernelCopier(mlir::ModuleOp container)
    : container(container),
      hostSyms(cir::getOffloadContainerHostModule(container)) {
  for (mlir::ModuleOp dm : cir::getOffloadContainerDeviceModules(container))
    deviceSyms.push_back(std::make_unique<mlir::SymbolTable>(dm));
  // The top-level ops of the host module that refer to each symbol.
  holders = collectSymbolHolders(cir::getOffloadContainerHostModule(container));
}

cir::LocalKernelCopier::~LocalKernelCopier() = default;

std::optional<cir::LocalKernelCopy>
cir::LocalKernelCopier::copy(cir::FuncOp getter, llvm::StringRef cloneName,
                             llvm::StringRef suffix) {
  cir::GlobalOp ctki = getter ? findGetterCompileTimeInfo(getter) : nullptr;
  std::optional<mlir::Attribute> init =
      ctki ? ctki.getInitialValue() : std::nullopt;
  // The name string: the global the CompileTimeKernelInfo's string_view
  // points to (repointCompileTimeInfo's shape, which needs the string_view's
  // {pointer, length}).
  auto outer = init ? mlir::dyn_cast<cir::ConstRecordAttr>(*init) : nullptr;
  auto sv = outer && !outer.getMembers().empty()
                ? mlir::dyn_cast<cir::ConstRecordAttr>(outer.getMembers()[0])
                : nullptr;
  auto view = sv && sv.getMembers().size() >= 2 &&
                      mlir::isa<cir::IntAttr>(sv.getMembers()[1])
                  ? mlir::dyn_cast<cir::GlobalViewAttr>(sv.getMembers()[0])
                  : nullptr;
  if (!view)
    return std::nullopt;
  auto nameStr =
      hostSyms.lookup<cir::GlobalOp>(view.getSymbol().getRootReference());
  llvm::StringRef kernelName =
      nameStr ? getGlobalStringValue(nameStr) : llvm::StringRef();
  if (kernelName.empty())
    return std::nullopt;
  llvm::SmallVector<cir::FuncOp, 2> deviceKernels;
  for (const std::unique_ptr<mlir::SymbolTable> &table : deviceSyms) {
    if (table->lookup(cloneName))
      return std::nullopt;
    if (auto kernel = table->lookup<cir::FuncOp>(kernelName))
      if (!kernel.isDeclaration() &&
          kernel.getCallingConv() == cir::CallingConv::SpirKernel)
        deviceKernels.push_back(kernel);
  }
  if (deviceKernels.empty())
    return std::nullopt;
  auto onlyInGetter = [&](cir::GlobalOp g, cir::FuncOp f) {
    auto it = holders.find(g.getSymNameAttr());
    return it != holders.end() &&
           llvm::all_of(it->second, [&](mlir::Operation *holder) {
             return holder == f.getOperation();
           });
  };
  std::optional<MintedKernelCopy> minted =
      mintSYCLKernelCopy(container, deviceKernels, nameStr, getter, cloneName,
                         /*local=*/true, suffix, &hostSyms, onlyInGetter);
  if (!minted)
    return std::nullopt;
  cir::LocalKernelCopy copy;
  copy.original = kernelName.str();
  copy.name = minted->spec.name;
  copy.deviceKernels = std::move(minted->spec.deviceKernels);
  copy.getter = minted->getter;
  copy.nameString = minted->nameString;
  copy.compileTimeInfo = minted->compileTimeInfo;
  copy.h1 = minted->h1;
  copy.h2 = minted->h2;
  copy.originalGetter = minted->originalGetter;
  return copy;
}

unsigned cir::LocalKernelCopier::redirectNameReads(
    const cir::LocalKernelCopy &copy, llvm::ArrayRef<mlir::Operation *> markers,
    llvm::StringRef suffix) {
  MintedKernelCopy minted;
  minted.spec.name = copy.name;
  minted.getter = copy.getter;
  minted.nameString = copy.nameString;
  minted.compileTimeInfo = copy.compileTimeInfo;
  minted.h1 = copy.h1;
  minted.h2 = copy.h2;
  minted.originalGetter = copy.originalGetter;
  return redirectKernelNameReads(cir::getOffloadContainerHostModule(container),
                                 hostSyms, holders, minted, markers,
                                 /*local=*/true, suffix);
}
