//===- KernelBindingTable.cpp - Host<->device kernel bindings -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLKernelParams.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "kernel-binding-table"

using namespace cir;

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
  auto fn =
      mlir::SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(call, callee);
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

// A mangled name embeds an unnamed closure when it carries the Itanium
// unnamed-lambda encoding `Ul...E` (closure-type) or clang's `$_` local-lambda
// spelling. Named free functions (e.g. `__sycl_kernel_vec_add`) and named
// kernel types never do. This is a name-only fallback for TU-local
// optimizability: it fires when the kernel NAME is itself an anonymous lambda
// mangling, but it cannot see a named tag (`parallel_for<class Tag>(lambda)`)
// whose symbol is `_ZTS<Tag>` even though the closure is a lambda. The closure
// signal below (isLambdaClosureKernel) supersedes it whenever the device
// kernel body is available.
static bool hasAnonymousClosure(llvm::StringRef name) {
  if (name.contains("$_"))
    return true;
  // A closure prefix `Ul` sits at a mangling boundary (after a scope or
  // substitution terminator), never inside a length-counted source name, so
  // require the preceding char to be one -- this rejects identifiers like
  // "Full" or "Ultra".
  for (size_t pos = name.find("Ul"); pos != llvm::StringRef::npos;
       pos = name.find("Ul", pos + 2)) {
    if (pos == 0 || name[pos - 1] == 'E' || name[pos - 1] == '_')
      return true;
  }
  return false;
}

// Whether a CIR record type NAME denotes a lambda / anonymous closure rather
// than a spellable class. CIRGen names a record with no C++ identifier (a
// lambda closure, an anonymous struct) `anon` or `anon.<N>` -- see
// CIRGenBuilder::getUniqueAnonRecordName -- whereas a named struct/functor
// keeps its printed qualified name. The AS-lowering pass may prefix an
// identified record that carries address-space members with `__cir_lowered.`
// (see TargetLowering.cpp); at the combine stage the device closure has already
// been through it, so strip that prefix before testing. This is what tells a
// lambda closure apart from a genuine functor: a functor has a nameable type
// and could be launched from another TU, so it must stay non-TU-local.
static bool isAnonymousClosureRecordName(llvm::StringRef name) {
  name.consume_front("__cir_lowered.");
  if (name == "anon")
    return true;
  if (!name.consume_front("anon."))
    return false;
  return !name.empty() && llvm::all_of(name, llvm::isDigit);
}

// Whether `v`, followed through the offset-0 reinterprets CIRGen inserts on a
// closure `this` (address-space cast, bitcast, base_class_addr), is passed as
// the receiver (operand 0) of some call. The kernel object is the one local
// record whose operator() the entry point invokes, so it is the record alloca
// reaching a call in receiver position; the SYCL item/id temporaries are passed
// by value (loaded first), never as a pointer receiver.
static bool valueFeedsCallReceiver(mlir::Value v) {
  llvm::SmallVector<mlir::Value, 4> worklist{v};
  llvm::SmallPtrSet<mlir::Operation *, 8> seen;
  while (!worklist.empty()) {
    mlir::Value cur = worklist.pop_back_val();
    for (mlir::Operation *user : cur.getUsers()) {
      if (auto call = mlir::dyn_cast<cir::CallOp>(user)) {
        mlir::OperandRange args = call.getArgOperands();
        if (!args.empty() && args[0] == cur)
          return true;
        continue;
      }
      if (mlir::isa<cir::CastOp, cir::BaseClassAddrOp>(user))
        if (seen.insert(user).second)
          worklist.push_back(user->getResult(0));
    }
  }
  return false;
}

// The closure record type a SYCL device kernel reconstructs and invokes, or a
// null type if the kernel has no closure object (a named free-function kernel,
// or a minimal fixture with an empty body). The kernel entry point materializes
// the kernel-object clone as a local alloca and calls its operator(); that
// alloca -- named `__SYCLKernel` for a lambda or after the functor otherwise --
// is the sole record alloca reaching a call as its receiver.
static cir::RecordType getSYCLClosureRecord(cir::FuncOp kernel) {
  cir::RecordType found;
  kernel.walk([&](cir::AllocaOp alloca) {
    if (found)
      return;
    auto rec = mlir::dyn_cast<cir::RecordType>(alloca.getAllocaType());
    if (!rec || !rec.getName())
      return;
    if (valueFeedsCallReceiver(alloca.getResult()))
      found = rec;
  });
  return found;
}

// The TU-local identity test: a kernel is optimizable when its CLOSURE is a
// lambda (a compiler-generated, TU-unique, unspellable type), regardless of the
// kernel-name tag. `parallel_for<class Tag>(lambda)` has a named `_ZTS<Tag>`
// symbol but an anonymous closure record, and is TU-local; a genuine functor
// (a named struct with operator(), possibly defined in a header and launchable
// from other TUs) has a nameable closure record and stays conservative. When
// the device kernel body carries no closure object -- an empty lit fixture, or
// a free-function kernel -- fall back to the name-only heuristic.
static bool isLambdaClosureKernel(cir::FuncOp kernel,
                                  llvm::StringRef symbolName) {
  if (cir::RecordType closure = getSYCLClosureRecord(kernel))
    return isAnonymousClosureRecordName(closure.getName().getValue());
  return hasAnonymousClosure(symbolName);
}

// Whether `name` appears as a registered (by-name / bundle-exposed) kernel in
// any module's `cir.sycl_registered_kernels` attribute -- an array of
// [user-name, mangled-name] string pairs. Such a kernel is reachable by name
// across TUs, so it can never be treated as TU-local.
static bool isRegisteredKernel(mlir::ModuleOp container, llvm::StringRef name) {
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

KernelBindingTable::KernelBindingTable(mlir::ModuleOp container) {
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

// Operand roles of the runtime launch call
// `submit_kernel_direct_with_event_impl(event& sret, queue&, nd_range_view&,
// HostKernelRefBase&, DeviceKernelInfo*, span, PropsHolder&, code_location&,
// bool)` -- a stable __SYCL_EXPORT ABI, so positions are fixed.
enum : unsigned {
  SyclGeometryOperand = 2,        // nd_range_view*
  SyclClosureOperand = 3,         // HostKernelRefBase*
  SyclDeviceKernelInfoOperand = 4 // DeviceKernelInfo*
};

// Whether `call` is the templated `sycl::detail::getDeviceKernelInfo<Kernel>()`
// launch marker: the template-args ('I') follow the name and it takes no
// operands. The non-templated overload taking a CompileTimeKernelInfo ref
// ('...InfoE...') is reached from inside it and is not a marker.
static bool isGetDeviceKernelInfoCall(cir::CallOp call) {
  std::optional<llvm::StringRef> callee = call.getCallee();
  if (!callee)
    return false;
  return callee->starts_with("_ZN4sycl3_V16detail19getDeviceKernelInfoI") &&
         call.getArgOperands().empty() && call->getNumResults() == 1;
}

// Whether `call` is the runtime launch `submit_kernel_direct_with_event_impl`.
static bool isSubmitKernelDirectImpl(cir::CallOp call) {
  std::optional<llvm::StringRef> callee = call.getCallee();
  return callee && callee->starts_with(
                       "_ZN4sycl3_V136submit_kernel_direct_with_event_impl");
}

//===----------------------------------------------------------------------===//
// SYCL handler-submit launch shape
//===----------------------------------------------------------------------===//
//
// The direct `q.parallel_for(...)` shortcut launches through
// submit_kernel_direct_with_event_impl. The far more common
// `q.submit([&](handler& h){ h.parallel_for(...); })` form launches through a
// set of handler methods instead, tied together by the shared `handler* this`:
//
//   handler::setDeviceKernelInfo<K>(this, closurePtr):     // templated,
//   inlined
//       handler::setKernelFunc(this, closurePtr)
//       %dki = detail::getDeviceKernelInfo<K>()            // SAME name marker
//       handler::setDeviceKernelInfoPtr(this, %dki)        // the dki sink
//   handler::setNDRangeDescriptor(this, nd_range_view)     // the geometry sink
//
// setDeviceKernelInfoPtr / setKernelFunc / setNDRangeDescriptor are
// __SYCL_EXPORT decl-only ABI boundaries, so they survive inlining as calls and
// their operands can be read directly. The name anchor (getDeviceKernelInfo) is
// identical to the direct path, so resolveLaunchedNameString is unchanged.

// Whether `call` is `handler::setDeviceKernelInfoPtr(handler,
// DeviceKernelInfo*)`
// -- the handler-path sink the marker's result flows into (arg0 is the handler,
// the join key for the launch's sibling calls).
static bool isHandlerSetDeviceKernelInfoPtr(cir::CallOp call) {
  std::optional<llvm::StringRef> callee = call.getCallee();
  return callee &&
         callee->starts_with("_ZN4sycl3_V17handler22setDeviceKernelInfoPtr");
}

// Whether `call` is `handler::setNDRangeDescriptor(handler, nd_range_view)` --
// the handler-path geometry sink (arg1 is the nd_range_view, same object
// traceNDRange consumes on the direct path).
static bool isHandlerSetNDRangeDescriptor(cir::CallOp call) {
  std::optional<llvm::StringRef> callee = call.getCallee();
  return callee && callee->contains("7handler20setNDRangeDescriptor");
}

// The operands of a recovered SYCL launch, from either launch shape. The direct
// path reads them from fixed positions of the runtime launch call; the handler
// path recovers them from sibling handler-method calls sharing the handler.
namespace {
struct RecoveredLaunch {
  cir::CallOp anchor;   // stored as LaunchSite::stubCall (launch-in @caller)
  mlir::Value geometry; // nd_range_view* operand
  mlir::Value
      closure; // HostKernelRef* (direct) / lambda-bearing value (handler)
};
} // namespace

// The nd_range_view a handler-path launch feeds setNDRangeDescriptor, found by
// the shared handler operand.
static mlir::Value findHandlerNDRangeView(mlir::Value handler) {
  for (mlir::Operation *user : handler.getUsers()) {
    auto call = mlir::dyn_cast<cir::CallOp>(user);
    if (!call || !isHandlerSetNDRangeDescriptor(call))
      continue;
    mlir::OperandRange args = call.getArgOperands();
    if (args.size() >= 2 && args[0] == handler)
      return args[1];
  }
  return {};
}

// Defined below; follows `v = load(slot)` back to the pointer stored into the
// slot, so a value reloaded before use resolves to the object it names.
static mlir::Value stripToPointerObject(mlir::Value v);

// Whether `call` constructs a `detail::HostKernel<...>` -- the wrapper the
// handler path builds around the user lambda. Its arg1 is the lambda object
// carrying the capture stores. (`10HostKernelI` is the templated wrapper; it
// excludes `14HostKernelBase`.)
static bool isHostKernelCtor(cir::CallOp call) {
  std::optional<llvm::StringRef> callee = call.getCallee();
  if (!callee)
    return false;
  return callee->contains("6detail10HostKernelI") &&
         (callee->contains("C1") || callee->contains("C2"));
}

// Whether `call` is `std::unique_ptr<HostKernelBase>::reset(HostKernelBase*)`
// -- the write that installs the wrapped kernel into handler.MHostKernel. Its
// arg1 is the freshly-built HostKernel object (as a HostKernelBase* base
// pointer).
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
// object's HostKernel constructor took the user lambda as arg1 (the temp
// carrying the capture stores). Keying off the handler's own member -- rather
// than scanning the enclosing function for the first HostKernel ctor -- is what
// disambiguates the launch when several q.submit(handler) sites were inlined
// into one caller: each has its own handler and its own MHostKernel. Returns
// the lambda operand; the closure trace strips it to the record object.
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
        if (!ctor || !isHostKernelCtor(ctor))
          continue;
        mlir::OperandRange ctorArgs = ctor.getArgOperands();
        if (ctorArgs.size() >= 2 && ctorArgs[0] == hostKernel)
          return ctorArgs[1];
      }
    }
  }
  return {};
}

// Follow the DeviceKernelInfo* result of a getDeviceKernelInfo call to the
// launch it drives, recognizing both launch shapes. CIRGen spills the pointer
// into a stack slot and reloads it before the consuming call, so hop
// store->slot->load once. Direct path: the pointer reaches
// submit_kernel_direct_with_event_impl, whose fixed operands carry the geometry
// and closure. Handler path: it reaches handler::setDeviceKernelInfoPtr, and
// the geometry/closure are recovered from the sibling handler calls.
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
      if (isSubmitKernelDirectImpl(call)) {
        mlir::OperandRange args = call.getArgOperands();
        if (args.size() <= SyclDeviceKernelInfoOperand)
          return std::nullopt;
        return RecoveredLaunch{call, args[SyclGeometryOperand],
                               args[SyclClosureOperand]};
      }
      if (isHandlerSetDeviceKernelInfoPtr(call)) {
        mlir::OperandRange args = call.getArgOperands();
        if (args.empty())
          return std::nullopt;
        mlir::Value handler = args[0];
        return RecoveredLaunch{call, findHandlerNDRangeView(handler),
                               findHandlerClosure(handler)};
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
  return mlir::SymbolTable::lookupNearestSymbolFrom<cir::GlobalOp>(
      info, strView.getSymbol());
}

// The _ZTS name-string GlobalOp a `getDeviceKernelInfo<Kernel>()` call targets.
// Its callee body references a `CompileTimeKernelInfo<Kernel>` global whose
// first field is the kernel-name string_view -- purely an IR/symbol walk, no
// AST and no callee-name substring matching.
static cir::GlobalOp resolveLaunchedNameString(cir::CallOp call) {
  mlir::FlatSymbolRefAttr callee = call.getCalleeAttr();
  if (!callee)
    return {};
  auto fn =
      mlir::SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(call, callee);
  if (!fn)
    return {};
  cir::GlobalOp nameString;
  fn.walk([&](cir::GetGlobalOp getGlobal) {
    auto global = mlir::SymbolTable::lookupNearestSymbolFrom<cir::GlobalOp>(
        getGlobal, getGlobal.getNameAttr());
    if (!global || !global.getName().contains("CompileTimeKernelInfo"))
      return;
    if (cir::GlobalOp name = getCompileTimeKernelName(global))
      nameString = name;
  });
  return nameString;
}

//===----------------------------------------------------------------------===//
// SYCL launch-geometry recovery (nd_range global/local work-group sizes)
//===----------------------------------------------------------------------===//
//
// The constant work-group sizes are not SSA-visible at the runtime launch
// call. CIRGen threads the nd_range object through a chain of launch shims
//   queue::parallel_for
//     -> detail::submit_kernel_direct_parallel_for
//        -> detail::submit_kernel_direct
//           -> submit_kernel_direct_with_event_impl   (the runtime launch)
// and the sizes live only in that object's memory, written by the range/array
// constructors up in the user function. mem2reg/SCCP cannot recover them: the
// nd_range object's address is taken (the nd_range_view constructor stores
// interior pointers to its array slots), so it never promotes to SSA.
//
// Recovery therefore (1) clones the host module, (2) flattens it so cir.return
// is a block terminator, (3) force-inlines the launch-shim chain and the
// geometry constructors into the user function, collapsing everything into one
// function, then (4) walks the nd_range_view object's memory back to the
// constant stores. The clone is discarded, so results are plain integers.

// Whether `fn` is a launch shim or a geometry constructor/accessor on the
// nd_range launch path -- a callee the recovery clone force-inlines. Matched by
// mangled-name shape so it is dimension-agnostic. Excludes the
// getDeviceKernelInfo marker (it names the kernel and must survive to identify
// the launch) and the __SYCL_EXPORT submit_kernel_direct_with_event_impl
// boundary (decl-only anyway). array::check_dimension, a throwing bounds check
// off the value path, is left as a call.
static bool isLaunchChainCallee(cir::FuncOp fn) {
  // The buffer and accessor API stays a call: its role is how buffer identity
  // is traced (a buffer constructor also matches the range-constructor shape
  // below). The launch-path, memory-command and destructor roles do not
  // change what is inlined here.
  if (auto role = fn->getAttrOfType<mlir::StringAttr>(
          CIRDialect::getSYCLHostAPIAttrName()))
    if ((role.getValue().starts_with("buffer.") ||
         role.getValue().starts_with("accessor.") ||
         role.getValue().starts_with("vector.")) &&
        !role.getValue().ends_with(".dtor"))
      return false;
  llvm::StringRef n = fn.getName();
  if (n.starts_with("_ZN4sycl3_V136submit_kernel_direct_with_event_impl"))
    return false;
  if (n.starts_with("_ZN4sycl3_V16detail19getDeviceKernelInfoI"))
    return false;
  // queue::parallel_for and the submit_kernel_direct* shims.
  if (n.contains("submit_kernel_direct") || n.contains("parallel_for"))
    return true;
  // The handler-submit launch chain (see the handler-submit shape above):
  // queue::submit builds the command-group functor object, and the handler
  // methods below thread the closure/geometry down to the surviving sinks. The
  // getDeviceKernelInfo marker (excluded first) and the __SYCL_EXPORT decl-only
  // sinks (setDeviceKernelInfoPtr / setNDRangeDescriptor / setKernelFunc, no
  // body) are left as calls. The HostKernel wrapper ctor is likewise left as a
  // call: StoreLambda installs it into handler.MHostKernel, so
  // findHandlerClosure recovers this launch's lambda by following that member
  // back to the ctor.
  if (n.contains("5queue6submit") || n.contains("wrap_kernel") ||
      n.contains("StoreLambda") || n.contains("setDeviceKernelInfoI") ||
      n.contains("convertToRangeViewAndSetDescriptor"))
    return true;
  // Geometry constructors, dimension-agnostic. The length digits in front of
  // each name keep "range" from matching inside "nd_range".
  bool isCtor = n.contains("C1") || n.contains("C2");
  if (n.contains("nd_range_view") && isCtor)
    return true;
  if (n.contains("8nd_rangeILi") && isCtor)
    return true;
  if (n.contains("5rangeILi") && isCtor)
    return true;
  // detail::array constructor and operator[] (but not check_dimension).
  if (n.contains("6detail5arrayILi") && (isCtor || n.contains("ixEi")))
    return true;
  return false;
}

// Bridge SYCL's type-erased command-group submit so the launch it performs can
// be inlined beside the user's allocations and geometry constants.
//
// `q.submit(cgf)` stores the command-group functor object and a static
// `invoker` trampoline into a `type_erased_cgfo_ty`, then hands it to the
// opaque runtime, which later calls `invoker(&cgf, handler)` ->
// `cgf.operator()(handler)`. No static call edge runs `operator()` from user
// code, so a generic inliner cannot reach it and the launch (with its
// getDeviceKernelInfo marker) never co-locates with the caller's USM
// allocations / nd_range constants. But the shape is a fixed SYCL contract: for
// each cgfo constructor on the clone, resolve the functor's `operator()` (via
// the invoker global the ctor stores) and synthesize the missing edge -- a
// direct `operator()(cgf, <fresh handler>)` call right after the ctor, tagged
// so inlineLaunchChain collapses it. The synthesized handler is an unused
// placeholder: it is only the receiver of the handler-method sinks, whose
// operands are read for recovery, never its value.
static void bridgeSubmitCGF(mlir::ModuleOp clone) {
  mlir::MLIRContext *ctx = clone.getContext();
  llvm::SmallVector<cir::CallOp> ctors;
  clone.walk([&](cir::CallOp call) {
    std::optional<llvm::StringRef> callee = call.getCallee();
    if (callee && callee->contains("19type_erased_cgfo_tyC2") &&
        call.getArgOperands().size() >= 2)
      ctors.push_back(call);
  });

  for (cir::CallOp ctor : ctors) {
    auto ctorFn = mlir::SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(
        ctor, ctor.getCalleeAttr());
    if (!ctorFn)
      continue;
    // The invoker trampoline: the (only) func-typed global the ctor stores.
    cir::FuncOp invoker;
    ctorFn.walk([&](cir::GetGlobalOp gg) {
      if (invoker)
        return;
      if (auto f = mlir::SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(
              gg, gg.getNameAttr()))
        invoker = f;
    });
    if (!invoker)
      continue;
    // operator(): the (only) bodied function the invoker calls.
    cir::FuncOp cgOperator;
    invoker.walk([&](cir::CallOp inner) {
      if (cgOperator || inner.isIndirect())
        return;
      auto f = mlir::SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(
          inner, inner.getCalleeAttr());
      if (f && f.getCallableRegion() && !f.getCallableRegion()->empty())
        cgOperator = f;
    });
    if (!cgOperator || cgOperator.getNumArguments() != 2)
      continue;

    auto handlerPtrTy =
        mlir::dyn_cast<cir::PointerType>(cgOperator.getArgumentTypes()[1]);
    if (!handlerPtrTy)
      continue;
    auto parentFn = ctor->getParentOfType<cir::FuncOp>();
    if (!parentFn || !parentFn.getCallableRegion())
      continue;

    // The functor object the ctor wraps. Bitcast if the ctor's reference type
    // differs from operator()'s receiver type.
    mlir::Value cgf = ctor.getArgOperands()[1];
    mlir::Type receiverTy = cgOperator.getArgumentTypes()[0];

    mlir::OpBuilder b(ctor);
    b.setInsertionPointToStart(&parentFn.getCallableRegion()->front());
    auto placeholder =
        cir::AllocaOp::create(b, ctor.getLoc(), handlerPtrTy,
                              "cg.handler.placeholder", b.getI64IntegerAttr(8));

    b.setInsertionPointAfter(ctor);
    if (cgf.getType() != receiverTy)
      cgf = cir::CastOp::create(b, ctor.getLoc(), receiverTy,
                                cir::CastKind::bitcast, cgf);
    auto invoke = cir::CallOp::create(
        b, ctor.getLoc(), mlir::FlatSymbolRefAttr::get(cgOperator),
        cir::VoidType::get(ctx), mlir::ValueRange{cgf, placeholder});
    invoke->setAttr("cir.sycl.cg_invoke", mlir::UnitAttr::get(ctx));
    // The command-group functor body is a launch shim for recovery purposes:
    // the inliner also collapses the launch chain into the ORIGINAL operator()
    // (its body is region-cloned, not consumed), where the closure captures are
    // only by-reference parameters with no visible allocation stores. Tag it so
    // the closure fold skips that copy and folds only the real,
    // inlined-into-caller site -- mirroring the isLaunchChainCallee exclusion
    // for the other shims.
    cgOperator->setAttr("cir.sycl.cg_body", mlir::UnitAttr::get(ctx));
  }
}

// Force-inline the launch-shim chain into the user function(s) on `clone`.
// `clone` must already be flattened. Iterates to a fixpoint because inlining a
// shim exposes the calls nested inside it. Uses region cloning (not splicing)
// so a shim called from several sites -- e.g. the range/array constructors that
// build both the global and local sizes -- stays intact for every site.
static void inlineLaunchChain(mlir::ModuleOp clone) {
  mlir::InlinerInterface interface(clone.getContext());
  mlir::InlinerConfig config;
  auto cloneCallback = config.getCloneCallback();

  bool changed = true;
  // The chain is ~4 shims deep plus the geometry ctors; 16 sweeps is plenty and
  // guards against an unexpected cycle.
  for (unsigned iter = 0; changed && iter < 16; ++iter) {
    changed = false;
    llvm::SmallVector<cir::CallOp> worklist;
    clone.walk([&](cir::CallOp call) {
      if (call.isIndirect())
        return;
      mlir::FlatSymbolRefAttr callee = call.getCalleeAttr();
      if (!callee)
        return;
      auto fn =
          mlir::SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(call, callee);
      // The synthetic command-group operator() call (see bridgeSubmitCGF) is
      // inlined by its tag rather than its name -- the lambda operator() has no
      // stable mangled shape to match.
      if (!fn ||
          (!isLaunchChainCallee(fn) && !call->hasAttr("cir.sycl.cg_invoke")))
        return;
      mlir::Region *region = fn.getCallableRegion();
      if (!region || region->empty())
        return;
      worklist.push_back(call);
    });
    for (cir::CallOp call : worklist) {
      auto callee = call.getCalleeAttr();
      auto fn =
          mlir::SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(call, callee);
      if (!fn)
        continue;
      auto callable = mlir::cast<mlir::CallableOpInterface>(fn.getOperation());
      auto callInterface =
          mlir::cast<mlir::CallOpInterface>(call.getOperation());
      if (mlir::succeeded(
              mlir::inlineCall(interface, cloneCallback, callInterface,
                               callable, callable.getCallableRegion(),
                               /*shouldCloneInlinedRegion=*/true))) {
        call.erase();
        changed = true;
      }
    }
  }
}

// The constant integer `v` holds, seeing through integral/bitcast reinterprets,
// or nullopt if it is not a compile-time constant integer.
static std::optional<int64_t> asConstInt(mlir::Value v) {
  for (unsigned i = 0; i < 16 && v; ++i) {
    if (auto c = v.getDefiningOp<cir::ConstantOp>()) {
      if (auto ia = mlir::dyn_cast<cir::IntAttr>(c.getValue()))
        return ia.getValue().getSExtValue();
      return std::nullopt;
    }
    auto cast = v.getDefiningOp<cir::CastOp>();
    if (cast && (cast.getKind() == cir::CastKind::integral ||
                 cast.getKind() == cir::CastKind::bitcast)) {
      v = cast.getSrc();
      continue;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

namespace {
/// A memory cell: `size` bytes at byte `offset` into the object `root`
/// addresses. `root` is an alloca or an opaque pointer value.
struct MemLoc {
  mlir::Value root;
  int64_t offset = 0;
};

/// Reads back, on the collapsed launch clone, the value the host stored into a
/// memory cell, through the by-value aggregate copies and by-reference captures
/// the launch shims build. Cells are byte ranges from the data layout rather
/// than member names, so a read survives a reinterpret of the object -- e.g.
/// the x86-64 ABI coercing a 16-byte closure into an anonymous `{ptr, ptr}`
/// on its way into queue::submit.
///
/// The reads are flow-insensitive, so a value is returned only when every
/// write that may reach the cell agrees on it: a cell with two different
/// writes, a partially overlapping write, a constructor or assignment run on
/// an overlapping object, or any call handed a pointer into a strict
/// subobject overlapping the cell reads as unknown. Writes are found through
/// every pointer derived from the root, including one spilled to a stack slot
/// and reloaded. Two kinds of access are assumed to only read: a call (other
/// than a constructor or assignment) handed the whole object -- on the launch
/// path those are setKernelFunc, the HostKernel constructor's source operand,
/// and operator() -- and a pointer stored into an aggregate rather than
/// spilled to a slot.
class MemoryReader {
public:
  explicit MemoryReader(mlir::ModuleOp module) : dl(module) {}

  /// The root object and constant byte offset `ptr` addresses. Member,
  /// constant-index element and base-class steps add their offsets;
  /// reinterprets add none; a pointer loaded from memory is replaced by the
  /// pointer stored there (a spill slot or a by-reference capture).
  MemLoc decompose(mlir::Value ptr, unsigned depth = 0) {
    MemLoc loc{ptr, 0};
    for (unsigned i = 0; i < 64 && loc.root; ++i) {
      mlir::Operation *def = loc.root.getDefiningOp();
      if (!def)
        break;
      if (auto gm = mlir::dyn_cast<cir::GetMemberOp>(def)) {
        auto rec = mlir::dyn_cast<cir::RecordType>(
            gm.getAddr().getType().getPointee());
        if (!rec || rec.isUnion())
          break;
        loc.offset += rec.getElementOffset(dl, gm.getIndex());
        loc.root = gm.getAddr();
        continue;
      }
      if (auto ge = mlir::dyn_cast<cir::GetElementOp>(def)) {
        std::optional<int64_t> idx = asConstInt(ge.getIndex());
        if (!idx)
          break;
        loc.offset += *idx * static_cast<int64_t>(dl.getTypeSize(
                                 ge.getResult().getType().getPointee()));
        loc.root = ge.getBase();
        continue;
      }
      if (auto bca = mlir::dyn_cast<cir::BaseClassAddrOp>(def)) {
        loc.offset += bca.getOffset().getZExtValue();
        loc.root = bca.getSrcAddr();
        continue;
      }
      if (auto cast = mlir::dyn_cast<cir::CastOp>(def)) {
        if (cast.getKind() == cir::CastKind::bitcast ||
            cast.getKind() == cir::CastKind::address_space ||
            cast.getKind() == cir::CastKind::array_to_ptrdecay) {
          loc.root = cast.getSrc();
          continue;
        }
        break;
      }
      if (auto load = mlir::dyn_cast<cir::LoadOp>(def)) {
        if (depth > 32)
          break;
        mlir::Value stored = loadedValue(load, depth + 1);
        if (!stored)
          break;
        MemLoc inner = decompose(stored, depth + 1);
        inner.offset += loc.offset;
        return inner;
      }
      break;
    }
    return loc;
  }

  /// The value a load reads, if a unique write of it reaches the loaded cell.
  mlir::Value loadedValue(cir::LoadOp load, unsigned depth = 0) {
    mlir::Type ty = load.getResult().getType();
    return valueAt(decompose(load.getAddr(), depth), dl.getTypeSize(ty), depth);
  }

  /// The value written into the `size`-byte cell at `loc`, or null when it is
  /// not determined by a unique write (see the class comment).
  mlir::Value valueAt(MemLoc loc, uint64_t size, unsigned depth = 0) {
    if (!loc.root || depth > 32)
      return {};
    // A loaded aggregate carries the bytes of the object it was loaded from.
    if (auto load = loc.root.getDefiningOp<cir::LoadOp>();
        load && !mlir::isa<cir::PointerType>(load.getResult().getType())) {
      MemLoc src = decompose(load.getAddr(), depth + 1);
      src.offset += loc.offset;
      return valueAt(src, size, depth + 1);
    }

    mlir::Value result;
    bool conflict = false;
    auto merge = [&](mlir::Value v) {
      if (!v || (result && result != v))
        conflict = true;
      else
        result = v;
    };

    // Every pointer into the root's storage, with its offset from the root.
    llvm::SmallVector<std::pair<mlir::Value, int64_t>, 16> aliases{
        {loc.root, 0}};
    llvm::SmallPtrSet<mlir::Value, 16> seenAlias{loc.root};
    for (unsigned ai = 0; ai < aliases.size() && !conflict; ++ai) {
      auto [ptr, base] = aliases[ai];
      for (mlir::Operation *user : ptr.getUsers()) {
        if (mlir::isa<cir::GetMemberOp, cir::GetElementOp, cir::BaseClassAddrOp,
                      cir::CastOp>(user)) {
          mlir::Value res = user->getResult(0);
          if (!mlir::isa<cir::PointerType>(res.getType()))
            continue;
          MemLoc sub = decompose(res, depth + 1);
          if (sub.root != loc.root) {
            // A step decompose cannot see through (a dynamic index): the
            // pointer may address any byte of the object.
            conflict = true;
            break;
          }
          if (seenAlias.insert(res).second)
            aliases.push_back({res, sub.offset});
          continue;
        }
        int64_t lo = base;
        if (auto st = mlir::dyn_cast<cir::StoreOp>(user)) {
          if (st.getAddr() != ptr) {
            // The pointer itself is spilled: its reloads alias the root too.
            forEachLoadOf(decompose(st.getAddr(), depth + 1),
                          [&](cir::LoadOp reload) {
                            if (seenAlias.insert(reload.getResult()).second)
                              aliases.push_back({reload.getResult(), base});
                          });
            continue;
          }
          mlir::Type ty = st.getValue().getType();
          int64_t hi = lo + dl.getTypeSize(ty);
          if (!overlaps(lo, hi, loc.offset, size))
            continue;
          if (lo == loc.offset && hi - lo == static_cast<int64_t>(size) &&
              !mlir::isa<cir::RecordType, cir::ArrayType>(ty)) {
            merge(st.getValue());
            continue;
          }
          if (!covers(lo, hi, loc.offset, size)) {
            conflict = true;
            break;
          }
          // An enclosing aggregate written whole: read within its source.
          auto srcLoad = st.getValue().getDefiningOp<cir::LoadOp>();
          if (!srcLoad) {
            conflict = true;
            break;
          }
          MemLoc src = decompose(srcLoad.getAddr(), depth + 1);
          src.offset += loc.offset - lo;
          merge(valueAt(src, size, depth + 1));
          continue;
        }
        if (auto cp = mlir::dyn_cast<cir::CopyOp>(user)) {
          if (cp.getDst() != ptr)
            continue;
          int64_t hi = lo + dl.getTypeSize(cp.getDst().getType().getPointee());
          if (!overlaps(lo, hi, loc.offset, size))
            continue;
          if (!covers(lo, hi, loc.offset, size)) {
            conflict = true;
            break;
          }
          MemLoc src = decompose(cp.getSrc(), depth + 1);
          src.offset += loc.offset - lo;
          merge(valueAt(src, size, depth + 1));
          continue;
        }
        if (auto call = mlir::dyn_cast<cir::CallOp>(user)) {
          auto ptrTy = mlir::dyn_cast<cir::PointerType>(ptr.getType());
          if (!ptrTy)
            continue;
          int64_t hi = lo + dl.getTypeSize(ptrTy.getPointee());
          if (!overlaps(lo, hi, loc.offset, size))
            continue;
          bool whole = base == 0 && hi >= rootSize(loc.root);
          if (!whole || writesReceiver(call, ptr)) {
            conflict = true;
            break;
          }
        }
      }
    }
    return conflict ? mlir::Value{} : result;
  }

  /// The constant integer in the `size`-byte cell at `loc`, following a
  /// stored value that is itself loaded from memory.
  std::optional<int64_t> constIntAt(MemLoc loc, uint64_t size) {
    mlir::Value v = valueAt(loc, size);
    for (unsigned i = 0; i < 16 && v; ++i) {
      if (std::optional<int64_t> c = asConstInt(v))
        return c;
      auto load = v.getDefiningOp<cir::LoadOp>();
      if (!load)
        break;
      v = loadedValue(load);
    }
    return std::nullopt;
  }

  /// The cell of the member named `name` of the record `obj` points at, found
  /// from a get_member on it.
  std::optional<MemLoc> memberLoc(mlir::Value obj, llvm::StringRef name) {
    for (mlir::Operation *u : obj.getUsers())
      if (auto gm = mlir::dyn_cast<cir::GetMemberOp>(u))
        if (gm.getName() == name)
          return decompose(gm.getResult());
    return std::nullopt;
  }

  uint64_t sizeOf(mlir::Type ty) const { return dl.getTypeSize(ty); }

  /// The single call that initializes the `size`-byte object at `loc`: a
  /// constructor run on it, or a call returning into it through `sret`. Null
  /// when there is no such call, more than one, or any other write overlaps
  /// the object (a store, a copy, an assignment, a call handed a pointer into
  /// it). Destructors and calls merely handed the object are not writes.
  cir::CallOp uniqueInitializer(MemLoc loc, uint64_t size) {
    if (!loc.root)
      return {};
    cir::CallOp init;
    llvm::SmallVector<std::pair<mlir::Value, int64_t>, 16> aliases{
        {loc.root, 0}};
    llvm::SmallPtrSet<mlir::Value, 16> seenAlias{loc.root};
    for (unsigned ai = 0; ai < aliases.size(); ++ai) {
      auto [ptr, base] = aliases[ai];
      for (mlir::Operation *user : ptr.getUsers()) {
        if (mlir::isa<cir::GetMemberOp, cir::GetElementOp, cir::BaseClassAddrOp,
                      cir::CastOp>(user)) {
          mlir::Value res = user->getResult(0);
          if (!mlir::isa<cir::PointerType>(res.getType()))
            continue;
          MemLoc sub = decompose(res);
          if (sub.root != loc.root)
            return {};
          if (seenAlias.insert(res).second)
            aliases.push_back({res, sub.offset});
          continue;
        }
        if (auto st = mlir::dyn_cast<cir::StoreOp>(user)) {
          if (st.getAddr() != ptr) {
            forEachLoadOf(decompose(st.getAddr()), [&](cir::LoadOp reload) {
              if (seenAlias.insert(reload.getResult()).second)
                aliases.push_back({reload.getResult(), base});
            });
            continue;
          }
          if (overlaps(base, base + dl.getTypeSize(st.getValue().getType()),
                       loc.offset, size))
            return {};
          continue;
        }
        if (auto cp = mlir::dyn_cast<cir::CopyOp>(user)) {
          if (cp.getDst() == ptr &&
              overlaps(base,
                       base +
                           dl.getTypeSize(cp.getDst().getType().getPointee()),
                       loc.offset, size))
            return {};
          continue;
        }
        auto call = mlir::dyn_cast<cir::CallOp>(user);
        if (!call)
          continue;
        auto ptrTy = mlir::dyn_cast<cir::PointerType>(ptr.getType());
        if (!ptrTy)
          continue;
        int64_t hi = base + dl.getTypeSize(ptrTy.getPointee());
        if (!overlaps(base, hi, loc.offset, size))
          continue;
        cir::FuncOp callee = calleeOf(call);
        if (callee && callee.isCxxDestructor())
          continue;
        bool isReceiver =
            !call.getArgOperands().empty() && call.getArgOperands()[0] == ptr;
        bool initializes =
            isReceiver && callee &&
            (callee.isCxxConstructor() ||
             callee.getArgAttr(
                 0, mlir::LLVM::LLVMDialect::getStructRetAttrName()));
        if (initializes && base == loc.offset &&
            hi - base == static_cast<int64_t>(size)) {
          if (init && init != call)
            return {};
          init = call;
          continue;
        }
        // A call handed a pointer covering the whole object (or an enclosing
        // one) is assumed to only read it, unless it constructs or assigns
        // the pointee; one handed a pointer into the object may write part
        // of it.
        if (!covers(base, hi, loc.offset, size) || writesReceiver(call, ptr))
          return {};
      }
    }
    return init;
  }

  static cir::FuncOp calleeOf(cir::CallOp call) {
    if (call.isIndirect())
      return {};
    return mlir::SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(
        call, call.getCalleeAttr());
  }

private:
  /// Calls `fn` on every load of exactly the cell `slot`.
  template <typename Fn> void forEachLoadOf(MemLoc slot, Fn fn) {
    if (!slot.root)
      return;
    // Loads can go through any view of the slot; walk those.
    llvm::SmallVector<mlir::Value, 8> ptrs{slot.root};
    for (unsigned i = 0; i < ptrs.size() && i < 64; ++i)
      for (mlir::Operation *u : ptrs[i].getUsers()) {
        if (auto load = mlir::dyn_cast<cir::LoadOp>(u)) {
          MemLoc at = decompose(load.getAddr());
          if (at.root == slot.root && at.offset == slot.offset)
            fn(load);
          continue;
        }
        if (mlir::isa<cir::GetMemberOp, cir::GetElementOp, cir::CastOp>(u))
          ptrs.push_back(u->getResult(0));
      }
  }

  /// Whether `call` constructs or assigns the object `ptr` addresses.
  static bool writesReceiver(cir::CallOp call, mlir::Value ptr) {
    if (call.getArgOperands().empty() || call.getArgOperands()[0] != ptr)
      return false;
    if (call.isIndirect())
      return true;
    auto callee = mlir::SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(
        call, call.getCalleeAttr());
    if (!callee)
      return true;
    return callee.isCxxConstructor() || callee.isCxxSpecialAssignment();
  }

  int64_t rootSize(mlir::Value root) const {
    auto ptrTy = mlir::dyn_cast<cir::PointerType>(root.getType());
    return ptrTy ? static_cast<int64_t>(dl.getTypeSize(ptrTy.getPointee())) : 0;
  }

  static bool overlaps(int64_t lo, int64_t hi, int64_t off, uint64_t size) {
    return lo < off + static_cast<int64_t>(size) && off < hi;
  }
  static bool covers(int64_t lo, int64_t hi, int64_t off, uint64_t size) {
    return lo <= off && off + static_cast<int64_t>(size) <= hi;
  }

  mlir::DataLayout dl;
};
} // namespace

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

// Follow a by-value aggregate copy back to the original object: an alloca whose
// whole-object content is written by `store (load %orig), %alloca` or
// `cir.copy %orig to %alloca`. handler::setNDRangeDescriptor takes the
// nd_range_view BY VALUE, so the launch operand is a copy while the field
// stores (MGlobalSize/MLocalSize/MDims) live on the original; strip to it.
static mlir::Value stripAggregateCopy(mlir::Value v) {
  for (unsigned i = 0; i < 8 && v; ++i) {
    mlir::Value src;
    for (mlir::Operation *u : v.getUsers()) {
      if (auto st = mlir::dyn_cast<cir::StoreOp>(u)) {
        if (st.getAddr() != v)
          continue;
        // A whole-object store of a loaded aggregate is a by-value copy.
        if (auto load = st.getValue().getDefiningOp<cir::LoadOp>())
          src = load.getAddr();
      } else if (auto cp = mlir::dyn_cast<cir::CopyOp>(u)) {
        if (cp.getDst() == v)
          src = cp.getSrc();
      }
    }
    if (!src)
      break;
    v = src;
  }
  return v;
}

// Recover the constant nd_range geometry from the nd_range_view pointer operand
// of a runtime launch call (on the inlined clone). The view constructor stored
// interior pointers into its MGlobalSize/MLocalSize fields that alias the
// nd_range object's array slots; trace those back to the constant stores.
static SyclGeometry traceNDRange(MemoryReader &reader,
                                 mlir::Value viewOperand) {
  SyclGeometry g;
  // The member names live on the original view object, whose fields the view
  // constructor wrote; the operand may be a by-value copy of it.
  mlir::Value view = stripAggregateCopy(stripToPointerObject(viewOperand));
  if (!view)
    return g;
  MemLoc viewLoc = reader.decompose(view);

  auto fieldValue = [&](llvm::StringRef field) -> mlir::Value {
    for (mlir::Operation *u : view.getUsers()) {
      auto gm = mlir::dyn_cast<cir::GetMemberOp>(u);
      if (!gm || gm.getName() != field)
        continue;
      MemLoc loc = reader.decompose(gm.getResult());
      if (loc.root != viewLoc.root)
        return {};
      return reader.valueAt(
          loc, reader.sizeOf(gm.getResult().getType().getPointee()));
    }
    return {};
  };

  unsigned dims = 0;
  if (mlir::Value d = fieldValue("MDims"))
    if (auto dv = asConstInt(d))
      dims = static_cast<unsigned>(*dv);
  if (dims == 0 || dims > 3)
    dims = 1;
  g.dims = dims;

  // Read `dims` consecutive elements of the size array a field points at. The
  // field holds &common_array[0].
  auto readSizes =
      [&](llvm::StringRef field,
          llvm::SmallVectorImpl<std::optional<int64_t>> &out) -> bool {
    mlir::Value interior = fieldValue(field);
    if (!interior)
      return false;
    // A plain-range launch stores a null pointer for MLocalSize (no explicit
    // local work-group size): that field is absent, not unrecovered.
    if (auto c = interior.getDefiningOp<cir::ConstantOp>())
      if (auto p = mlir::dyn_cast<cir::ConstPtrAttr>(c.getValue()))
        if (p.isNullValue())
          return false;
    auto ptrTy = mlir::dyn_cast<cir::PointerType>(interior.getType());
    if (!ptrTy)
      return false;
    uint64_t elt = reader.sizeOf(ptrTy.getPointee());
    MemLoc base = reader.decompose(interior);
    for (unsigned i = 0; i < dims; ++i) {
      MemLoc p = base;
      p.offset += i * elt;
      out.push_back(reader.constIntAt(p, elt));
    }
    return true;
  };

  bool haveGlobal = readSizes("MGlobalSize", g.global);
  bool haveLocal = readSizes("MLocalSize", g.local);
  // A plain-range launch (RoundedRangeKernel) carries no explicit local size;
  // do not fabricate one.
  if (!haveLocal)
    g.local.clear();
  g.valid = haveGlobal;
  return g;
}

//===----------------------------------------------------------------------===//
// SYCL closure-pointer allocation distinctness (llvm.noalias inference)
//===----------------------------------------------------------------------===//
//
// A device kernel pointer parameter is bound to a closure capture field (see
// the spir_kernel prologue: arg -> slot -> address_space cast -> store into a
// cir.get_member of the closure). On the host, the same field is written the
// captured pointer value. If two captured pointers come from two DISTINCT USM
// allocations then their device parameters cannot alias, so each may carry
// `llvm.noalias`. This runs on the same collapsed clone as the geometry
// recovery: the field stores live in the user function, ahead of the launch
// chain, so a single inline exposes them.

// Whether `call` is a SYCL USM allocation (malloc_device / malloc_shared /
// malloc_host). Two distinct such calls return provably non-overlapping
// storage.
static bool isUSMAllocCall(cir::CallOp call) {
  std::optional<llvm::StringRef> callee = call.getCallee();
  if (!callee)
    return false;
  return callee->contains("malloc_device") ||
         callee->contains("malloc_shared") || callee->contains("malloc_host");
}

// Follow a captured-pointer field's stored value back to the USM allocation
// that produced it, seeing through offset-0 reinterprets and through values
// reloaded from memory -- a spill slot, or a `[&]`-captured pointer read
// through the closure copies (which is what lets noalias fire on the common
// q.submit(handler) path, where the USM allocations live in the caller and
// reach the closure only by reference). Only a value that resolves to a USM
// allocation call is an allocation identity; pointer arithmetic (a + offset),
// an opaque parameter, a global or a buffer stops the walk and yields null --
// conservatively "not a distinct allocation".
static cir::CallOp resolveUSMAllocation(MemoryReader &reader, mlir::Value v) {
  for (unsigned i = 0; i < 24 && v; ++i) {
    mlir::Operation *def = v.getDefiningOp();
    if (!def)
      return {};
    if (auto call = mlir::dyn_cast<cir::CallOp>(def))
      return isUSMAllocCall(call) ? call : cir::CallOp{};
    if (auto cast = mlir::dyn_cast<cir::CastOp>(def)) {
      if (cast.getKind() == cir::CastKind::bitcast ||
          cast.getKind() == cir::CastKind::address_space) {
        v = cast.getSrc();
        continue;
      }
      return {};
    }
    if (auto load = mlir::dyn_cast<cir::LoadOp>(def)) {
      v = reader.loadedValue(load);
      continue;
    }
    return {};
  }
  return {};
}

// Peel base_class_addr / bitcast / address_space reinterprets and spill slots
// off `v` to reach the record object it addresses.
static mlir::Value stripToRecordObject(mlir::Value v) {
  for (unsigned i = 0; i < 16 && v; ++i) {
    v = stripToPointerObject(v);
    mlir::Operation *def = v.getDefiningOp();
    if (!def)
      break;
    if (auto bca = mlir::dyn_cast<cir::BaseClassAddrOp>(def)) {
      v = bca.getSrcAddr();
      continue;
    }
    if (auto cast = mlir::dyn_cast<cir::CastOp>(def)) {
      if (cast.getKind() == cir::CastKind::bitcast ||
          cast.getKind() == cir::CastKind::address_space) {
        v = cast.getSrc();
        continue;
      }
    }
    break;
  }
  return v;
}

// The lambda closure object a runtime launch captures. Direct path: the operand
// is a HostKernelRefBase*; strip to the HostKernelRef record and take the
// lambda pointer its constructor stored (constructor arg 1). Handler path: the
// operand is already the lambda-bearing value (the HostKernel ctor's arg1, see
// findHandlerClosure), which strips straight to the lambda record and has no
// HostKernelRef wrapper -- so when no such wrapper ctor is found, the stripped
// record is itself the closure. Null only if the operand does not reach a
// record object (e.g. a free-function kernel).
static mlir::Value traceClosureObject(mlir::Value closureOperand) {
  mlir::Value hostKernelRef = stripToRecordObject(closureOperand);
  if (!hostKernelRef)
    return {};
  for (mlir::Operation *user : hostKernelRef.getUsers()) {
    auto call = mlir::dyn_cast<cir::CallOp>(user);
    if (!call)
      continue;
    std::optional<llvm::StringRef> callee = call.getCallee();
    mlir::OperandRange args = call.getArgOperands();
    if (!callee || !callee->contains("HostKernelRef") || args.size() < 2 ||
        args[0] != hostKernelRef)
      continue;
    // A HostKernelRef constructor stores the captured lambda (arg 1).
    if (!(callee->contains("C1") || callee->contains("C2")))
      continue;
    return stripToRecordObject(args[1]);
  }
  return hostKernelRef;
}

// The capture fields of a closure object accepted by `wanted` (on the field's
// type) and the value written into each, read by layout so a field written
// only through a by-value copy of the closure is found too. A field whose
// value is not determined by a unique write is reported with a null value.
template <typename Pred>
static void collectClosureFields(
    MemoryReader &reader, mlir::Value closure, Pred wanted,
    llvm::SmallVectorImpl<std::pair<llvm::StringRef, mlir::Value>> &out) {
  if (!closure)
    return;
  MemLoc closureLoc = reader.decompose(closure);
  llvm::StringSet<> seen;
  for (mlir::Operation *user : closure.getUsers()) {
    auto gm = mlir::dyn_cast<cir::GetMemberOp>(user);
    if (!gm || !wanted(gm.getResult().getType().getPointee()))
      continue;
    if (!seen.insert(gm.getName()).second)
      continue;
    MemLoc loc = reader.decompose(gm.getResult());
    mlir::Value stored;
    if (loc.root == closureLoc.root)
      stored = reader.valueAt(
          loc, reader.sizeOf(gm.getResult().getType().getPointee()));
    out.emplace_back(gm.getName(), stored);
  }
}

//===----------------------------------------------------------------------===//
// SYCL scalar-capture constant propagation
//===----------------------------------------------------------------------===//
//
// A closure SCALAR capture field (int/float/bool, not a pointer/accessor) is
// written the captured value on the host. When that value is the SAME
// compile-time constant at every visible launch site, the device kernel
// parameter bound to the field can be replaced by the constant. This mirrors
// the noalias fold above, but over scalar fields, and records the agreed
// constant attribute rather than a distinctness bit. Runs on the same collapsed
// clone as the geometry / noalias recovery.

// The scalar constant `v` holds, seeing through integral/bitcast reinterprets
// CIRGen inserts around a captured scalar, or null if it is not a compile-time
// scalar constant. Only int/float/bool constants qualify (a pointer/aggregate
// constant is not a scalar capture we propagate).
static mlir::TypedAttr asConstScalar(mlir::Value v) {
  for (unsigned i = 0; i < 16 && v; ++i) {
    if (auto c = v.getDefiningOp<cir::ConstantOp>()) {
      mlir::TypedAttr attr = c.getValue();
      if (mlir::isa<cir::IntAttr, cir::FPAttr, cir::BoolAttr>(attr))
        return attr;
      return {};
    }
    auto cast = v.getDefiningOp<cir::CastOp>();
    if (cast && (cast.getKind() == cir::CastKind::integral ||
                 cast.getKind() == cir::CastKind::bitcast)) {
      v = cast.getSrc();
      continue;
    }
    return {};
  }
  return {};
}

// The scalar capture fields (int/float/bool, not a pointer/accessor) of a
// closure object and the value written into each.
static void collectClosureScalarFields(
    MemoryReader &reader, mlir::Value closure,
    llvm::SmallVectorImpl<std::pair<llvm::StringRef, mlir::Value>> &out) {
  collectClosureFields(
      reader, closure,
      [](mlir::Type t) {
        return mlir::isa<cir::IntType, cir::FPTypeInterface, cir::BoolType>(t);
      },
      out);
}

// Fold one launch site's closure into the running constant-argument agreement
// for a kernel. A field maps here to the constant it is stored; a field stored
// a non-constant is simply absent. Agreement is ANDed across sites: a field
// survives only if present with the SAME constant at this site and every
// earlier one, so a disagreeing or non-constant store at any site drops it.
static void foldClosureConstArgs(MemoryReader &reader,
                                 mlir::Value closureOperand, bool &seenAnySite,
                                 llvm::StringMap<mlir::TypedAttr> &consts) {
  mlir::Value closure = traceClosureObject(closureOperand);
  llvm::SmallVector<std::pair<llvm::StringRef, mlir::Value>, 4> fields;
  collectClosureScalarFields(reader, closure, fields);

  llvm::StringMap<mlir::TypedAttr> here;
  for (auto &f : fields) {
    mlir::TypedAttr c = asConstScalar(f.second);
    // A field appearing twice, or a non-constant store, is not a clean
    // single-constant source: drop it for this site.
    if (!c || here.count(f.first))
      here[f.first] = nullptr;
    else
      here[f.first] = c;
  }

  if (!seenAnySite) {
    consts = std::move(here);
    seenAnySite = true;
    return;
  }
  // AND across sites: keep a field only when it is present with the same
  // constant here and previously.
  llvm::StringMap<mlir::TypedAttr> merged;
  for (auto &e : consts) {
    mlir::TypedAttr hereVal = here.lookup(e.first());
    if (e.second && hereVal && e.second == hereVal)
      merged[e.first()] = e.second;
  }
  consts = std::move(merged);
}

//===----------------------------------------------------------------------===//
// SYCL accessor buffer identity
//===----------------------------------------------------------------------===//
//
// A device kernel accessor pointer parameter is bound to a closure field
// holding a sycl::accessor. On the host the field is copy-constructed from the
// command group's local accessor, which was built from a buffer (accessor(buf,
// h, mode) or buf.get_access(h, ...)), reached through the command group's
// by-reference capture. Two accessor parameters cannot alias when their
// buffers are provably backed by different memory. CIRGen tags the buffer and
// accessor API by declaration (cir.sycl_host_api), so the trace matches roles,
// not mangled names.
//
// The memory identity of a buffer is:
//  - buffer(range): fresh storage, identified by the buffer object itself;
//  - buffer(hostData, range): possibly the host memory itself (use_host_ptr,
//    or zero-copy on an integrated device), so identified by the allocation
//    hostData points into -- a std::vector, a new/malloc result, a local
//    array, or a USM allocation (shared with the USM pointer domain, so it can
//    alias a USM kernel argument). The spec makes two live buffers over the
//    same host memory undefined behavior ("Constructing more than one SYCL
//    buffer managing the same host pointer is still undefined behavior";
//    host memory must not be "used to construct another buffer ... during the
//    lifetime of this buffer"), but lifetimes are not tracked here, so this
//    does not rely on it: buffers over the same allocation share an identity;
//  - a copy, sub-buffer or reinterpret of another buffer: that buffer's
//    identity (sub-buffer ranges are not compared);
//  - anything else -- a buffer reached through a function parameter, a global,
//    the heap, an interop constructor, or an assignment -- is unknown.

namespace {
/// The memory a captured pointer or accessor is resolved to: an allocation
/// identity (the defining op and an offset into it) and its description for
/// remarks, or, when `op` is null, why it could not be resolved.
struct MemIdentity {
  mlir::Operation *op = nullptr;
  int64_t offset = 0;
  std::string what;

  static MemIdentity unknown(llvm::Twine why) {
    MemIdentity id;
    id.what = why.str();
    return id;
  }
  bool known() const { return op != nullptr; }
  bool operator==(const MemIdentity &o) const {
    return op == o.op && offset == o.offset;
  }
};
} // namespace

/// The cir.sycl_host_api role of `call`'s callee, and the operand holding the
/// role's source-level argument `cir.sycl_host_api_arg` (past `sret` and
/// `this`).
static llvm::StringRef hostAPIRole(cir::CallOp call, mlir::Value *roleArg) {
  cir::FuncOp callee = MemoryReader::calleeOf(call);
  if (!callee)
    return {};
  auto role = callee->getAttrOfType<mlir::StringAttr>(
      CIRDialect::getSYCLHostAPIAttrName());
  if (!role)
    return {};
  if (roleArg) {
    unsigned idx = 0;
    if (auto a = callee->getAttrOfType<mlir::IntegerAttr>(
            CIRDialect::getSYCLHostAPIArgAttrName()))
      idx = a.getInt();
    unsigned first =
        callee.getArgAttr(0, mlir::LLVM::LLVMDialect::getStructRetAttrName())
            ? 2  // sret, this
            : 1; // this
    mlir::OperandRange args = call.getArgOperands();
    *roleArg = first + idx < args.size() ? args[first + idx] : mlir::Value{};
  }
  return role.getValue();
}

/// The `this` operand of a method call, past an `sret` result slot.
static mlir::Value receiverOf(cir::CallOp call) {
  cir::FuncOp callee = MemoryReader::calleeOf(call);
  unsigned idx =
      callee && callee.getArgAttr(
                    0, mlir::LLVM::LLVMDialect::getStructRetAttrName())
          ? 1
          : 0;
  mlir::OperandRange args = call.getArgOperands();
  return idx < args.size() ? args[idx] : mlir::Value{};
}

/// A short name for the object at `loc`, from its alloca, for remarks.
static std::string objectName(MemLoc loc) {
  if (auto alloca =
          loc.root ? loc.root.getDefiningOp<cir::AllocaOp>() : cir::AllocaOp{})
    return ("'" + alloca.getName() + "'").str();
  return "<unnamed>";
}

/// Why the object `loc` addresses is not a visible local: a function
/// parameter, a global, or memory the trace could not follow.
static std::string notLocalReason(llvm::StringRef what, MemLoc loc) {
  if (!loc.root)
    return (what + " is not visible").str();
  if (mlir::isa<mlir::BlockArgument>(loc.root))
    return (what + " is a function parameter").str();
  if (loc.root.getDefiningOp<cir::GetGlobalOp>())
    return (what + " is a global").str();
  if (loc.root.getDefiningOp<cir::CallOp>())
    return (what + " is on the heap").str();
  return (what + " is reached through memory the analysis cannot follow").str();
}

/// The allocation a buffer's host pointer points into.
static MemIdentity resolveHostAllocation(MemoryReader &reader, mlir::Value ptr,
                                         llvm::StringRef bufName) {
  for (unsigned i = 0; i < 24 && ptr; ++i) {
    mlir::Operation *def = ptr.getDefiningOp();
    if (!def)
      break;
    if (auto cast = mlir::dyn_cast<cir::CastOp>(def)) {
      if (cast.getKind() == cir::CastKind::bitcast ||
          cast.getKind() == cir::CastKind::address_space) {
        ptr = cast.getSrc();
        continue;
      }
      if (cast.getKind() == cir::CastKind::array_to_ptrdecay) {
        MemLoc arr = reader.decompose(cast.getSrc());
        if (auto alloca = arr.root ? arr.root.getDefiningOp<cir::AllocaOp>()
                                   : cir::AllocaOp{}) {
          MemIdentity id;
          id.op = alloca;
          id.what = "local array " + objectName(arr);
          return id;
        }
        break;
      }
      break;
    }
    if (auto load = mlir::dyn_cast<cir::LoadOp>(def)) {
      ptr = reader.loadedValue(load);
      continue;
    }
    if (auto call = mlir::dyn_cast<cir::CallOp>(def)) {
      if (isUSMAllocCall(call)) {
        MemIdentity id;
        id.op = call;
        id.what = "a USM allocation";
        return id;
      }
      std::optional<llvm::StringRef> callee = call.getCallee();
      if (callee &&
          (*callee == "_Znwm" || *callee == "_Znam" || *callee == "malloc" ||
           *callee == "calloc" || *callee == "aligned_alloc")) {
        MemIdentity id;
        id.op = call;
        id.what = "a heap allocation";
        return id;
      }
      if (hostAPIRole(call, nullptr) == "vector.data") {
        // Distinct std::vector objects own distinct storage.
        MemLoc vec = reader.decompose(receiverOf(call));
        auto alloca = vec.root ? vec.root.getDefiningOp<cir::AllocaOp>()
                               : cir::AllocaOp{};
        if (!alloca)
          return MemIdentity::unknown(notLocalReason(
              ("the std::vector behind buffer " + bufName).str(), vec));
        MemIdentity id;
        id.op = alloca;
        id.offset = vec.offset;
        id.what = "std::vector " + objectName(vec);
        return id;
      }
    }
    break;
  }
  return MemIdentity::unknown("the host pointer of buffer " + bufName +
                              " has unknown provenance");
}

/// The memory identity of the sycl::buffer object `bufPtr` addresses.
static MemIdentity resolveBufferIdentity(MemoryReader &reader,
                                         mlir::Value bufPtr, unsigned depth) {
  MemLoc loc = reader.decompose(bufPtr);
  std::string name = objectName(loc);
  if (depth > 8)
    return MemIdentity::unknown("buffer " + name + " is derived too deeply");
  auto alloca =
      loc.root ? loc.root.getDefiningOp<cir::AllocaOp>() : cir::AllocaOp{};
  auto ptrTy = mlir::dyn_cast<cir::PointerType>(bufPtr.getType());
  if (!alloca || !ptrTy)
    return MemIdentity::unknown(notLocalReason("the buffer", loc));
  cir::CallOp init =
      reader.uniqueInitializer(loc, reader.sizeOf(ptrTy.getPointee()));
  if (!init)
    return MemIdentity::unknown("buffer " + name +
                                " is not constructed exactly once");
  mlir::Value arg;
  llvm::StringRef role = hostAPIRole(init, &arg);
  if (role == "buffer.alloc") {
    MemIdentity id;
    id.op = alloca;
    id.offset = loc.offset;
    id.what = "buffer " + name;
    return id;
  }
  if (role == "buffer.host_ptr") {
    MemIdentity id = resolveHostAllocation(reader, arg, name);
    if (id.known())
      id.what = "buffer " + name + " over " + id.what;
    return id;
  }
  if (role == "buffer.copy" || role == "buffer.sub")
    return arg ? resolveBufferIdentity(reader, arg, depth + 1)
               : MemIdentity::unknown("buffer " + name + " has no source");
  if (role == "buffer.reinterpret")
    return resolveBufferIdentity(reader, receiverOf(init), depth + 1);
  return MemIdentity::unknown("buffer " + name +
                              " is built by a constructor the analysis does "
                              "not model");
}

/// The memory identity of the buffer behind the sycl::accessor object at
/// `loc`, through accessor copies back to its construction from a buffer.
static MemIdentity resolveAccessorIdentity(MemoryReader &reader, MemLoc loc,
                                           uint64_t size, unsigned depth) {
  std::string name = objectName(loc);
  if (depth > 8)
    return MemIdentity::unknown("accessor " + name + " is copied too deeply");
  cir::CallOp init = reader.uniqueInitializer(loc, size);
  if (!init)
    return MemIdentity::unknown("accessor " + name +
                                " is not constructed exactly once");
  mlir::Value arg;
  llvm::StringRef role = hostAPIRole(init, &arg);
  if (role == "accessor.copy" && arg)
    return resolveAccessorIdentity(reader, reader.decompose(arg), size,
                                   depth + 1);
  if (role == "accessor.from_buffer" && arg)
    return resolveBufferIdentity(reader, arg, 0);
  if (role == "buffer.get_access")
    return resolveBufferIdentity(reader, receiverOf(init), 0);
  return MemIdentity::unknown("accessor " + name +
                              " is built by a constructor the analysis does "
                              "not model");
}

namespace {
/// Which closure fields of a SYCL kernel are candidates for llvm.noalias, read
/// off the device kernel's parameters (cir.sycl_kernel_param): captured USM
/// pointers and global-memory accessors held directly by the closure. Any
/// other parameter that can carry a pointer to global memory -- a pointer or
/// accessor nested in a captured struct, a struct holding a pointer, a stream
/// or annotated pointer, a parameter with no recorded field -- may alias the
/// candidates without being proven distinct, so it disables noalias for the
/// whole kernel.
struct NoAliasCandidates {
  llvm::StringMap<bool> fields; // field -> holds an accessor (else a pointer)
  std::string poison;           // why noalias is off for the kernel, if it is
};
} // namespace

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

static void classifyKernelParams(cir::FuncOp kernel, NoAliasCandidates &out) {
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

namespace {
/// The running noalias verdict for one candidate closure field of a kernel,
/// with the memory it resolved to or why it was rejected.
struct NoAliasDecision {
  bool ok = false;
  std::string note;
};
} // namespace

/// The memory a candidate closure field refers to: the USM allocation of a
/// captured pointer, or the buffer behind a captured accessor.
static MemIdentity resolveClosureField(MemoryReader &reader,
                                       mlir::Value closure,
                                       llvm::StringRef field, bool isAccessor) {
  MemLoc closureLoc = reader.decompose(closure);
  for (mlir::Operation *user : closure.getUsers()) {
    auto gm = mlir::dyn_cast<cir::GetMemberOp>(user);
    if (!gm || gm.getName() != field)
      continue;
    MemLoc loc = reader.decompose(gm.getResult());
    if (loc.root != closureLoc.root)
      break;
    uint64_t size = reader.sizeOf(gm.getResult().getType().getPointee());
    if (isAccessor)
      return resolveAccessorIdentity(reader, loc, size, 0);
    mlir::Value stored = reader.valueAt(loc, size);
    if (cir::CallOp alloc = resolveUSMAllocation(reader, stored)) {
      MemIdentity id;
      id.op = alloc;
      id.what = "a USM allocation";
      return id;
    }
    return MemIdentity::unknown("pointer '" + field +
                                "' is not a visible USM allocation");
  }
  return MemIdentity::unknown("'" + field +
                              "' is not found in the launch's closure");
}

// Fold one launch site's closure into the running noalias verdicts for a
// kernel. A candidate field (see NoAliasCandidates) is eligible at this site
// iff its memory is known and distinct from that of every other candidate of
// the same closure; an unresolved field poisons itself and, being possibly
// aliasing, every other field. The verdict is ANDed across sites, so a field
// kept here must have held at every earlier site; the first rejection's
// reason is kept.
static void foldClosureNoAlias(MemoryReader &reader, mlir::Value closureOperand,
                               const NoAliasCandidates &candidates,
                               bool &seenAnySite,
                               llvm::StringMap<NoAliasDecision> &decisions) {
  mlir::Value closure = traceClosureObject(closureOperand);
  llvm::SmallVector<llvm::StringRef, 4> names;
  for (const auto &c : candidates.fields)
    names.push_back(c.first());
  llvm::sort(names);

  llvm::SmallVector<MemIdentity, 4> ids;
  for (llvm::StringRef name : names) {
    ids.push_back(
        closure ? resolveClosureField(reader, closure, name,
                                      candidates.fields.lookup(name))
                : MemIdentity::unknown("the launch's closure is not visible"));
    LLVM_DEBUG(llvm::dbgs() << "noalias: field '" << name << "' -> "
                            << (ids.back().known() ? "" : "unknown: ")
                            << ids.back().what << "\n");
  }

  llvm::StringMap<NoAliasDecision> here;
  for (auto [i, name] : llvm::enumerate(names)) {
    NoAliasDecision &d = here[name];
    if (!candidates.poison.empty()) {
      d.note = candidates.poison;
      continue;
    }
    if (!ids[i].known()) {
      d.note = ids[i].what;
      continue;
    }
    d.ok = true;
    d.note = ids[i].what;
    for (auto [j, other] : llvm::enumerate(names)) {
      if (i == j)
        continue;
      if (!ids[j].known()) {
        d = {false, ("'" + other + "' may alias it: " + ids[j].what).str()};
        break;
      }
      if (ids[j] == ids[i]) {
        d = {false, ("'" + name + "' and '" + other +
                     "' use the same memory, " + ids[i].what)
                        .str()};
        break;
      }
    }
  }

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

// Recover per-kernel constant nd_range geometry for the whole container, and
// (out) the closure capture fields whose USM allocations are pairwise distinct
// at every launch site. Clones the host module, collapses the launch chain,
// then for each surviving getDeviceKernelInfo marker resolves the kernel it
// names and traces the launch it feeds. Keyed by the kernel's _ZTS mangled name
// so the results map straight onto the SYCL bindings, which are keyed the same.
static llvm::StringMap<SyclGeometry> computeSYCLGeometry(
    mlir::ModuleOp container,
    const llvm::StringMap<NoAliasCandidates> &noAliasCandidates,
    llvm::StringMap<llvm::StringSet<>> &noAliasFields,
    llvm::StringMap<llvm::StringMap<std::string>> &noAliasNotes,
    llvm::StringMap<llvm::StringMap<mlir::TypedAttr>> &constArgs) {
  llvm::StringMap<SyclGeometry> result;
  // Per-kernel running noalias verdicts, ANDed across the kernel's sites.
  llvm::StringMap<bool> seenSite;
  llvm::StringMap<llvm::StringMap<NoAliasDecision>> eligible;
  // Kernels with a launch in user code whose closure was not recovered: their
  // verdicts cannot hold at that launch.
  llvm::StringSet<> unrecoveredSite;
  // Per-kernel running constant-argument agreement, ANDed across the sites.
  llvm::StringMap<bool> constSeenSite;
  llvm::StringMap<llvm::StringMap<mlir::TypedAttr>> constEligible;
  mlir::ModuleOp hostModule = cir::getOffloadContainerHostModule(container);

  // Throwaway clone -- the real emitted module is never mutated.
  mlir::OwningOpRef<mlir::ModuleOp> clone(
      mlir::cast<mlir::ModuleOp>(hostModule->clone()));

  // This is a throwaway analysis clone, so exception edges do not matter. Mark
  // every call nothrow before flattening: otherwise FlattenCFG lowers throwing
  // calls inside cleanup scopes into cir.try_call terminators, which the
  // call-based inliner below cannot collapse (the launch shims call each other
  // from inside cleanup scopes). With nothrow set, the cleanup scopes flatten
  // to plain control flow and every launch-chain call stays a cir.call.
  clone->walk([](cir::CallOp call) { call.setNothrow(true); });

  // Flatten so cir.return becomes a block terminator (the inliner's
  // handleTerminator only rewrites top-level return terminators).
  mlir::PassManager pm(clone->getContext(), mlir::ModuleOp::getOperationName());
  pm.addPass(mlir::createCIRFlattenCFGPass());
  if (mlir::failed(pm.run(*clone)))
    return result;

  // Synthesize the type-erased-submit call edge (handler path) before inlining,
  // so the handler-submit launch collapses into the user function too.
  bridgeSubmitCGF(*clone);
  inlineLaunchChain(*clone);
  LLVM_DEBUG(llvm::dbgs() << "collapsed launch clone:\n" << *clone << "\n");
  MemoryReader reader(*clone);

  clone->walk([&](cir::CallOp call) {
    if (!isGetDeviceKernelInfoCall(call))
      return;
    cir::GlobalOp nameString = resolveLaunchedNameString(call);
    if (!nameString)
      return;
    std::optional<RecoveredLaunch> launch = recoverLaunch(call->getResult(0));
    llvm::StringRef kernel = getGlobalStringValue(nameString);
    LLVM_DEBUG(llvm::dbgs()
               << "launch of " << kernel << ": "
               << (launch ? "recovered" : "not recovered") << "\n");
    auto enclosing = call->getParentOfType<cir::FuncOp>();
    bool inUserCode = enclosing && !isLaunchChainCallee(enclosing) &&
                      !enclosing->hasAttr("cir.sycl.cg_body");
    if (inUserCode && (!launch || !launch->closure))
      unrecoveredSite.insert(kernel);
    if (!launch)
      return;
    SyclGeometry geom = traceNDRange(reader, launch->geometry);
    result[kernel] = geom;
    // Fold this launch's closure into the kernel's noalias eligibility, but
    // only for the collapsed launch sites in user code. Inlining leaves a copy
    // of each marker inside the still-present launch-shim bodies too, where the
    // closure is only a by-reference parameter (a block argument) with no
    // visible capture stores; folding those would spuriously poison every
    // field. The real launch sites are the inlined ones in non-shim functions.
    if (inUserCode && launch->closure) {
      auto cands = noAliasCandidates.find(kernel);
      if (cands != noAliasCandidates.end()) {
        bool seen = seenSite.lookup(kernel);
        foldClosureNoAlias(reader, launch->closure, cands->second, seen,
                           eligible[kernel]);
        seenSite[kernel] = seen;
      }
      bool constSeen = constSeenSite.lookup(kernel);
      foldClosureConstArgs(reader, launch->closure, constSeen,
                           constEligible[kernel]);
      constSeenSite[kernel] = constSeen;
    }
  });

  // Publish the surviving distinct-memory fields per kernel, and the verdict
  // notes for remarks.
  for (auto &e : eligible) {
    bool unrecovered = unrecoveredSite.contains(e.first());
    llvm::StringSet<> &fields = noAliasFields[e.first()];
    llvm::StringMap<std::string> &notes = noAliasNotes[e.first()];
    for (auto &f : e.second) {
      if (f.second.ok && !unrecovered)
        fields.insert(f.first());
      notes[f.first()] = unrecovered && f.second.ok
                             ? "a launch of the kernel is not recovered"
                             : f.second.note;
    }
  }

  // Publish the surviving agreed-constant scalar fields per kernel.
  for (auto &e : constEligible) {
    llvm::StringMap<mlir::TypedAttr> &fields = constArgs[e.first()];
    for (auto &f : e.second)
      if (f.second)
        fields[f.first()] = f.second;
  }

  return result;
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
      auto uses = mlir::SymbolTable::getSymbolUses(nameString, hostModule);
      if (!uses || uses->empty())
        return;

      KernelBinding &binding = bindings[name];
      // First device module to reach this name owns the SYCL classification;
      // later modules (multi-arch) only append their kernel copy.
      if (binding.deviceKernels.empty()) {
        binding.kind = BindingKind::SYCLName;
        binding.hostNameString = nameString;
        binding.isTULocalOptimizable = isLambdaClosureKernel(kernel, name) &&
                                       !isRegisteredKernel(container, name);
      }
      binding.deviceKernels.push_back(kernel);
    });
  }

  recordSYCLLaunchSites(container);
}

// Records one launch site per host `getDeviceKernelInfo<Kernel>()` marker,
// resolving the kernel it names through the CompileTimeKernelInfo chain and the
// runtime launch call it feeds. Bindings already carry their name-strings.
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

  // Recover constant nd_range work-group sizes once for the whole container
  // (inline the launch-shim chain on a throwaway clone, then trace memory).
  // Keyed by _ZTS mangled name, the same key the bindings use. The same clone
  // also yields the closure capture fields backed by pairwise-distinct USM
  // allocations, so noalias can be inferred without cloning twice.
  // The noalias candidates of each kernel, from its device parameters. Every
  // device copy (multi-arch) must agree on them.
  llvm::StringMap<NoAliasCandidates> noAliasCandidates;
  for (auto &binding : bindings) {
    if (binding.second.kind != BindingKind::SYCLName)
      continue;
    NoAliasCandidates merged;
    for (auto [i, kernel] : llvm::enumerate(binding.second.deviceKernels)) {
      NoAliasCandidates c;
      classifyKernelParams(kernel, c);
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
    noAliasCandidates[binding.first] = std::move(merged);
  }

  llvm::StringMap<llvm::StringSet<>> noAliasFields;
  llvm::StringMap<llvm::StringMap<std::string>> noAliasNotes;
  llvm::StringMap<llvm::StringMap<mlir::TypedAttr>> constArgs;
  llvm::StringMap<SyclGeometry> geometry = computeSYCLGeometry(
      container, noAliasCandidates, noAliasFields, noAliasNotes, constArgs);

  for (auto &binding : bindings) {
    if (binding.second.kind != BindingKind::SYCLName)
      continue;
    auto it = noAliasFields.find(binding.first);
    if (it != noAliasFields.end())
      binding.second.syclNoAliasFields = std::move(it->second);
    auto nit = noAliasNotes.find(binding.first);
    if (nit != noAliasNotes.end())
      binding.second.syclNoAliasNotes = std::move(nit->second);
    auto candIt = noAliasCandidates.find(binding.first);
    if (candIt != noAliasCandidates.end())
      binding.second.syclNoAliasBlocked = candIt->second.poison;
    auto cit = constArgs.find(binding.first);
    if (cit != constArgs.end())
      binding.second.syclConstArgs = std::move(cit->second);
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
    // The launch consumes the DeviceKernelInfo* this marker produced -- either
    // submit_kernel_direct_with_event_impl (direct) or handler::
    // setDeviceKernelInfoPtr (handler submit). A marker whose result never
    // reaches a launch (e.g. inlined away) records nothing rather than a site
    // with no call. On the real (un-inlined) module the geometry/closure
    // operands are co-located with the marker only on the direct path; on the
    // handler path they are recovered on the analysis clone (see
    // computeSYCLGeometry) and read back from the per-kernel maps below, so
    // leaving the Values null here is expected.
    std::optional<RecoveredLaunch> launch = recoverLaunch(call->getResult(0));
    if (!launch)
      return;

    KernelBinding &binding = bindings[it->second];
    LaunchSite site;
    site.kind = BindingKind::SYCLName;
    site.stubCall = launch->anchor;
    site.syclMarkerCall = call;
    site.syclKernelName = it->second;
    site.allLaunchesVisible = binding.isTULocalOptimizable;
    site.syclGeometry = launch->geometry;
    site.syclClosure = launch->closure;
    auto geomIt = geometry.find(it->second);
    if (geomIt != geometry.end())
      site.syclNDRange = geomIt->second;
    binding.launchSites.push_back(site);
  });
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
         << (binding.second.isTULocalOptimizable ? "optimizable"
                                                 : "conservative")
         << ")\n";
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
        os << " (sycl, "
           << (launch.allLaunchesVisible ? "all-visible" : "partial")
           << "), closure " << (launch.getSYCLClosure() ? "traced" : "none");
        // Recovered constant nd_range sizes, per dimension. "?" is a dimension
        // that was not a compile-time constant; an empty local list is a plain
        // range launch with no explicit work-group size.
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
// copyKernel: specialize a SYCL kernel by clone-and-redirect
//===----------------------------------------------------------------------===//

// The CompileTimeKernelInfo global a getDeviceKernelInfo getter reads to name
// its kernel: the (only) global it `cir.get_global`s whose symbol carries the
// CompileTimeKernelInfo tag. Mirrors resolveLaunchedNameString's walk.
static cir::GlobalOp findGetterCompileTimeInfo(cir::FuncOp getter) {
  cir::GlobalOp info;
  getter.walk([&](cir::GetGlobalOp gg) {
    auto g = mlir::SymbolTable::lookupNearestSymbolFrom<cir::GlobalOp>(
        gg, gg.getNameAttr());
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

std::optional<cir::SpecializedKernel> cir::specializeSYCLKernelForLaunches(
    mlir::ModuleOp container, const cir::KernelBinding &binding,
    llvm::ArrayRef<cir::LaunchSite> sites, llvm::StringRef uniqueTag) {
  // Only a SYCL launch we can redirect: it needs at least one marker call (the
  // redirect edit point), a device kernel to clone, and a host name-string to
  // re-mint. The mint is driven by the first site's marker; every site in
  // `sites` shares that kernel's getter instantiation.
  if (binding.kind != cir::BindingKind::SYCLName || sites.empty() ||
      !sites.front().syclMarkerCall || binding.deviceKernels.empty() ||
      !binding.hostNameString)
    return std::nullopt;
  const cir::LaunchSite &site = sites.front();

  mlir::ModuleOp hostModule = cir::getOffloadContainerHostModule(container);
  mlir::MLIRContext *ctx = container.getContext();

  // A fresh, unique clone name. The name is an arbitrary raw string key
  // downstream (sycl-post-link enumerates spir_kernels by calling-convention +
  // getName, no demangling), so ".spec.<tag>" is fine; keep it unique across
  // every device module's kernel symbols. (MLIR op handles are value wrappers
  // whose accessors are non-const, so copy out of the const binding first.)
  cir::FuncOp firstKernel = binding.deviceKernels.front();
  std::string cloneName = (firstKernel.getName() + ".spec." + uniqueTag).str();
  auto deviceCollides = [&](llvm::StringRef n) {
    for (mlir::ModuleOp dm : cir::getOffloadContainerDeviceModules(container))
      if (mlir::SymbolTable::lookupSymbolIn(dm, n))
        return true;
    return false;
  };
  for (unsigned n = 0; deviceCollides(cloneName); ++n)
    cloneName =
        (firstKernel.getName() + ".spec." + uniqueTag + "." + llvm::Twine(n))
            .str();

  cir::SpecializedKernel spec;
  spec.name = cloneName;

  // 1. Device clone(s): one per arch module, external linkage inherited from
  // the original (an internal clone is dead-stripped at sycl-post-link), tagged
  // with the combine-private single-caller provenance marker.
  auto specPrivate = mlir::UnitAttr::get(ctx);
  for (cir::FuncOp kernel : binding.deviceKernels) {
    auto clone = mlir::cast<cir::FuncOp>(kernel->clone());
    mlir::SymbolTable::setSymbolName(clone, cloneName);
    clone->setAttr("cir.specialization_private", specPrivate);
    mlir::OpBuilder b(kernel.getOperation());
    b.setInsertionPointAfter(kernel.getOperation());
    b.insert(clone);
    spec.deviceKernels.push_back(clone);
  }

  mlir::SymbolTable hostSyms(hostModule);

  // 2. H1: a mangled-name string global holding the clone name. Cloned from the
  // original name-string so linkage/visibility match, then retyped and
  // re-inited for the (different-length) clone name.
  cir::GlobalOp nameStr = binding.hostNameString;
  cir::GlobalOp h1 = mlir::cast<cir::GlobalOp>(nameStr->clone());
  auto origArrTy = mlir::cast<cir::ArrayType>(nameStr.getSymType());
  mlir::Type eltTy = origArrTy.getElementType();
  uint64_t nameLen = cloneName.size();
  auto fullTy = cir::ArrayType::get(eltTy, nameLen + 1);
  auto strAttr =
      mlir::StringAttr::get(cloneName, cir::ArrayType::get(eltTy, nameLen));
  h1.setSymType(fullTy);
  h1.setInitialValueAttr(cir::ConstArrayAttr::get(fullTy, strAttr));
  mlir::StringAttr h1Name = hostSyms.insert(h1);

  // 3. Resolve the getter this launch marker calls and the
  // CompileTimeKernelInfo it names.
  cir::CallOp marker = site.syclMarkerCall;
  auto getter = mlir::SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(
      marker, marker.getCalleeAttr());
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
  mlir::StringAttr h2Name = hostSyms.insert(h2);

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
    auto g = mlir::SymbolTable::lookupNearestSymbolFrom<cir::GlobalOp>(
        gg, gg.getNameAttr());
    if (!g || g == ctki || !seen.insert(g).second)
      return;
    auto uses = mlir::SymbolTable::getSymbolUses(g, hostModule);
    if (!uses)
      return;
    for (const mlir::SymbolTable::SymbolUse &u : *uses)
      if (!getter->isAncestor(u.getUser()))
        return; // Referenced outside the getter: shared, do not clone.
    auto c = mlir::cast<cir::GlobalOp>(g->clone());
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
  cir::FuncOp h3 = mlir::cast<cir::FuncOp>(getter->clone());
  mlir::StringAttr h3Name = hostSyms.insert(h3);
  h3.walk([&](cir::GetGlobalOp gg) {
    auto it = remap.find(gg.getNameAttr().getAttr());
    if (it != remap.end())
      gg.setNameAttr(mlir::FlatSymbolRefAttr::get(it->second));
  });

  // 7. H4: redirect every requested launch's marker call at the getter clone,
  // moving exactly those launches to the clone while every other launch (this
  // TU's or another's) keeps the original. All sites share the kernel's getter
  // instantiation, so they repoint to the one H3.
  auto h3Ref = mlir::FlatSymbolRefAttr::get(h3Name);
  for (const cir::LaunchSite &s : sites)
    if (cir::CallOp siteMarker = s.syclMarkerCall)
      siteMarker.setCalleeAttr(h3Ref);

  return spec;
}
