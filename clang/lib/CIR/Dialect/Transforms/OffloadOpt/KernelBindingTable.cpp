//===- KernelBindingTable.cpp - Host<->device kernel bindings -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

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
static bool isRegisteredKernel(mlir::ModuleOp container,
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
  for (mlir::ModuleOp deviceMod : cir::getOffloadContainerDeviceModules(container))
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
  if (!binding)
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

  for (mlir::ModuleOp deviceMod : cir::getOffloadContainerDeviceModules(container)) {
    for (auto &binding : bindings) {
      if (cir::FuncOp kernel = llvm::dyn_cast_if_present<cir::FuncOp>(
              deviceMod.lookupSymbol(binding.first))) {
        binding.second.deviceKernels.push_back(kernel);
      }
    }
  }

  bindSYCLKernels(container);
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

// Follow the DeviceKernelInfo* result of a getDeviceKernelInfo call to the
// runtime launch call that consumes it. CIRGen spills the pointer into a stack
// slot and reloads it before the call, so hop store->slot->load once.
static cir::CallOp traceLaunchCall(mlir::Value deviceKernelInfo) {
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
    for (mlir::Operation *user : v.getUsers())
      if (auto call = mlir::dyn_cast<cir::CallOp>(user))
        if (isSubmitKernelDirectImpl(call))
          return call;
  return {};
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
  llvm::StringRef n = fn.getName();
  if (n.starts_with("_ZN4sycl3_V136submit_kernel_direct_with_event_impl"))
    return false;
  if (n.starts_with("_ZN4sycl3_V16detail19getDeviceKernelInfoI"))
    return false;
  // queue::parallel_for and the submit_kernel_direct* shims.
  if (n.contains("submit_kernel_direct") || n.contains("parallel_for"))
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
      if (!fn || !isLaunchChainCallee(fn))
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

// A memory-access path step, root-to-leaf: a named struct member or a constant
// array index.
namespace {
struct PathStep {
  llvm::StringRef member;
  int64_t index = 0;
  bool isIndex = false;
};
using AccessPath = llvm::SmallVector<PathStep, 6>;
} // namespace

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

// Peel pointer-navigation ops off `ptr`, recording the access path from the
// root object down to `ptr`. base_class_addr and bitcast are offset-0
// reinterprets (the base subobject shares the head), so they add no step;
// array_to_ptrdecay yields element 0. Returns the root pointer (an alloca or an
// opaque value) with `path` filled root-to-leaf.
static mlir::Value decomposePointer(mlir::Value ptr, AccessPath &path) {
  AccessPath rev;
  while (mlir::Operation *def = ptr.getDefiningOp()) {
    if (auto gm = mlir::dyn_cast<cir::GetMemberOp>(def)) {
      PathStep s;
      s.member = gm.getName();
      rev.push_back(s);
      ptr = gm.getAddr();
      continue;
    }
    if (auto ge = mlir::dyn_cast<cir::GetElementOp>(def)) {
      PathStep s;
      s.isIndex = true;
      s.index = asConstInt(ge.getIndex()).value_or(-1);
      rev.push_back(s);
      ptr = ge.getBase();
      continue;
    }
    if (auto bca = mlir::dyn_cast<cir::BaseClassAddrOp>(def)) {
      ptr = bca.getSrcAddr();
      continue;
    }
    if (auto cast = mlir::dyn_cast<cir::CastOp>(def)) {
      if (cast.getKind() == cir::CastKind::bitcast) {
        ptr = cast.getSrc();
        continue;
      }
      if (cast.getKind() == cir::CastKind::array_to_ptrdecay) {
        PathStep s;
        s.isIndex = true;
        s.index = 0;
        rev.push_back(s);
        ptr = cast.getSrc();
        continue;
      }
    }
    break;
  }
  for (auto it = rev.rbegin(); it != rev.rend(); ++it)
    path.push_back(*it);
  return ptr;
}

static bool stepEq(const PathStep &a, const PathStep &b) {
  if (a.isIndex != b.isIndex)
    return false;
  return a.isIndex ? a.index == b.index : a.member == b.member;
}

// If `pre` is a prefix of `full`, returns the remaining suffix; else nullopt.
static std::optional<AccessPath> matchPrefix(const AccessPath &pre,
                                             const AccessPath &full) {
  if (pre.size() > full.size())
    return std::nullopt;
  for (size_t i = 0; i < pre.size(); ++i)
    if (!stepEq(pre[i], full[i]))
      return std::nullopt;
  return AccessPath(full.begin() + pre.size(), full.end());
}

// Resolve the constant integer stored at memory cell (`root` + `target` path),
// following whole-object and sub-object writes back through the aggregate
// load/store/copy/bitcast chain the launch shims build. Returns nullopt if the
// cell is not written by a compile-time constant.
static std::optional<int64_t>
resolveScalarAt(mlir::Value root, const AccessPath &target, unsigned depth) {
  if (depth > 64 || !root)
    return std::nullopt;

  // A loaded value carries the bytes of the object it was loaded from; read the
  // same path within that object.
  if (auto load = root.getDefiningOp<cir::LoadOp>()) {
    AccessPath p;
    mlir::Value r = decomposePointer(load.getAddr(), p);
    p.append(target.begin(), target.end());
    return resolveScalarAt(r, p, depth + 1);
  }

  // Gather every pointer that addresses into `root`'s storage: `root` itself
  // and the results of the navigation ops (get_member / get_element /
  // base_class / bitcast) applied to it, transitively. A store into a subfield
  // writes through such a navigation result, so it is a user of that result,
  // not of `root`.
  llvm::SmallVector<mlir::Value, 16> aliases{root};
  llvm::SmallPtrSet<mlir::Operation *, 32> seen;
  for (unsigned ai = 0; ai < aliases.size(); ++ai) {
    mlir::Value ptr = aliases[ai];
    for (mlir::Operation *user : ptr.getUsers()) {
      if (mlir::isa<cir::GetMemberOp, cir::GetElementOp, cir::BaseClassAddrOp>(
              user)) {
        aliases.push_back(user->getResult(0));
        continue;
      }
      if (auto cast = mlir::dyn_cast<cir::CastOp>(user)) {
        if (cast.getKind() == cir::CastKind::bitcast ||
            cast.getKind() == cir::CastKind::array_to_ptrdecay)
          aliases.push_back(cast.getResult());
        continue;
      }
      mlir::Value destPtr, srcVal;
      bool isCopy = false;
      if (auto st = mlir::dyn_cast<cir::StoreOp>(user)) {
        destPtr = st.getAddr();
        srcVal = st.getValue();
      } else if (auto cp = mlir::dyn_cast<cir::CopyOp>(user)) {
        destPtr = cp.getDst();
        srcVal = cp.getSrc();
        isCopy = true;
      } else {
        continue;
      }
      // Only a write *through this pointer* (not one that merely uses it as a
      // stored value) writes the aggregate here.
      if (destPtr != ptr)
        continue;
      if (!seen.insert(user).second)
        continue;
      AccessPath sp;
      mlir::Value sroot = decomposePointer(destPtr, sp);
      if (sroot != root)
        continue;
      std::optional<AccessPath> rem = matchPrefix(sp, target);
      if (!rem)
        continue;

      if (isCopy) {
        // cir.copy transfers a subobject from a source pointer; read the
        // remaining path within that source.
        AccessPath p;
        mlir::Value r = decomposePointer(srcVal, p);
        p.append(rem->begin(), rem->end());
        if (auto v = resolveScalarAt(r, p, depth + 1))
          return v;
        continue;
      }
      if (rem->empty()) {
        // Exact scalar cell.
        if (auto v = asConstInt(srcVal))
          return v;
        if (auto load = srcVal.getDefiningOp<cir::LoadOp>()) {
          AccessPath p;
          mlir::Value r = decomposePointer(load.getAddr(), p);
          if (auto v = resolveScalarAt(r, p, depth + 1))
            return v;
        }
        continue;
      }
      // The store wrote an enclosing aggregate; its value must be a loaded
      // aggregate -- read the remaining path within its source object.
      if (auto load = srcVal.getDefiningOp<cir::LoadOp>()) {
        AccessPath p;
        mlir::Value r = decomposePointer(load.getAddr(), p);
        p.append(rem->begin(), rem->end());
        if (auto v = resolveScalarAt(r, p, depth + 1))
          return v;
      }
    }
  }
  return std::nullopt;
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

// The value stored into `view`'s named field, or null. Matches the get_member
// naming the field and the store into it.
static mlir::Value findFieldStoredValue(mlir::Value view,
                                        llvm::StringRef field) {
  for (mlir::Operation *u : view.getUsers()) {
    auto gm = mlir::dyn_cast<cir::GetMemberOp>(u);
    if (!gm || gm.getName() != field)
      continue;
    for (mlir::Operation *gu : gm.getResult().getUsers())
      if (auto st = mlir::dyn_cast<cir::StoreOp>(gu))
        if (st.getAddr() == gm.getResult())
          return st.getValue();
  }
  return {};
}

// Recover the constant nd_range geometry from the nd_range_view pointer operand
// of a runtime launch call (on the inlined clone). The view constructor stored
// interior pointers into its MGlobalSize/MLocalSize fields that alias the
// nd_range object's array slots; trace those back to the constant stores.
static SyclGeometry traceNDRange(mlir::Value viewOperand) {
  SyclGeometry g;
  mlir::Value view = stripToPointerObject(viewOperand);
  if (!view)
    return g;

  unsigned dims = 0;
  if (mlir::Value d = findFieldStoredValue(view, "MDims"))
    if (auto dv = asConstInt(d))
      dims = static_cast<unsigned>(*dv);
  if (dims == 0 || dims > 3)
    dims = 1;
  g.dims = dims;

  // Read `dims` consecutive elements of the size array a field points at. The
  // field holds &common_array[0]; vary the trailing constant index per dim.
  auto readSizes =
      [&](llvm::StringRef field,
          llvm::SmallVectorImpl<std::optional<int64_t>> &out) -> bool {
    mlir::Value interior = findFieldStoredValue(view, field);
    if (!interior)
      return false;
    // A plain-range launch stores a null pointer for MLocalSize (no explicit
    // local work-group size): that field is absent, not unrecovered.
    if (auto c = interior.getDefiningOp<cir::ConstantOp>())
      if (auto p = mlir::dyn_cast<cir::ConstPtrAttr>(c.getValue()))
        if (p.isNullValue())
          return false;
    AccessPath base;
    mlir::Value root = decomposePointer(interior, base);
    for (unsigned i = 0; i < dims; ++i) {
      AccessPath p = base;
      if (!p.empty() && p.back().isIndex)
        p.back().index = static_cast<int64_t>(i);
      out.push_back(resolveScalarAt(root, p, 0));
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

// Recover per-kernel constant nd_range geometry for the whole container. Clones
// the host module, collapses the launch chain, then for each surviving
// getDeviceKernelInfo marker resolves the kernel it names and traces the launch
// it feeds. Keyed by the kernel's _ZTS mangled name so the result maps straight
// onto the SYCL bindings, which are keyed the same way.
static llvm::StringMap<SyclGeometry>
computeSYCLGeometry(mlir::ModuleOp container) {
  llvm::StringMap<SyclGeometry> result;
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

  inlineLaunchChain(*clone);

  clone->walk([&](cir::CallOp call) {
    if (!isGetDeviceKernelInfoCall(call))
      return;
    cir::GlobalOp nameString = resolveLaunchedNameString(call);
    if (!nameString)
      return;
    cir::CallOp launch = traceLaunchCall(call->getResult(0));
    if (!launch)
      return;
    mlir::OperandRange args = launch.getArgOperands();
    if (args.size() <= SyclDeviceKernelInfoOperand)
      return;
    SyclGeometry geom = traceNDRange(args[SyclGeometryOperand]);
    result[getGlobalStringValue(nameString)] = geom;
  });

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

  for (mlir::ModuleOp deviceMod : cir::getOffloadContainerDeviceModules(container)) {
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
void KernelBindingTable::recordSYCLLaunchSites(
    mlir::ModuleOp container) {
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
  // Keyed by _ZTS mangled name, the same key the bindings use.
  llvm::StringMap<SyclGeometry> geometry = computeSYCLGeometry(container);

  hostModule.walk([&](cir::CallOp call) {
    if (!isGetDeviceKernelInfoCall(call))
      return;
    cir::GlobalOp nameString = resolveLaunchedNameString(call);
    if (!nameString)
      return;
    auto it = byNameString.find(nameString.getOperation());
    if (it == byNameString.end())
      return;
    // The launch op consumes the DeviceKernelInfo* this marker produced. A
    // marker whose result never reaches a launch (e.g. inlined away) records
    // nothing rather than a site with no call.
    cir::CallOp launch = traceLaunchCall(call->getResult(0));
    if (!launch)
      return;

    KernelBinding &binding = bindings[it->second];
    LaunchSite site;
    site.kind = BindingKind::SYCLName;
    site.stubCall = launch;
    site.syclKernelName = it->second;
    site.allLaunchesVisible = binding.isTULocalOptimizable;
    mlir::OperandRange args = launch.getArgOperands();
    if (args.size() > SyclDeviceKernelInfoOperand) {
      site.syclGeometry = args[SyclGeometryOperand];
      site.syclClosure = args[SyclClosureOperand];
    }
    auto geomIt = geometry.find(it->second);
    if (geomIt != geometry.end())
      site.syclNDRange = geomIt->second;
    binding.launchSites.push_back(site);
  });
}

const KernelBinding *KernelBindingTable::lookup(llvm::StringRef kernelName) const {
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