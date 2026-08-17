//===- KernelBindingTable.cpp - Host<->device kernel bindings -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "mlir/IR/SymbolTable.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
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
// kernel types never do. This is the identity test for TU-local optimizability:
// a full demangle would be more precise, but the closure marker is enough to
// tell an anonymous kernel from a nameable one.
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
    it->second.launchSites.push_back(
        LaunchSite{stubCall, tracePushConfiguration(stubCall)});
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
        binding.isTULocalOptimizable =
            hasAnonymousClosure(name) && !isRegisteredKernel(container, name);
      }
      binding.deviceKernels.push_back(kernel);
    });
  }
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
      os << "//   launch in @" << caller.getName() << " : "
         << launch.getNumArgs() << " args";
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