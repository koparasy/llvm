//===- KernelBindingTable.h - Host<->device kernel bindings -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Resolves the host <-> device kernel correspondence inside a
// cir.offload.container, along with the host call sites that launch each
// kernel.
//
// Two binding shapes are recognized:
//   - CUDA/HIP: the bridge is the `cu.kernel_name` attribute, whose value is
//     the device kernel's mangled symbol name; both the host device-stub and
//     the calls to it carry it.
//   - SYCL: there is no host stub. The host references a device kernel by its
//     mangled-name string constant (a `cir.global` char array whose value
//     equals the device kernel's symbol, e.g. the string returned by
//     `FreeFunctionInfoData<Func>::getFunctionName()`). The bridge is that
//     string, matched by equality against a `cc(spir_kernel)` device func.
//
// Usable as an MLIR analysis: getAnalysis<KernelBindingTable>() on a pass
// anchored at the offload container module.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_KERNELBINDINGTABLE_H
#define LLVM_CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_KERNELBINDINGTABLE_H

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <optional>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace cir {

// How a binding's host side was discovered. CUDA/HIP anchors on a device-stub
// FuncOp carrying cu.kernel_name; SYCL has no stub -- the host names the device
// kernel through a mangled-name string constant.
enum class BindingKind { CUDAStub, SYCLName };

// Recovered constant nd_range geometry for a SYCL launch: the per-dimension
// global and (optionally) local work-group sizes.
//
// The sizes do not survive as SSA constants at the launch call -- CIRGen buries
// them in the nd_range object's memory behind a chain of launch shims. They are
// recovered by inlining that shim chain on a throwaway clone of the host module
// and tracing the nd_range_view object's memory back to the constant stores
// (see SYCL geometry recovery in KernelBindingTable.cpp). Because the clone is
// discarded, the result is held as plain integers, not IR values.
//
// A per-dimension entry is nullopt when that component was not a compile-time
// constant. `local` is empty for a plain `range` launch (RoundedRangeKernel),
// which carries no explicit local size. `valid` is false when recovery did not
// run or could not locate the geometry at all.
struct SyclGeometry {
  llvm::SmallVector<std::optional<int64_t>, 3> global;
  llvm::SmallVector<std::optional<int64_t>, 3> local;
  unsigned dims = 0;
  bool valid = false;
};

// One host-side launch of a kernel.
//
// CUDA: `stubCall` is the call to the device stub, which carries the kernel
// arguments, and `pushConfigCall` is the `__cudaPushCallConfiguration` /
// `__hipPushCallConfiguration` call preceding it, which carries the geometry.
// Only the two anchor calls are stored; argument values, launch dimensions and
// their constants are read back out of the IR on demand, so a descriptor can
// never report a value the IR has since stopped holding.
//
// SYCL: `stubCall` is the runtime launch call
// (`submit_kernel_direct_with_event_impl`) reached from a
// `getDeviceKernelInfo<Kernel>()` marker; `syclGeometry`/`syclClosure` are the
// nd_range_view and HostKernelRef pointer operands of that call. The CUDA
// geometry accessors do not apply (`hasGeometry()` is false).
struct LaunchSite {
  BindingKind kind = BindingKind::CUDAStub;

  cir::CallOp stubCall;

  // Null when the geometry could not be traced: the launch was not emitted in
  // the shape CIRGen produces, or the CFG has already been flattened. Always
  // null for SYCL.
  cir::CallOp pushConfigCall;

  // SYCL only. Operands of the runtime launch call, kept as Values so constants
  // are read back on demand (mirrors the CUDA accessors). Null for CUDA.
  mlir::Value syclGeometry; // nd_range_view* operand
  mlir::Value syclClosure;  // HostKernelRef(Base)* operand

  // SYCL only. The bound _ZTS mangled-name string this launch targets, and
  // whether every launch of it is visible in this TU (the binding is
  // TU-local/optimizable) so the combine may const-fold across all sites.
  llvm::StringRef syclKernelName;
  bool allLaunchesVisible = false;

  // SYCL only. Constant nd_range work-group sizes recovered for this launch by
  // the geometry-recovery pre-analysis (inline the launch-shim chain on a
  // throwaway clone, then trace the nd_range object memory). Default-invalid
  // when recovery did not run or found nothing.
  SyclGeometry syclNDRange;

  mlir::Value getSYCLGeometry() const { return syclGeometry; }
  mlir::Value getSYCLClosure() const { return syclClosure; }
  const SyclGeometry &getSYCLNDRange() const { return syclNDRange; }

  llvm::StringRef getKernelName() const;

  unsigned getNumArgs() const;
  mlir::Value getArg(unsigned i) const;
  // The constant passed for argument `i`, or null if it is not a constant.
  mlir::TypedAttr getConstArg(unsigned i) const;

  // A launch dimension, recovered from the dim3 temporary the push call reads.
  //
  // Failing to recover a dimension and recovering a dimension that turns out to
  // be a runtime value are different answers: `isTraced` reports the first,
  // `isFullyConstant` the second. A component is null when it was not traced.
  struct Dim3 {
    mlir::Value x, y, z;
    mlir::TypedAttr constX() const;
    mlir::TypedAttr constY() const;
    mlir::TypedAttr constZ() const;
    bool isTraced() const;
    bool isFullyConstant() const;
  };

  Dim3 getGridDim() const;
  Dim3 getBlockDim() const;

  mlir::Value getSharedMemBytes() const;
  mlir::TypedAttr getConstSharedMem() const;

  mlir::Value getStream() const;
  bool isDefaultStream() const;

  bool hasGeometry() const { return pushConfigCall != nullptr; }
};

// Binding for one kernel, keyed by its device mangled name. A name may resolve
// in several device modules (multi-arch); the owning module of each kernel is
// recoverable via k->getParentOfType<mlir::ModuleOp>().
//
// CUDA: `hostStub` is the device stub, and `launchSites` are the host calls to
// it, which carry the same `cu.kernel_name` key as the stub itself.
//
// SYCL: `hostNameString` is the host string global whose value equals the
// device kernel's mangled name and which host code references; there is no
// stub. `launchSites` are the runtime launch calls reached from each
// `getDeviceKernelInfo<Kernel>()` marker naming this kernel.
struct KernelBinding {
  BindingKind kind = BindingKind::CUDAStub;

  // CUDA: the host device-stub carrying cu.kernel_name. Null for SYCL.
  cir::FuncOp hostStub;

  // SYCL: the host mangled-name string global naming the device kernel. Null
  // for CUDA.
  cir::GlobalOp hostNameString;

  // SYCL only: true iff the kernel's CLOSURE is a lambda (a compiler-generated,
  // TU-unique, unspellable type) and it is not by-name/bundle-exposed (absent
  // from sycl_registered_kernels), so its launches are fully TU-local and the
  // combine may optimize/eliminate it. This keys on the closure type, not the
  // kernel-name tag: `parallel_for<class Tag>(lambda)` has a named `_ZTS<Tag>`
  // symbol but a lambda closure, and is still TU-local. A genuine functor
  // (named struct with operator()), a free function, or a registered kernel is
  // cross-TU/by-name reachable and stays conservative (false). Always false for
  // CUDA (stub-anchored).
  bool isTULocalOptimizable = false;

  // SYCL only: names of closure capture fields whose captured pointer is, at
  // EVERY visible launch site, backed by a USM allocation (malloc_device /
  // malloc_shared / malloc_host) that is provably distinct from every other
  // captured pointer of the same closure. The device kernel parameter bound to
  // such a field can carry `llvm.noalias`. Populated only for TU-local
  // optimizable bindings; empty when recovery could not prove distinctness
  // (unresolved source, shared base, pointer arithmetic -> conservatively out).
  llvm::StringSet<> syclNoAliasFields;

  // SYCL only: closure capture fields whose stored SCALAR value is, at EVERY
  // visible launch site, the SAME compile-time constant. Maps the field name to
  // that constant (an int/float/bool cir.const value). The device kernel
  // parameter bound to such a field can have its uses replaced by the constant
  // (the argument becomes unused; the signature/ABI is preserved). Populated
  // only for TU-local optimizable bindings; a field whose value is non-constant
  // at any site, or whose sites disagree, is absent (conservatively out). This
  // is the scalar analogue of syclNoAliasFields.
  llvm::StringMap<mlir::TypedAttr> syclConstArgs;

  llvm::SmallVector<cir::FuncOp, 2> deviceKernels;
  llvm::SmallVector<LaunchSite, 2> launchSites;
};

// The kernel `op` launches, or a null attribute if it is not a launch site.
// This is what the table records as a launch site, and what a pass must accept
// when checking that it can see every launch of a stub: the two answers come
// from here so they cannot drift apart.
cir::CUDAKernelNameAttr getLaunchedKernel(mlir::Operation *op);

// Whether any reference to `fn` within `scope` lives outside `fn`'s own body. A
// device stub hands its own address to cudaLaunchKernel, so it always
// references itself; that self-reference is not a use. Conservatively true when
// the uses cannot be enumerated.
bool hasUseOutsideSelf(cir::FuncOp fn, mlir::Operation *scope);

class KernelBindingTable {
public:
  // `container` is the offload container module, i.e. a module carrying the
  // `cir.offload.container` unit attribute. The host and device modules are
  // recovered from it via the `cir` container helpers.
  explicit KernelBindingTable(mlir::ModuleOp container);

  const KernelBinding *lookup(llvm::StringRef kernelName) const;

  // Device kernels bound to a host stub. The stub carries the `cu.kernel_name`
  // key itself, so a pass holding a launch callee can resolve device kernels
  // without re-deriving the mangled name. Empty if the func is not a stub.
  llvm::ArrayRef<cir::FuncOp> getDeviceFuncs(cir::FuncOp hostFn) const;

  // Host calls launching the named kernel. Empty if the name is unbound or the
  // kernel is never launched in this TU.
  llvm::ArrayRef<cir::LaunchSite>
  getLaunchSites(llvm::StringRef kernelName) const;

  // Whether the recorded launch sites of the named kernel are all of them, so a
  // pass may draw conclusions from them about the kernel itself. False when
  // another TU can call the stub, when a reference to it is not a launch this
  // table recorded, or when the name is unbound.
  bool allLaunchSitesVisible(llvm::StringRef kernelName) const;

  bool empty() const { return bindings.empty(); }
  size_t size() const { return bindings.size(); }
  auto begin() const { return bindings.begin(); }
  auto end() const { return bindings.end(); }

  void print(llvm::raw_ostream &os) const;

private:
  // Adds the SYCL name-string <-> spir_kernel bindings and their eligibility.
  void bindSYCLKernels(mlir::ModuleOp container);

  // Records a launch site on each SYCL binding for every host
  // `getDeviceKernelInfo<Kernel>()` marker that names it. Runs after
  // bindSYCLKernels, so every binding already has its name-string.
  void recordSYCLLaunchSites(mlir::ModuleOp container);

  // Keys are StringRefs into the `cu.kernel_name` attribute storage (CUDA) or
  // the device kernel's symbol-name storage (SYCL), so the table's lifetime is
  // tied to the container IR it was built from.
  llvm::MapVector<llvm::StringRef, KernelBinding> bindings;
  // The scope the stubs and launch sites live in, so a visibility question
  // about a binding does not depend on the caller passing the right module.
  mlir::ModuleOp hostModule;
};

} // namespace cir

#endif // LLVM_CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_KERNELBINDINGTABLE_H