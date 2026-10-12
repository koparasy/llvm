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
//     The launches are found by the host-API roles CIRGen tags the SYCL
//     launch API with (cir.sycl_host_api): the kernel.info marker and the
//     launch call it feeds. The host facts of a SYCL kernel (noalias fields,
//     scalar-capture constants, launch geometry) come from SYCLHostFacts,
//     the points-to analysis of the host module.
//
// Usable as an MLIR analysis: getAnalysis<KernelBindingTable>() on a pass
// anchored at the offload container module.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_KERNELBINDINGTABLE_H
#define LLVM_CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_KERNELBINDINGTABLE_H

#include "mlir/IR/SymbolTable.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <memory>
#include <optional>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace cir {

class SYCLHostFactsCache;

// How a binding's host side was discovered. CUDA/HIP anchors on a device-stub
// FuncOp carrying cu.kernel_name; SYCL has no stub -- the host names the device
// kernel through a mangled-name string constant.
enum class BindingKind { CUDAStub, SYCLName };

// Recovered constant nd_range geometry for a SYCL launch: the per-dimension
// global and (optionally) local work-group sizes.
//
// The sizes do not survive as SSA constants at the launch call -- CIRGen buries
// them in the nd_range object's memory behind a chain of launch shims. The
// stored-value queries of the points-to analysis recover them
// (SYCLHostFacts::geometry); the result is held as plain integers.
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

// "global=[g0,g1] local=[l0,l1]" ("?" for a component that is not a
// constant; no local part for a plain range launch), or "none" if invalid.
std::string formatSyclGeometry(const SyclGeometry &geometry);

// One host-side launch of a kernel.
//
// CUDA: `stubCall` is the call to the device stub, which carries the kernel
// arguments, and `pushConfigCall` is the `__cudaPushCallConfiguration` /
// `__hipPushCallConfiguration` call preceding it, which carries the geometry.
// Only the two anchor calls are stored; argument values, launch dimensions and
// their constants are read back out of the IR on demand, so a descriptor can
// never report a value the IR has since stopped holding.
//
// SYCL: `stubCall` is the launch call (launch.direct_impl, or
// handler.set_kernel_info on the handler path) a
// `getDeviceKernelInfo<Kernel>()` marker (kernel.info) feeds; `syclClosure` is
// the HostKernelRef pointer operand of that call (direct path) or the lambda
// the handler's HostKernel is built from (handler path). The CUDA geometry
// accessors do not apply (`hasGeometry()` is false).
struct LaunchSite {
  BindingKind kind = BindingKind::CUDAStub;

  cir::CallOp stubCall;

  // Null when the geometry could not be traced: the launch was not emitted in
  // the shape CIRGen produces, or the CFG has already been flattened. Always
  // null for SYCL.
  cir::CallOp pushConfigCall;

  // SYCL only. The closure operand of the launch, or null (CUDA; not
  // recovered).
  mlir::Value syclClosure;

  // SYCL only. The `getDeviceKernelInfo<Kernel>()` marker call that names this
  // launch's kernel and whose DeviceKernelInfo* result feeds `stubCall`. This
  // is the edit point for redirecting the launch to a kernel copy (launch-path
  // internalization repoints its callee at the copy's getter). Null for CUDA.
  cir::CallOp syclMarkerCall;

  // SYCL only. The bound _ZTS mangled-name string this launch targets.
  llvm::StringRef syclKernelName;

  // SYCL only. The constant nd_range work-group sizes of the kernel's
  // launches (SYCLHostFacts::geometry, the same for every launch site of a
  // kernel). Default-invalid without host facts or when nothing is known.
  SyclGeometry syclNDRange;

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

// Where KernelBinding::syclNoAliasFields, syclConstArgs and the launch
// sites' syclNDRange come from (the --cir-offload-host-facts flag of
// cir-offload-merge):
//  - IPO: SYCLHostFacts, built on the points-to analysis of the host module
//    (the default);
//  - None: no host facts (nothing is stamped).
enum class HostFactsMode { None, IPO };

// The name of `mode` as the flag spells it ("none", "ipo"), and the mode a
// name spells, if any.
llvm::StringRef getHostFactsModeName(HostFactsMode mode);
std::optional<HostFactsMode> parseHostFactsMode(llvm::StringRef name);

// The points-to analysis behind the host facts. By default it is the
// bottom-up mode with subobject nodes (and top-down queries); bottomUp =
// false runs the context-insensitive analysis (the passes' host-facts-bu).
struct HostFactsOptions {
  HostFactsMode mode = HostFactsMode::IPO;
  // The points-to analysis treats every function as having only the callers
  // in the module (PointsToAnalysisOptions::assumeClosedTU). Off by default
  // (plan D2); -cir-offload-assume-closed-tu sets it.
  bool assumeClosedTU = false;
  // The points-to analysis runs in its bottom-up mode (PointsToMode::BU):
  // the launch records of a function its callers copied are read in the
  // callers' graphs. On by default.
  bool bottomUp = true;
  /// Give class, union and array members their own points-to nodes
  /// (PointsToAnalysisOptions::subobjects). On by default.
  bool subobjects = true;
  // Multiply every budget of the points-to analysis by this factor
  // (PointsToAnalysisOptions::budgetScale; tests set it tiny).
  double budgetScale = 1.0;
  // Analyse only the functions that can matter to a kernel launch
  // (PointsToAnalysisOptions::relevanceSlice). Off by default.
  bool relevanceSlice = false;
  // The wall-clock (seconds) and memory (MB) the points-to analysis may use
  // (PointsToAnalysisOptions::maxSeconds, maxMemoryMB); past it, the
  // bottom-up analysis falls back to the context-insensitive one, and that
  // to no facts. 0: no limit.
  double maxSeconds = 120;
  unsigned maxMemoryMB = 8192;
  // Two buffers constructed over equal host pointers are distinct storages:
  // constructing more than one buffer managing the same host pointer is
  // undefined behavior (SYCL 2020 PI 5358-5359, plan D1). On by default; the
  // hidden -cir-offload-host-facts-host-ptr-ub=false turns it off.
  bool assumeSpecHostPtrUB = true;
  // The launch paths were internalized (T26,
  // cir-offload-internalize-launch-paths); meaningful only without
  // assumeClosedTU. The points-to analysis counts a local function whose
  // address only analysed code takes as having only visible callers
  // (PointsToAnalysisOptions::localFunctionsClosed). Off by default;
  // cir-offload-merge sets it without closed-tu.
  bool internalize = false;
  // Whether the open-mode analysis rules of internalize apply.
  bool internalizedOpenMode() const {
    return internalize && !assumeClosedTU && mode == HostFactsMode::IPO;
  }
};

// The options that select one points-to analysis: every field but the mode,
// as a key for SYCLHostFactsCache.
std::string getHostFactsAnalysisKey(const HostFactsOptions &options);

// "(closed-tu|open, bu|ci[, subobjects][, internalized])", the analysis
// options the host-facts remarks name.
std::string formatHostFactsOptions(const HostFactsOptions &options);

// Which closure fields of a SYCL kernel are candidates for llvm.noalias, read
// off the device kernel's parameters (cir.sycl_kernel_param): captured USM
// pointers and global-memory accessors held directly by the closure. Any other
// parameter that can carry a pointer to global memory -- a pointer or accessor
// nested in a captured struct, a struct holding a pointer, a stream or
// annotated pointer, a parameter with no recorded field -- may alias the
// candidates without being proven distinct, so it disables noalias for the
// whole kernel. SYCLHostFacts takes its candidates and poison from here.
struct SYCLNoAliasCandidates {
  llvm::StringMap<bool> fields; // field -> holds an accessor (else a pointer)
  std::string poison;           // why noalias is off for the kernel, if it is
};

// The candidates of one device kernel.
void classifySYCLKernelParams(cir::FuncOp kernel, SYCLNoAliasCandidates &out);

// The candidates of a kernel compiled for several devices (multi-arch): every
// device copy must agree on them, else the kernel is poisoned.
SYCLNoAliasCandidates
getSYCLNoAliasCandidates(llvm::ArrayRef<cir::FuncOp> deviceKernels);

// The noalias verdict for one candidate closure field of a kernel, with the
// memory it resolved to or why it was rejected.
struct NoAliasDecision {
  bool ok = false;
  std::string note;
};

// Fold the verdicts of one launch (`here`) into the running verdicts of a
// kernel, ANDed over its launches: a field stays eligible only if it is here
// and at every earlier launch; the first rejection's note is kept, and a field
// missing at a launch is rejected as "not captured at every launch".
void andNoAliasDecisions(llvm::StringMap<NoAliasDecision> here,
                         bool &seenAnySite,
                         llvm::StringMap<NoAliasDecision> &decisions);

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

  // SYCL only: the kernel is registered (by-name / bundle-exposed:
  // cir.sycl_registered_kernels), so the runtime may launch it by name from
  // anywhere.
  bool isRegistered = false;

  // SYCL only: the open rule: no other TU can launch the kernel
  // (isTUUniqueSYCLKernel: an internalized copy, a closure whose call
  // operator has local linkage, or a kernel-name type with internal linkage)
  // and it is not by-name/bundle-exposed. A lambda in an inline function (an
  // in-class member, a header template, an inline variable) has the same
  // closure type in every TU ([basic.def.odr]p14-15) and its device kernel
  // is one weak_odr definition for the program, so "a lambda" is not enough:
  // this TU's facts are an AND over this TU's launches only.
  bool isTUUnique = false;

  // Whether the facts of this TU's launches may be stamped on the kernel in
  // place: without closed-TU, isTUUnique. Under the closed-TU assumption
  // (-cir-offload-assume-closed-tu: every caller of every host function is in
  // this TU) every launch of a kernel that is not registered is one of this
  // TU's: another TU could launch it only through the launch functions of its
  // kernel type (queue::submit, handler::parallel_for<K, ...>, which are this
  // TU's functions too by the ODR), i.e. as a caller of this TU's functions
  // that is not in this TU, or by name, which only a registered kernel
  // allows. So it is stamped in place. Always false for CUDA (stub-anchored).
  bool stampsInPlace(bool assumeClosedTU) const {
    if (kind != BindingKind::SYCLName)
      return false;
    return assumeClosedTU ? !isRegistered : isTUUnique;
  }

  // SYCL only: names of closure capture fields -- captured USM pointers and
  // global-memory accessors -- whose memory is, at EVERY visible launch site,
  // an allocation (a USM or heap allocation, a buffer's storage) provably
  // distinct from the memory of every other such capture of the same closure
  // (SYCLHostFacts::noAliasFields). The device kernel pointer parameter bound
  // to such a field can carry `llvm.noalias`. Empty without host facts or
  // when nothing is proven.
  llvm::StringSet<> syclNoAliasFields;

  // SYCL only: for each noalias candidate field, the memory it resolved to if
  // it is in syclNoAliasFields, or why it is not -- for remarks.
  llvm::StringMap<std::string> syclNoAliasNotes;

  // SYCL only: why noalias is off for every parameter of the kernel (e.g. a
  // pointer or accessor nested in a captured struct), or empty.
  std::string syclNoAliasBlocked;

  // SYCL only: closure capture fields whose stored SCALAR value is, at EVERY
  // visible launch site, the SAME compile-time constant. Maps the field name to
  // that constant (an int/float/bool cir.const value). The device kernel
  // parameter bound to such a field can have its uses replaced by the constant
  // (the argument becomes unused; the signature/ABI is preserved). A field
  // whose value is non-constant at any site, or whose sites disagree, is
  // absent (SYCLHostFacts::constantFields). This is the scalar analogue of
  // syclNoAliasFields.
  llvm::StringMap<mlir::TypedAttr> syclConstArgs;

  // SYCL only, with host facts: for every scalar-capture candidate field,
  // where its constant was found or why none is known; and the same for the
  // kernel's geometry (SYCLHostFacts::ConstFact, GeometryFact), for remarks
  // and the table's print.
  llvm::StringMap<std::string> syclConstNotes;
  std::string syclGeometryNote;

  llvm::SmallVector<cir::FuncOp, 2> deviceKernels;
  llvm::SmallVector<LaunchSite, 2> launchSites;
};

// On a device kernel copy made for an internalized launch path (T26): a
// dictionary {original = "<K>", roots = ["main", ...]}.
inline constexpr llvm::StringLiteral kInternalizedKernelAttr =
    "cir.offload.internalized_kernel";
// On such a copy, once a pass applied a host fact to it (a noalias, a
// required work-group size, a constant); the prune pass keeps only these.
inline constexpr llvm::StringLiteral kInternalizedUsedAttr =
    "cir.offload.internalized_used";
// The call operator of the closure the SYCL device kernel `kernel` invokes
// (its closure alloca passed as the receiver), or null.
cir::FuncOp getSYCLClosureCallOperator(cir::FuncOp kernel);
// Whether `kernel` is a copy made for an internalized launch path.
bool isInternalizedSYCLKernel(cir::FuncOp kernel);
// Record that a pass applied a host fact to `kernel` (no-op for a kernel
// that is not an internalized copy).
void noteInternalizedSYCLKernelUse(cir::FuncOp kernel);
// For an internalized copy: " through the internalized launch path from
// root @main[, @...]" (for the remarks), else empty.
std::string describeInternalizedSYCLKernel(cir::FuncOp kernel);
// Why a fact of a kernel another TU can launch (not isTUUnique) is not
// stamped on it without -cir-offload-assume-closed-tu, for the remarks;
// `internalized`: the launch paths were internalized.
std::string getSharedSYCLKernelNote(bool internalized);
// Whether no other TU can launch the device kernel `kernel` (named `name`):
// it is an internalized copy, its closure's call operator has local linkage
// (a lambda in a non-inline function, a functor in an unnamed namespace), or
// every host kernel.info getter naming it has (the kernel-name type has
// internal linkage). Another TU's kernel of that name would be another
// kernel with the same name, which SYCL excludes ("the kernels have to be
// uniquely identified by both host and device compiler",
// sec:naming.kernels).
// `getterLocality`, if given, is getSYCLKernelInfoLocality(container).
bool isTUUniqueSYCLKernel(
    mlir::ModuleOp container, cir::FuncOp kernel, llvm::StringRef name,
    const llvm::StringMap<bool> *getterLocality = nullptr);
// Whether `name` is a registered (by-name / bundle-exposed) kernel of
// `container` (cir.sycl_registered_kernels): reachable by name across TUs.
bool isRegisteredSYCLKernel(mlir::ModuleOp container, llvm::StringRef name);
// For each kernel name a host kernel.info getter names: whether every such
// getter has local linkage.
llvm::StringMap<bool> getSYCLKernelInfoLocality(mlir::ModuleOp container);

// A copy of a SYCL kernel for launch-path internalization (T26), made by
// copySYCLKernelLocally: the device clones under `name`, and the getter copy
// (H3) whose callers launch them.
struct LocalKernelCopy {
  std::string original;
  std::string name;
  llvm::SmallVector<cir::FuncOp, 2> deviceKernels;
  cir::FuncOp getter;
  // The original's naming artifacts (its name string, CompileTimeKernelInfo
  // and getter) and the copies H1, H2 of the first two.
  cir::GlobalOp nameString, compileTimeInfo, h1, h2;
  cir::FuncOp originalGetter;
};

// Makes a copy of the kernel a kernel.info getter names: a device spir_kernel
// clone of each of its device kernels under `cloneName`, and the host
// artifacts that make the runtime resolve it: a mangled-name string global
// (H1), a CompileTimeKernelInfo copy naming that string (H2), and a
// getDeviceKernelInfo getter copy plus its per-instantiation static-local /
// guard cluster (H3). Every host artifact is local to this TU: internal
// linkage, named `<original><suffix>`, tagged cir.offload.internalized (so
// that no other TU can bind to them). The device clones keep the
// original's linkage (sycl-post-link dead-strips an internal kernel): the
// clone name must be unique in the program (the caller derives it from a
// module id). Nothing is redirected: the caller repoints the marker calls it
// chose at the getter copy. The symbol tables and the uses of the host
// globals are computed once, for every copy of one container.
class LocalKernelCopier {
public:
  explicit LocalKernelCopier(mlir::ModuleOp container);
  ~LocalKernelCopier();
  // nullopt (and no change) if `getter` does not name a device kernel of the
  // container, or `cloneName` is taken.
  std::optional<LocalKernelCopy>
  copy(cir::FuncOp getter, llvm::StringRef cloneName, llvm::StringRef suffix);
  // Once the caller has repointed `markers` (calls of the original getter)
  // at `copy`'s getter: redirect every other reference to the kernel's
  // naming artifacts on the launch paths to those markers (the name string,
  // the CompileTimeKernelInfo, the getter, and the constants that carry
  // one, such as the `__const.<f>.Info` copy handler::setDeviceKernelInfo
  // stores the handler's kernel name from) to `copy`'s, through copies
  // named `<original><suffix>`; then check that no function of those paths
  // names the original kernel any more. Returns the number of functions
  // that still do (each remarked: kernel-name-redirect).
  unsigned redirectNameReads(const LocalKernelCopy &copy,
                             llvm::ArrayRef<mlir::Operation *> markers,
                             llvm::StringRef suffix);

private:
  mlir::ModuleOp container;
  mlir::SymbolTable hostSyms;
  llvm::SmallVector<std::unique_ptr<mlir::SymbolTable>, 1> deviceSyms;
  llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<mlir::Operation *, 1>>
      holders;
};

// The opt-in remarks of the SYCL host-facts passes: mlir::remark remarks
// (printed only when a remark filter selects them, e.g. cir-opt or
// cir-offload-merge --remarks-filter=cir-offload-noalias), named by the pass
// and filtered by category. `text` is the one-line decision and its reason.
enum class OffloadRemarkKind { Passed, Missed, Analysis };
inline constexpr llvm::StringLiteral kNoAliasRemarkCategory =
    "cir-offload-noalias";
inline constexpr llvm::StringLiteral kKernelArgConstantRemarkCategory =
    "cir-offload-kernel-arg-constant";
inline constexpr llvm::StringLiteral kReqdWorkGroupSizeRemarkCategory =
    "cir-offload-reqd-wg";
inline constexpr llvm::StringLiteral kInternalizeRemarkCategory =
    "cir-offload-internalize";
void emitOffloadRemark(mlir::Location loc, OffloadRemarkKind kind,
                       llvm::StringRef pass, llvm::StringRef category,
                       llvm::StringRef function, const llvm::Twine &text);

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
  // The bindings and launch sites only: no host facts (HostFactsMode::None).
  // What getAnalysis<KernelBindingTable>() builds.
  explicit KernelBindingTable(mlir::ModuleOp container)
      : KernelBindingTable(container, noFacts(), nullptr) {}
  // With the host facts `options` select (see HostFactsMode), from the
  // SYCLHostFacts of `cache` (built once per container and set of analysis
  // options, see SYCLHostFactsCache) or, without a cache, from facts built
  // for this table alone.
  KernelBindingTable(mlir::ModuleOp container, const HostFactsOptions &options,
                     SYCLHostFactsCache *cache = nullptr);

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

  const HostFactsOptions &getHostFactsOptions() const {
    return hostFactsOptions;
  }

private:
  static HostFactsOptions noFacts() {
    HostFactsOptions options;
    options.mode = HostFactsMode::None;
    return options;
  }

  // Adds the SYCL name-string <-> spir_kernel bindings and their eligibility.
  void bindSYCLKernels(mlir::ModuleOp container);

  // Records a launch site on each SYCL binding for every host
  // `getDeviceKernelInfo<Kernel>()` marker that names it. Runs after
  // bindSYCLKernels, so every binding already has its name-string.
  void recordSYCLLaunchSites(mlir::ModuleOp container);

  // Under HostFactsMode::IPO, the noalias verdicts, the scalar-capture
  // constants and the geometry of every SYCL kernel, from SYCLHostFacts
  // (syclNoAliasFields/Notes, syclConstArgs, every launch site's
  // syclNDRange). Runs after the launch sites are recorded.
  void applyHostFacts(mlir::ModuleOp container,
                      const llvm::StringMap<SYCLNoAliasCandidates> &candidates);

  // Keys are StringRefs into the `cu.kernel_name` attribute storage (CUDA) or
  // the device kernel's symbol-name storage (SYCL), so the table's lifetime is
  // tied to the container IR it was built from.
  llvm::MapVector<llvm::StringRef, KernelBinding> bindings;
  // The scope the stubs and launch sites live in, so a visibility question
  // about a binding does not depend on the caller passing the right module.
  mlir::ModuleOp hostModule;
  HostFactsOptions hostFactsOptions;
  SYCLHostFactsCache *hostFactsCache = nullptr;
};

} // namespace cir

#endif // LLVM_CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_KERNELBINDINGTABLE_H
