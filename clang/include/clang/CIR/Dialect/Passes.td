//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_PASSES_TD
#define CLANG_CIR_DIALECT_PASSES_TD

include "mlir/Pass/PassBase.td"

def CIRCanonicalize : Pass<"cir-canonicalize"> {
  let summary = "Performs CIR canonicalization";
  let description = [{
    Perform canonicalizations on CIR and removes some redundant operations.

    This pass performs basic cleanup and canonicalization transformations that
    are not intended to affect CIR-to-source fidelity and high-level code
    analysis passes. Example transformations performed in this pass include
    empty scope cleanup, trivial `try` cleanup, redundant branch cleanup, etc.
    Those more "heavyweight" transformations and those transformations that
    could significantly affect CIR-to-source fidelity are performed in the
    `cir-simplify` pass.
  }];

  let constructor = "mlir::createCIRCanonicalizePass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def CIRSimplify : Pass<"cir-simplify"> {
  let summary = "Performs CIR simplification and code optimization";
  let description = [{
    The pass performs semantics-preserving code simplifications and optimizations
    on CIR while maintaining strict program correctness.

    Unlike the `cir-canonicalize` pass, these transformations may reduce the IR's
    structural similarity to the original source code as a trade-off for improved
    code quality. This can affect debugging fidelity by altering intermediate
    representations of folded expressions, hoisted operations, and other
    optimized constructs.

    Example transformations include ternary expression folding and code hoisting
    while preserving program semantics.
  }];
  let constructor = "mlir::createCIRSimplifyPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def HoistAllocas : Pass<"cir-hoist-allocas"> {
  let summary = "Hoist allocas to the entry of the function";
  let description = [{
    This pass hoist all non-dynamic allocas to the entry of the function.
    This is helpful for later code generation.
  }];
  let constructor = "mlir::createHoistAllocasPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def CIRFlattenCFG : Pass<"cir-flatten-cfg"> {
  let summary = "Produces flatten CFG";
  let description = [{
    This pass transforms CIR by inlining all the nested regions. Thus,
    the following conditions are true after the pass applied:
    - there are no nested regions in any function body
    - all the blocks in a function belong to the parent region
    In other words, this pass removes such CIR operations like IfOp, LoopOp,
    ScopeOp and etc. and produces a flat CIR.
  }];
  let constructor = "mlir::createCIRFlattenCFGPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def GotoSolver : Pass<"cir-goto-solver"> {
  let summary = "Replaces goto operations with branches";
  let description = [{
    This pass transforms CIR and replaces goto-s with branch
    operations to the proper blocks.
  }];
  let constructor = "mlir::createGotoSolverPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def CXXABILowering : Pass<"cir-cxxabi-lowering", "mlir::ModuleOp"> {
  let summary = "Lower CIR according to C++ ABI requirements";
  let description = [{
    This pass lowers CIR operations and types that represent high-level C/C++
    constructs to a more fundamental form according to the target ABI
    requirements.

    See the description of the `TargetLowering` pass for the difference between
    this pass and the `TargetLowering` pass.
  }];
  let constructor = "mlir::createCXXABILoweringPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def TargetLowering : Pass<"cir-target-lowering", "mlir::ModuleOp"> {
  let summary = "Lower CIR to a target-specific form";
  let description = [{
    This pass lowers CIR operations from a target-agnostic form to a
    target-specific form without considering ABI requirements.

    CIR has three passes in its lowering pipeline that transform input CIR
    according to target-specific requirements, scheduled in the pipeline by the
    following order:

      1. The `TargetLowering` pass.
      2. The `CXXABILowering` pass.
      3. The `CallConvLowering` pass.

    The `TargetLowering` pass acts more like a legalization pass. It ensures
    every operation in CIR conforms to the target's constraints. An example
    would be the handling of synchronization scopes of atomic operations. The
    x86 family of targets only support a system-wide synchronization scope, thus
    any atomic operations with a different synchronization scope would be
    transformed to use the system-wide scope in this pass.

    The `CXXABILowering` pass and the `CallConvLowering` pass transform the
    CIR according to the target's ABI requirements. The former handles all
    ABI-related lowering except for calling convention handling, which is
    handled specifically in the latter. Example transformations that the
    `CXXABILowering` pass could make include:

      - Replace C/C++ types that have an ABI-defined layout with more
        fundamental types corresponding to the ABI requirements. For example,
        the layout of the pointer-to-data-member type in C++ is ABI-defined,
        thus the `CXXABILowering` pass would replace all occurrences of this
        type according to the ABI requirements. With a typical target ABI, it
        will be replaced by the `ptrdiff_t` type since this is the type most
        ABIs use for the layout of the pointer-to-data-member type.
      - Replace CIR operations that have ABI-dependent implementations with more
        fundamental operations. For example, the `dynamic_cast` operator in C++
        is implemented in an ABI-dependent way, typically by calling into a
        library function provided by the implementation. The `CXXABILowering`
        pass would thus replace all `cir.dyn_cast` operations with corresponding
        library function calls.

    The `CallConvLowering` pass is dedicated to handle calling conventions. It
    rewrites function signatures according to calling convention requirements,
    and updates function body and call sites accordingly. For example, when
    passing a large struct by value in C/C++, most ABIs require passing a
    pointer to a copy of the struct instead. Thus, the `CallConvLowering` pass
    would rewrite function signatures that take a value of a large struct type
    to take a pointer to the struct type instead, and update the function body
    and the call sites accordingly.
  }];
  let constructor = "mlir::createTargetLoweringPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def CIREHABILowering : Pass<"cir-eh-abi-lowering", "mlir::ModuleOp"> {
  let summary = "Lower flattened CIR EH operations to target-specific ABI form";
  let description = [{
    This pass lowers the ABI-agnostic exception handling operations produced by
    CFG flattening into an ABI-specific form. Currently only the Itanium C++
    ABI is implemented.

    For the Itanium ABI, the pass performs the following transformations:
      - Replaces `cir.eh.initiate` with `cir.eh.inflight_exception`
      - Replaces `cir.eh.dispatch` with `cir.eh.typeid` + comparison chains
      - Removes `cir.begin_cleanup` and `cir.end_cleanup` operations
      - Replaces `cir.begin_catch` with a call to `__cxa_begin_catch`
      - Replaces `cir.end_catch` with a call to `__cxa_end_catch`
      - Replaces `cir.resume` with `cir.resume.flat`
      - Sets the personality function attribute on functions that require EH

    If a non-Itanium ABI is specified, the pass emits a diagnostic indicating
    that the target is not yet implemented.
  }];
  let constructor = "mlir::createCIREHABILoweringPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def LoweringPrepare : Pass<"cir-lowering-prepare", "mlir::ModuleOp"> {
  let summary = "Lower to more fine-grained CIR operations before lowering to "
    "other dialects";
  let description = [{
    This pass does preparation work for lowering to other dialects. For example,
    it may expand the global variable initialziation in a more ABI-friendly form.
  }];
  let constructor = "mlir::createLoweringPreparePass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def IdiomRecognizer : Pass<"cir-idiom-recognizer", "mlir::ModuleOp"> {
  let summary = "Raise calls to C/C++ libraries to CIR operations";
  let description = [{
    This pass recognizes idiomatic C++ usage and captures information about C++
    standard library containers, library functions calls, and types into CIR
    operations, attributes and types.
  }];
  let constructor = "mlir::createIdiomRecognizerPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def LibOpt : Pass<"cir-lib-opt"> {
  let summary = "Optimize C/C++ library calls";
  let description = [{
    This pass applies transformations on C/C++ standard library idioms,
    such as library function calls and structs raised to CIR operations
     using the `cir-idiom-recognize` pass.
  }];

  let constructor = "mlir::createLibOptPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def CallConvLowering : Pass<"cir-call-conv-lowering", "mlir::ModuleOp"> {
  let summary = "Lower CIR function signatures and call sites to match target ABI";
  let description = [{
    This pass rewrites `cir.func` signatures and `cir.call` sites to match the
    target ABI's calling convention requirements (extension attributes, struct
    coercion, sret/byval indirect passing, struct flattening, etc.).

    Requirements on the input IR:

    - All C++-specific CIR types must already be lowered to ABI-relevant
      shapes.  In particular, member pointers and method types must have
      been rewritten by `CXXABILowering` so that the pass sees only the
      final ABI representation of each value.
    - The module must carry a `dlti.dl_spec` attribute so the classifier
      can query type sizes and alignments.

    Two driver modes select how each function's classification is computed:

    - `target=<name>` selects an ABI target.  Currently only `"test"` (the
      MLIR test target in `mlir/lib/ABI/Targets/Test/`) is supported.  Real
      targets (x86_64, AArch64, ...) will be added once the LLVM ABI library
      ships them.
    - `classification-attr=<name>` reads a `DictionaryAttr` named `<name>`
      from each `cir.func` and parses it via the test-target injection
      helper.  Used by tests to inject arbitrary classifications without
      depending on a real ABI target.

    Exactly one of the two options must be set.
  }];
  let constructor = "mlir::createCallConvLoweringPass()";
  let dependentDialects = ["cir::CIRDialect"];
  let options = [
    Option<"target", "target", "cir::CallConvTarget",
           /*default=*/"cir::CallConvTarget::None",
           "Target whose ABI rules drive classification",
           [{::llvm::cl::values(
             clEnumValN(cir::CallConvTarget::Test, "test",
                        "MLIR test ABI target"),
             clEnumValN(cir::CallConvTarget::X86_64, "x86_64",
                        "x86_64 System V"),
             clEnumValN(cir::CallConvTarget::AMDGPU, "amdgpu",
                        "AMDGPU"),
             clEnumValN(cir::CallConvTarget::SPIRV, "spirv",
                        "SPIR / SPIR-V")
           )}]>,
    Option<"classificationAttr", "classification-attr", "std::string",
           /*default=*/"\"\"",
           "Function attribute name carrying a pre-built FunctionClassification "
           "DictionaryAttr (alternative to target=, used by tests)">,
    Option<"x86AvxAbiLevel", "x86-avx-abi-level", "llvm::abi::X86AVXABILevel",
           /*default=*/"llvm::abi::X86AVXABILevel::None",
           "AVX ABI level for the x86_64 target",
           [{::llvm::cl::values(
             clEnumValN(llvm::abi::X86AVXABILevel::None, "none", "No AVX"),
             clEnumValN(llvm::abi::X86AVXABILevel::AVX, "avx", "AVX"),
             clEnumValN(llvm::abi::X86AVXABILevel::AVX512, "avx512", "AVX512")
           )}]>,
    Option<"allowsX86TargetAttrAvx", "x86-target-attr-avx", "bool",
           /*default=*/"true",
           "Let a function's own target features raise its AVX ABI level "
           "above x86-avx-abi-level">,
  ];
}

// The --cir-offload-host-facts options of the container passes that read
// the SYCL host facts (SYCLHostFacts, through the KernelBindingTable they
// share: SYCLHostFactsCache).
defvar OffloadHostFactsBudgetOptions = [
  Option<"hostFactsBudgetScale", "host-facts-budget-scale", "double",
         /*default=*/"1.0",
         "Multiply every budget of the points-to analysis by "
         "this factor (PointsToAnalysisOptions::budgetScale; for tests).">,
  Option<"hostFactsSlice", "host-facts-slice", "bool", /*default=*/"false",
         "Analyse only the functions that can matter to a "
         "kernel launch (PointsToAnalysisOptions::relevanceSlice).">,
  Option<"hostFactsMaxSeconds", "host-facts-max-seconds", "double",
         /*default=*/"120",
         "The wall-clock seconds the points-to analysis may "
         "take; past it the bottom-up analysis falls back to the "
         "context-insensitive one, and that to no facts (0: no limit).">,
  Option<"hostFactsMaxMemoryMB", "host-facts-max-memory-mb", "unsigned",
         /*default=*/"8192",
         "The memory (MB) the points-to analysis may "
         "allocate, with the same fallback (0: no limit).">,
  Option<"hostFactsInternalize", "host-facts-internalize", "bool",
         /*default=*/"false",
         "Without closed-tu: the launch paths were internalized "
         "(cir-offload-internalize-launch-paths). The points-to analysis then "
         "counts local functions whose address only analysed code takes as "
         "having visible callers "
         "(PointsToAnalysisOptions::localFunctionsClosed), and skips the "
         "unused ODR functions (skipUnusedODRFunctions); the remarks of a "
         "kernel other TUs can launch say so.">,
];

defvar OffloadHostFactsOptions = !listconcat(OffloadHostFactsBudgetOptions, [
  Option<"hostFacts", "host-facts", "std::string", /*default=*/"\"ipo\"",
         "Where the host facts come from: ipo (SYCLHostFacts, from the "
         "points-to analysis of the host module; the default) or none (no "
         "facts: the pass does nothing).">,
  Option<"hostFactsClosedTU", "host-facts-closed-tu", "bool",
         /*default=*/"false",
         "Assume the module holds every caller of every function "
         "(PointsToAnalysisOptions::assumeClosedTU). Without it a kernel is "
         "specialized in place only if no other TU can launch it "
         "(KernelBinding::isTUUnique: a TU-unique kernel, or the kernel copy "
         "of an internalized launch path); with it every kernel that is not "
         "registered (KernelBinding::stampsInPlace).">,
  Option<"hostFactsBottomUp", "host-facts-bu", "bool", /*default=*/"true",
         "Run the points-to analysis bottom-up (the "
         "default); false runs it context-insensitively.">,
  Option<"hostFactsSubobjects", "host-facts-subobjects", "bool",
         /*default=*/"true",
         "Give class, union and array members their own "
         "points-to nodes (the default).">,
]);

def OffloadDeadKernelElimination
    : Pass<"cir-offload-dead-kernel-elimination", "mlir::ModuleOp"> {
  let summary = "Remove device kernels that can never be launched";
  let description = [{
    Runs on an offload container module and deletes device kernels that are
    provably unreachable, using the host<->device correspondence from
    `KernelBindingTable`.

    The pass is anchored on a module; it acts only when the module carries the
    `cir.offload.container` unit attribute, leaving any other module alone.

    A kernel is removed when it is not launchable outside this container:

      - a host stub with internal linkage and no launch left (its device
        kernels die with it), or
      - a device kernel with no host stub at all -- a static/anonymous kernel
        never launched in this TU, since an externally-linked kernel always
        emits an external stub -- provided device code does not still
        reference it (dynamic parallelism / address-taken).

    Externally-linked kernels are always preserved: another TU may launch them
    through their external host stub. Scoped to the non-RDC pipeline.

    Dynamic parallelism follows NVIDIA's terminology: kernel A launches kernel B
    from device code, so B has no host stub -- yet B is live and must be kept.
    https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html
  }];
  let constructor = "mlir::createOffloadDeadKernelEliminationPass()";
  let dependentDialects = ["cir::CIRDialect"];
  let statistics = [
    Statistic<"numStubsErased", "num-stubs-erased",
              "Number of dead host stubs removed">,
    Statistic<"numKernelsErased", "num-kernels-erased",
              "Number of unlaunchable device kernels removed">,
    Statistic<"numKernelsPreserved", "num-kernels-preserved",
              "Number of stubless device kernels kept because device code "
              "still references them">,
  ];
}

def OffloadKernelArgConstantPropagation
    : Pass<"cir-offload-kernel-arg-const-prop", "mlir::ModuleOp"> {
  let summary = "Propagate constant launch arguments into device kernels";
  let description = [{
    Runs on an offload container module. When every launch of a kernel passes
    the same constant for a given argument, the uses of that argument inside
    the device kernel are replaced by the constant.

    The pass is anchored on a module; it acts only when the module carries the
    `cir.offload.container` unit attribute, leaving any other module alone.

    Only device code is rewritten. The kernel signature, the host argument
    buffer and the registration of the kernel are all left alone, so the
    argument simply becomes unused on the device side.

    The rewrite requires every launch to be visible. That holds when the host
    stub has internal linkage, so no other TU can call it, and every reference
    to the stub outside its own body is a direct call: a stub whose address
    escapes may be called indirectly with arguments this pass cannot see. The
    stub always references itself, since it passes its own address to
    `cudaLaunchKernel`.

    Requires the launch arguments to already be constants at the call site,
    which for anything but a literal means `mem2reg` and `sccp` have run.
  }];
  let constructor = "mlir::createOffloadKernelArgConstantPropagationPass()";
  let dependentDialects = ["cir::CIRDialect"];
  let statistics = [
    Statistic<"numArgsPropagated", "num-args-propagated",
              "Number of device kernel arguments replaced by a constant">,
    Statistic<"numKernelsRewritten", "num-kernels-rewritten",
              "Number of device kernels with at least one argument replaced">,
    Statistic<"numKernelsSkipped", "num-kernels-skipped",
              "Number of kernels skipped before any argument was examined">,
  ];
}

def OffloadLaunchBoundsPropagation
    : Pass<"cir-offload-launch-bounds-propagation", "mlir::ModuleOp"> {
  let summary = "Infer device kernel launch bounds from host launch sites";
  let description = [{
    Runs over a `cir.offload.container`. When every launch of a kernel is
    visible and uses a compile-time block size, the largest block size any of
    them uses is recorded on the device kernel as `nvvm.maxntid`, the bound the
    user could have written with `__launch_bounds__`.

    The bound lets ptxas budget registers against the real block size instead of
    the 1024-thread ceiling, and lets `NVVMIntrRange` narrow the range metadata
    on `tid` reads.

    Requires every launch to be visible, on the same terms as the kernel
    argument constant propagation pass: the host stub has internal linkage and
    every reference to it outside its own body is a recorded launch. A launch
    this container cannot see may use a larger block, and exceeding `maxntid` is
    a launch failure rather than a slowdown.

    An existing bound is only ever tightened. A kernel whose geometry could not
    be traced, or whose block size is a runtime value at any site, is left
    alone.

    Requires the launch geometry to already be constant at the call site, which
    for anything but a literal means `mem2reg` and `sccp` have run.
  }];
  let constructor = "mlir::createOffloadLaunchBoundsPropagationPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def OffloadStampReqdWorkGroupSize
    : Pass<"cir-offload-stamp-reqd-work-group-size", "mlir::ModuleOp"> {
  let summary = "Stamp reqd_work_group_size from constant nd_range local sizes";
  let description = [{
    Runs over a `cir.offload.container` and, for each SYCL device kernel whose
    launches all use the same compile-time-constant nd_range local work-group
    size, records that size on the kernel as a `cir.reqd_work_group_size`
    attribute (natural {X,Y,Z} order, length == number of dims), using the
    host<->device correspondence of `KernelBindingTable` and the geometry of
    SYCLHostFacts (the stored-value queries of the points-to analysis).

    A kernel is stamped only when no other TU can launch it
    (`KernelBinding::stampsInPlace`: a TU-unique kernel or the kernel copy of
    an internalized launch path) and every launch site agrees on a
    fully-constant, non-empty local size. Plain-`range` launches (no explicit
    local size), runtime local sizes, and disagreeing sites are left
    untouched. An existing `cir.reqd_work_group_size` is never overridden.
    Each decision is a remark of category `cir-offload-reqd-wg`.

    The SYCL-shaped metadata (`!work_group_num_dim` and the reversed/padded
    `!reqd_work_group_size`) is emitted from this attribute later, at LLVM
    lowering.
  }];
  let constructor = "mlir::createOffloadStampReqdWorkGroupSizePass()";
  let dependentDialects = ["cir::CIRDialect"];
  let options = OffloadHostFactsOptions;
  let statistics = [
    Statistic<"numHostFactsBuilt", "host-facts-analyses-built",
              "Points-to analyses built for the host facts (0 when an "
              "earlier container pass built the shared one)">,
    Statistic<"numTablesBuilt", "host-facts-tables-built",
              "Kernel binding tables built with the host facts (0 when an "
              "earlier container pass built the shared one)">
  ];
}

def OffloadInferKernelParamNoAlias
    : Pass<"cir-offload-infer-kernel-param-noalias", "mlir::ModuleOp"> {
  let summary = "Infer llvm.noalias on SYCL kernel params from distinct memory";
  let description = [{
    Runs over a `cir.offload.container`. A SYCL device kernel pointer parameter
    -- a captured USM pointer, or the `ptr` part of a captured accessor -- is
    bound to a closure capture field (`cir.sycl_kernel_param`). When, at every
    visible launch site, that field is backed by memory (a USM allocation, a
    buffer's storage) provably distinct from every other candidate field of
    the same closure, the parameter cannot alias the others and is stamped
    `llvm.noalias`, using the host<->device correspondence of
    `KernelBindingTable` and the host facts of SYCLHostFacts (the points-to
    analysis of the host module).

    A kernel no other TU can launch (`KernelBinding::stampsInPlace`: a
    TU-unique kernel or the kernel copy of an internalized launch path) is
    annotated in place; any other kernel is not annotated (its facts reach
    the kernel copy of an internalized launch path instead). A parameter that
    already carries `llvm.noalias` is left alone, as are an accessor's
    range/offset parameters and its `readonly`. Each candidate parameter gets
    one remark of category `cir-offload-noalias` with the decision and its
    reason. The LLVM `noalias` param attribute is emitted from this arg
    attribute later, at LLVM lowering.
  }];
  let constructor = "mlir::createOffloadInferKernelParamNoAliasPass()";
  let dependentDialects = ["cir::CIRDialect"];
  let options = OffloadHostFactsOptions;
  let statistics = [
    Statistic<"numHostFactsBuilt", "host-facts-analyses-built",
              "Points-to analyses built for the host facts (0 when an "
              "earlier container pass built the shared one)">,
    Statistic<"numTablesBuilt", "host-facts-tables-built",
              "Kernel binding tables built with the host facts (0 when an "
              "earlier container pass built the shared one)">
  ];
}

def OffloadInternalizeLaunchPaths
    : Pass<"cir-offload-internalize-launch-paths", "mlir::ModuleOp"> {
  let summary = "Clone the host launch paths of SYCL kernels from closed roots";
  let description = [{
    Runs over a `cir.offload.container` without `host-facts-closed-tu`
    (T26, plan 2.8). Every host function on a path from a closed root
    (`main`, the dynamic initializers) to a `kernel.info` marker that another
    TU could call (external, linkonce_odr, weak) is cloned as an `internal`
    copy and the path's edges are redirected to the copies
    (LaunchPathInternalization.h: direct calls, the CGF invoker a
    `cgfo.ctor` stores, guarded direct calls for indirect calls,
    std::thread state vtables). The markers of the copies name a new kernel
    `<K>.int.<module id>` with local host artifacts (LocalKernelCopier), so
    the facts the host-facts passes derive from this TU's callers apply to it
    and not to the original kernel, which the originals keep launching. A
    kernel no closed root reaches gets its paths internalized from the open
    roots (T28): the externally visible functions nothing in the TU uses,
    kept in place, whose callers the points-to analysis does not see (their
    formals are Incomplete), so the facts that survive hold for every
    caller. Runs before the passes that read the host facts; the analysis
    they share (SYCLHostFactsCache) is built on the internalized module.
    Remarks of category `cir-offload-internalize`.
  }];
  let constructor = "mlir::createOffloadInternalizeLaunchPathsPass()";
  let dependentDialects = ["cir::CIRDialect"];
  let options = [
    Option<"hostFacts", "host-facts", "std::string", /*default=*/"\"ipo\"",
           "The host-facts mode (ipo, or none: the pass does nothing).">,
    Option<"hostFactsClosedTU", "host-facts-closed-tu", "bool",
           /*default=*/"false",
           "Closed-TU assumption: if set, the pass does nothing.">,
    Option<"maxFunctions", "max-functions", "unsigned", /*default=*/"2000",
           "The most function copies per TU (the budget; a kernel whose "
           "paths would exceed it is not internalized, with a remark).">,
    Option<"maxOps", "max-ops", "uint64_t", /*default=*/"400000",
           "The most ops in the function copies per TU.">,
  ];
  let statistics = [
    Statistic<"numCopies", "function-copies", "Host function copies made">,
    Statistic<"numKernelCopies", "kernel-copies",
              "Kernels specialized for an internalized path">,
    Statistic<"numNameReadViolations", "name-read-violations",
              "Functions of an internalized path that still name the "
              "original kernel">,
  ];
}

def OffloadPruneInternalizedLaunchPaths
    : Pass<"cir-offload-prune-internalized-launch-paths", "mlir::ModuleOp"> {
  let summary = "Undo the internalized launch paths that got no fact";
  let description = [{
    Runs over a `cir.offload.container` after the passes that read the host
    facts. A kernel copy made by `cir-offload-internalize-launch-paths` that
    none of them changed (no noalias, no required work-group size, no
    constant; they tag the kernels they change) is erased with its host
    artifacts, its launches go back to the original kernel, every redirected
    edge into a host copy that no longer leads to a kept kernel copy is
    restored, and the unused copies are erased. The internalization tags are
    then removed.
  }];
  let constructor = "mlir::createOffloadPruneInternalizedLaunchPathsPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def OffloadPropagateSYCLKernelArgs
    : Pass<"cir-offload-propagate-sycl-kernel-args", "mlir::ModuleOp"> {
  let summary = "Propagate constant SYCL scalar captures into device kernels";
  let description = [{
    Runs over a `cir.offload.container`. A SYCL device kernel scalar parameter
    (int/float/bool) is bound to a closure capture field. When, at every visible
    launch site, that field is written the SAME compile-time constant, the uses
    of the parameter inside the device kernel are replaced by the constant,
    using the host<->device correspondence of `KernelBindingTable` and the
    constants of SYCLHostFacts (the stored-value queries of the points-to
    analysis).

    Only device code is rewritten, and the kernel signature/ABI is left alone:
    the host still passes the argument, which simply becomes unused on the
    device side. Only kernels no other TU launches
    (`KernelBinding::stampsInPlace`: a TU-unique kernel or the kernel copy of
    an internalized launch path) are touched, in place. A parameter that is a
    pointer/accessor, whose field has no agreed constant, or whose sites
    disagree, is left untouched. Each decision is a remark of category
    `cir-offload-kernel-arg-constant`.

    The captured value is stored into the closure record and read back inside
    the always_inline operator(), so the branch/arithmetic a captured flag or
    coefficient feeds only folds once the value reaches it: this pass makes the
    closure-field store a constant, and a follow-up device-module mem2reg/SCCP
    (or the device backend's inline + SROA) performs the fold.
  }];
  let constructor = "mlir::createOffloadPropagateSYCLKernelArgsPass()";
  let dependentDialects = ["cir::CIRDialect"];
  let options = OffloadHostFactsOptions;
  let statistics = [
    Statistic<"numHostFactsBuilt", "host-facts-analyses-built",
              "Points-to analyses built for the host facts (0 when an "
              "earlier container pass built the shared one)">,
    Statistic<"numTablesBuilt", "host-facts-tables-built",
              "Kernel binding tables built with the host facts (0 when an "
              "earlier container pass built the shared one)">
  ];
}

def OffloadHostDiagnostics
    : Pass<"cir-offload-host-diagnostics", "mlir::ModuleOp"> {
  let summary = "Warn on SYCL host-side bugs the host facts prove";
  let description = [{
    Runs over a `cir.offload.container` after the passes that read the SYCL
    host facts, on the same points-to analysis of the host module
    (SYCLHostFactsCache; no second analysis), and warns where the facts
    prove a bug at a kernel launch. Each check has a tag, which its warnings
    carry (`[<tag>]`); `checks` selects them: a comma-separated list of
    tags, `all` (the default: every check that is on by default) or `none`. A check warns only on a must fact
    (one known origin of the bad kind, complete, in every calling context
    the analysis covers); when unsure it stays silent. Analysis only: the
    pass never changes the module and preserves every analysis.

    Checks:
      - `sycl-host-ptr-in-kernel`: a pointer the closure of a launch passes
        to a device kernel that dereferences it points to host memory (a
        stack variable or a global), which the device cannot access (SYCL
        2020 sec:usm). The warning is at the kernel's access, with a note at
        the allocation.
      - `sycl-host-heap-ptr-in-kernel` (off by default: not in `all`, runs
        only when named): the same for a host heap allocation (malloc, new,
        std::allocator; not USM), which only a device with
        aspect::usm_system_allocations may access.
      - `sycl-host-deref-device-usm`: a host load, store, copy or libc
        memory access (or a call of a function that accesses its pointer
        argument, when the callee's own accesses cannot be judged) through a
        pointer that must point into a device USM allocation (malloc_device,
        aligned_alloc_device, sycl::malloc with a constant
        usm::alloc::device), which the host cannot access (SYCL 2020
        sec:usm). The SYCL implementation and the host instantiation of
        kernel bodies are exempt. The warning is at the access, with a note
        at the allocation.
      - `sycl-buffer-same-host-ptr`: a buffer(hostData, range) constructed
        over host memory that must overlap the host memory of another one,
        constructed before it in the same straight-line code and not yet
        destroyed (undefined behavior, SYCL 2020 sec:buffer and
        sec:buf-sync-rules). Sub-buffers, copies and reinterpreted buffers
        are exempt. The warning is at the second construction, with a note
        at the first one and at the allocation.
  }];
  let constructor = "mlir::createOffloadHostDiagnosticsPass()";
  let dependentDialects = ["cir::CIRDialect"];
  let options = !listconcat(OffloadHostFactsOptions, [
    Option<"checks", "checks", "std::string", /*default=*/"\"all\"",
           "The checks to run: a comma-separated list of tags, all or "
           "none.">,
  ]);
  let statistics = [
    Statistic<"numWarnings", "num-warnings", "Warnings emitted">,
    Statistic<"numHostFactsBuilt", "host-facts-analyses-built",
              "Points-to analyses built for the host facts (0 when an "
              "earlier container pass built the shared one)">,
  ];
}

def OffloadDetectHostUniformIntDivisor
    : Pass<"cir-offload-detect-host-uniform-int-divisor",
           "mlir::ModuleOp"> {
  let summary = "Remark on device compute driven by a launch-uniform capture";
  let description = [{
    Runs over a `cir.offload.container`. For each SYCL device kernel, walks the
    kernel body for expensive computations whose operands are
    *kernel-scope-uniform*: invariant across the launch because they are derived
    only from captured scalar closure fields (classified with the same
    param->field mapping the scalar-capture propagation pass uses) and
    `cir.const`, propagated through pure arithmetic and pure (`const`) library
    calls. A chain that reaches a work-item/index builtin (`__spirv_BuiltIn*`),
    a non-captured memory load, or a parameter not backed by a captured scalar
    is NOT uniform.

    Two remark kinds, both diagnostic (the strength reduction / hoist the front
    end cannot see -- it lacks the whole-launch view -- and the back end cannot
    see -- the value is a runtime kernel argument on the device side):

    - integer/floating-point division or remainder whose DIVISOR is uniform (the
      classic `lid / C_`): suggest host-precomputing a reciprocal magic-multiply
      whose constants are a function of the divisor. Skips a constant divisor --
      the backend already strength-reduces those.
    - an expensive fully-uniform computation (a transcendental math call such as
      `cos`/`sin`/`exp`/`pow`/`sqrt`, or a division with both operands uniform):
      the whole result is a launch constant, so suggest host-precomputing it and
      passing the scalar. Reported once, at the outermost expensive op of a
      uniform expression, and only when a runtime capture is actually involved.

    Every SYCL device kernel is a candidate, TU-local or not: a remark is sound
    cross-TU (unlike the in-place-edit opts), and the motivating case is a named
    functor. Diagnostic only: the pass never mutates the module and is not part
    of the default merge pipeline, so it must be requested explicitly
    (CMPLRLLVM-76652).
  }];
  let constructor = "mlir::createOffloadDetectHostUniformIntDivisorPass()";
  let dependentDialects = ["cir::CIRDialect"];
  let options = [
    Option<"scanAllKernels", "scan-all-kernels", "bool", /*default=*/"false",
           "Scan every device spir_kernel directly, bypassing the "
           "host<->device binding table. For surveying a device-only module "
           "(no host side); the default path scans only kernels reached "
           "through a recovered SYCL launch binding.">,
  ];
}

def OffloadDetectHostDeviceClosureLayoutMismatch
    : Pass<"cir-offload-detect-host-device-closure-layout-mismatch",
           "mlir::ModuleOp"> {
  let summary =
      "Warn when a SYCL kernel's captured-closure layout diverges host vs device";
  let description = [{
    Runs over a `cir.offload.container`. A SYCL kernel lambda is compiled twice
    -- once for the host and once for the SPIR-V device -- so preprocessor
    divergence (`#ifdef __SYCL_DEVICE_ONLY__`) or divergent `[=]` implicit
    captures can make the two closure records disagree in a field's type/layout
    while keeping the same total size. The `sizeof(KernelType) == KernelSize`
    static_assert in `sycl/queue.hpp` only checks total size, so an equal-size /
    different-type divergence compiles clean and produces silent wrong results at
    runtime. Only the joined host+device container sees both closure layouts, so
    only a container pass can catch it.

    For each SYCL device `spir_kernel` this pass recovers, through the
    KernelBindingTable, the host closure record (from the launch site's
    HostKernelRef closure operand) and the device closure record (the record the
    kernel prologue reconstructs). Matched by kernel binding -- never by
    type-name equality, since the two records carry different aliases (`anon.0`
    vs `__cir_lowered.anon.0`) -- it compares the ordered field lists (ignoring
    a captured pointer's address space, a device-lowering artifact) and emits an
    MLIR *warning* for each divergence (a differing field count, or a field whose
    type differs), anchored at the capture's source line and naming the field.
    Such a mismatch is a host/device closure mismatch (likely
    `__SYCL_DEVICE_ONLY__` or divergent captures) and is undefined behavior.

    Analysis only: the pass never mutates the module. It runs at the head of the
    cir-offload-merge combine pipeline, before the mem2reg that would promote the
    device closure alloca the device-record recovery reads.
  }];
  let constructor =
      "mlir::createOffloadDetectHostDeviceClosureLayoutMismatchPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

#endif // CLANG_CIR_DIALECT_PASSES_TD
