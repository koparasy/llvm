//===- LaunchPathInternalization.h - Clone host paths from roots -*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host-path internalization (T26, plan 2.8): make every function on a path
// from a closed root (`main`, and the dynamic initializers and destructors of
// namespace-scope variables) to a set of sink functions have only callers in
// this TU, by cloning the ones another TU could call (external, linkonce_odr
// or weak linkage) as `internal` copies and redirecting the edges of the path
// to the copies. The originals stay, for the other TUs that may call them.
//
// Why it is sound. An inline function has the same body in every TU (C++20
// [basic.def.odr]p14), so a copy behaves like the original; a call to the copy
// is a call to the original. The transformation does not change what the
// program does: every redirected edge reaches an equivalent function. What
// changes is what an analysis of this TU may assume: a copy has local linkage
// and its only users are functions of the path, so its callers are all here
// (PointsToAnalysisOptions::localFunctionsClosed).
//
// Clean functions. A root is clean; a clone is clean (only clean functions
// use it); a function with local linkage is clean iff every use of its symbol
// is in a clean function, or in the initializer of a local global whose uses
// are (greatest fixpoint: a local cycle reached only from roots is clean). A
// function another TU can call is never clean. A function of a path is kept
// in place if it is clean (its edges into copies are redirected in place: a
// call of C is a call of C', so that is semantics-preserving), and cloned
// otherwise. Clean-ness only decides in place or clone; soundness does not
// depend on it.
//
// Edges of a path, and how each is redirected (P holds the edge):
//  - a direct call: the callee becomes the copy;
//  - an address-of (cir.get_global of a function) the client vouches for
//    (InternalizationOptions::vouchAddress: the address is only called, never
//    compared, by code the client knows, e.g. the SYCL runtime calling the
//    CGF invoker a cgfo.ctor stores): the global becomes the copy. Any other
//    address-of is not followed: the function's address could be compared
//    ([dcl.inline]p6 promises one address in all TUs) or called by code we
//    don't see with anything;
//  - an indirect call in P (a virtual call, a function pointer call): a
//    guarded direct call `fn == &G ? G'(args) : fn(args)` per candidate G of
//    the path (speculative devirtualization). Nothing observable changes:
//    the vtable and the pointer are untouched, G' never has its address
//    taken. Candidates: for `vtable.get_virtual_fn_addr[i]`, slot i of every
//    vtable address point of the module, with the call's arity and
//    pointer-castable operands (an override takes its own `this`); for any
//    other pointer, every address-taken function of exactly the pointee type
//    (a call through a pointer to another function type is undefined,
//    C++20 [expr.call]p6), so that no guard adds a direct call the program
//    cannot make. A call of a function's own address (after mem2reg/SCCP,
//    a local pointer that only ever holds one function) becomes a direct
//    call of the copy, without a guard;
//  - std::thread (the T27 model: _M_start_thread runs `_M_run` of the state
//    on another thread, from inside libstdc++): P calls the
//    std::thread::_State_impl constructor C, whose vtable V holds `_M_run`.
//    The vtable is cloned (V', internal, `_M_run` -> its copy) with a copy C'
//    of the constructor storing V'; P calls C'. Only the vptr of a state
//    object changes: it is made by std::thread's constructor and only passed
//    to _M_start_thread, which calls _M_run and deletes it; no user code can
//    name _State_impl, so nothing compares its vptr.
//
// Every copy and every redirected op is tagged (the attribute names below)
// so that pruneInternalizedPaths can undo the paths a client did not use.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_LAUNCHPATHINTERNALIZATION_H
#define CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_LAUNCHPATHINTERNALIZATION_H

#include "mlir/IR/BuiltinOps.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <functional>
#include <string>

namespace cir {

/// On a copy (function or global): the name of the original (a string, not a
/// symbol reference, so that it is not a use of the original).
inline constexpr llvm::StringLiteral kInternalizedAttr =
    "cir.offload.internalized";
/// On an op of a clean function whose callee or global was redirected to a
/// copy: the name of the original target.
inline constexpr llvm::StringLiteral kInternalizedEdgeAttr =
    "cir.offload.internalized_edge";
/// On the cir.if / cir.ternary of a guarded direct call: the name of the
/// original candidate whose copy the true branch calls.
inline constexpr llvm::StringLiteral kInternalizedGuardAttr =
    "cir.offload.internalized_guard";
/// On an op of a launch path whose reference to a kernel's naming artifact
/// (its name string, CompileTimeKernelInfo, kernel.info getter, or a constant
/// that carries one) was redirected to a kernel copy's: a dictionary
/// {getter = "<the kernel copy's getter>", attrs = <the op's attributes
/// before>}, so that the reference is restored with the copy's markers.
inline constexpr llvm::StringLiteral kInternalizedNameReadAttr =
    "cir.offload.internalized_name_read";

struct InternalizationOptions {
  /// Whether the function address \p addr (in \p holder) may be redirected to
  /// a copy: only code the client vouches for uses it, and only to call it.
  /// Null: none.
  std::function<bool(cir::GetGlobalOp addr, cir::FuncOp holder)> vouchAddress;
  /// Whether the closed roots (isClosedRoot) are roots.
  bool closedRoots = true;
  /// More roots than the closed ones, kept in place like them. Null: none.
  /// Only for a client whose facts about a path do not depend on the root's
  /// callers: the SYCL host facts' open roots (externally visible functions
  /// nothing in the module uses, whose formals the points-to analysis treats
  /// as Incomplete; see InternalizeLaunchPaths.cpp).
  std::function<bool(cir::FuncOp)> extraRoot;
  /// Clone the sinks too (otherwise they are the ends of the paths, kept as
  /// they are; the client redirects the calls to them).
  bool cloneSinks = false;
  /// Guard the indirect calls of the paths (see the file comment).
  bool guardIndirectCalls = true;
  /// Redirect std::thread state vtables (see the file comment).
  bool redirectThreadStates = true;
  /// The budget: the most functions (copies, including the constructor
  /// copies of the vtable redirect) and ops in them. Sinks are planned in
  /// order; one whose paths would exceed the budget is skipped.
  unsigned maxFunctions = 2000;
  uint64_t maxOps = 400000;
  /// The suffix of a copy's name.
  std::string suffix = ".int";
};

struct InternalizationResult {
  struct Sink {
    cir::FuncOp sink;
    /// The roots it is reached from, in module order.
    llvm::SmallVector<std::string, 1> roots;
    /// The direct calls of the sink in clean functions or copies (the ops
    /// of the copies): the calls the client may redirect.
    llvm::SmallVector<mlir::Operation *, 2> calls;
    /// Whether one of the calls is in a copy (otherwise every function of
    /// its paths was clean already: nothing was cloned for it).
    bool throughClone = false;
  };
  llvm::SmallVector<Sink> sinks;
  /// The sinks the budget skipped, and why.
  llvm::SmallVector<cir::FuncOp> skipped;
  std::string budgetWhy;
  /// The sinks no path reaches, each with why (no closed root reaches it,
  /// or only through an edge that cannot be redirected).
  llvm::SmallVector<std::pair<cir::FuncOp, std::string>> unreached;
  /// The copies made, original -> copy.
  llvm::DenseMap<mlir::Operation *, cir::FuncOp> copies;
  unsigned numCloned = 0, numInPlace = 0, numGuards = 0, numVTables = 0,
           numRedirected = 0;
  uint64_t numOps = 0;
  /// Edges of a path that were not followed (an unvouched address-of, an
  /// indirect call with no candidate), by kind: the reason a path stops.
  unsigned numUnvouchedAddresses = 0;
};

/// Internalize the paths from the closed roots of \p module to \p sinks.
InternalizationResult
internalizeLaunchPaths(mlir::ModuleOp module, llvm::ArrayRef<cir::FuncOp> sinks,
                       const InternalizationOptions &options);

/// Whether \p func is a closed root: `main`, or a dynamic initializer or
/// destructor of a namespace-scope variable (global_ctor / global_dtor).
bool isClosedRoot(cir::FuncOp func);

/// Undo the paths that do not lead to a call of a function \p keep accepts:
/// restore every redirected edge (direct call, address-of, guarded call,
/// vtable constructor) whose copy does not lead to such a call, then erase
/// the copies and copied globals nothing uses. Returns the number of
/// functions erased.
unsigned pruneInternalizedPaths(mlir::ModuleOp module,
                                llvm::function_ref<bool(cir::FuncOp)> keep);

/// Remove the internalization tags from every op of \p module.
void stripInternalizationTags(mlir::ModuleOp module);

} // namespace cir

#endif // CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_LAUNCHPATHINTERNALIZATION_H
