//===- DetectHostUniformIntDivisor.cpp - launch-uniform int-div remark ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Remarks on a SYCL device computation driven by a kernel-scope-uniform value:
// one that is invariant across the whole launch because it is derived only from
// captured SCALAR closure fields and compile-time constants, through pure
// arithmetic and pure (`const`) library calls. The motivating case
// (CMPLRLLVM-76652) is a pytorch kernel `c = lid / C_`, where
// `lid = item.get_local_id(0)` varies per work-item but `C_` is a captured
// scalar kernel argument -- the same value at every work-item of the launch.
//
// Two remark kinds:
//   - a division / remainder (integer or floating-point) whose DIVISOR is
//     uniform: strength-reduce to a reciprocal magic-multiply (integer) or a
//     reciprocal multiply (fp, under fast-math) whose constants are a function
//     of the divisor, precomputed once on the host.
//   - an expensive FULLY-uniform computation -- a transcendental math call
//     (cos/sin/exp/pow/sqrt/...) or a division with both operands uniform --
//     whose entire result is a launch constant: host-precompute it and pass the
//     scalar. Reported once, at the outermost expensive op of a uniform
//     expression (a nested `a/b` under a `sqrt` is subsumed by the sqrt).
//
// This is analysis + remark ONLY. It never mutates the module, adds no kernel
// argument, and does not rewrite anything (the auto-transform is Flavor B,
// deliberately out of scope). The point is to demonstrate the combine's unique
// host+device visibility: the front end cannot see the value is uniform across
// the launch (it sees one work-item's control flow), and the back end cannot
// see it is a runtime value known on the host (it sees an opaque kernel
// argument). Only the joined host+device container sees both.
//
// The uniformity oracle mirrors the scalar-capture classification of
// PropagateSYCLKernelArgs.cpp: a spir_kernel scalar parameter that the prologue
// stores into a closure capture field (arg -> [spill slot ->] cir.get_member
// store) is a captured scalar, and hence uniform. From there a value is uniform
// iff every step back is such a captured field, a cir.const, a pure
// (memory-effect-free) op over uniform operands, or a `const` (readnone) call
// whose callee is NOT a work-item/index builtin (`__spirv_BuiltIn*`) and whose
// arguments are uniform -- which admits transcendental math (lowered to
// `__spirv_ocl_*` const calls) while rejecting the invocation-state builtins. A
// chain reaching such a builtin, a non-captured memory load, or a parameter
// with no captured-scalar field is NOT uniform. A constant divisor is skipped
// (the backend already strength-reduces those); a fully-uniform expression with
// no runtime capture is skipped (the backend already constant-folds it).
//
// A SYCL device kernel calls its functor's `operator()` (an always_inline
// helper) rather than containing the computation directly, so this pass expects
// the CIR inliner to have folded operator() into the kernel first; run it after
// `-inline` (the lit fixtures model both the pre- and post-inline shapes). It
// is not part of the default cir-offload-merge pipeline; request it explicitly.
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
#define GEN_PASS_DECL_OFFLOADDETECTHOSTUNIFORMINTDIVISOR
#define GEN_PASS_DEF_OFFLOADDETECTHOSTUNIFORMINTDIVISOR
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace cir;

// Whether `call` reads and writes no memory (`memory_effects` is `none`), i.e.
// the callee is a pure function of its arguments.
static bool isReadNoneCall(cir::CallOp call) {
  std::optional<cir::MemoryEffectsAttr> effects = call.getMemoryEffects();
  return effects && *effects == cir::MemoryEffectsAttr::none(call.getContext());
}

namespace {

// Peel value-preserving pointer reinterprets (address-space / bitcast casts) so
// two references to the same closure record compare equal regardless of the
// address-space cast the kernel inserts before handing the closure to
// operator().
static mlir::Value peelPtrCasts(mlir::Value v) {
  while (auto cast = v.getDefiningOp<cir::CastOp>()) {
    if (cast.getKind() != cir::CastKind::address_space &&
        cast.getKind() != cir::CastKind::bitcast)
      break;
    v = cast.getSrc();
  }
  return v;
}

// The single closure capture field a spir_kernel SCALAR parameter is stored
// into, together with the closure record it is a member of, or an empty field
// if the parameter reaches no field or more than one. Shares the forward-walk
// shape with PropagateSYCLKernelArgs.cpp's mapper (arg -> [spill slot ->]
// cir.get_member store), and additionally reports the get_member base so a
// later divisor load can be matched to the very closure the parameter feeds.
struct CapturedField {
  llvm::StringRef name;
  mlir::Value closure; // the record the field lives in (casts peeled)
};

static CapturedField closureFieldForArg(mlir::Value arg) {
  CapturedField result;
  llvm::SmallVector<mlir::Value, 4> worklist{arg};
  llvm::SmallPtrSet<mlir::Value, 8> seen{arg};
  while (!worklist.empty()) {
    mlir::Value v = worklist.pop_back_val();
    for (mlir::Operation *user : v.getUsers()) {
      if (auto st = mlir::dyn_cast<cir::StoreOp>(user)) {
        if (st.getValue() != v)
          continue;
        if (auto gm = st.getAddr().getDefiningOp<cir::GetMemberOp>()) {
          if (!result.name.empty() && result.name != gm.getName())
            return {}; // Two distinct fields: ambiguous, give up.
          result.name = gm.getName();
          result.closure = peelPtrCasts(gm.getAddr());
          continue;
        }
        // A spill into a stack slot: follow the reloads of that slot.
        for (mlir::Operation *slotUser : st.getAddr().getUsers())
          if (auto load = mlir::dyn_cast<cir::LoadOp>(slotUser))
            if (seen.insert(load.getResult()).second)
              worklist.push_back(load.getResult());
        continue;
      }
      if (auto cast = mlir::dyn_cast<cir::CastOp>(user)) {
        if ((cast.getKind() == cir::CastKind::integral ||
             cast.getKind() == cir::CastKind::bitcast) &&
            seen.insert(cast.getResult()).second)
          worklist.push_back(cast.getResult());
      }
    }
  }
  return result;
}

// Uniformity oracle over the def-use chain of a divisor. `fields` maps each
// captured-scalar field name to the closure record it belongs to; `outField`
// receives the first captured field the chain reaches, to name in the remark.
static bool isUniform(mlir::Value v, const llvm::StringMap<mlir::Value> &fields,
                      llvm::SmallPtrSetImpl<mlir::Value> &visited,
                      llvm::StringRef &outField, unsigned depth) {
  // A value's uniformity is a fixed property; a value re-reached on a second
  // operand path was already accepted (a rejected one short-circuits the whole
  // divisor), so treat a revisit as uniform. The depth cap guards pathological
  // chains.
  if (depth > 64)
    return false;
  if (!visited.insert(v).second)
    return true;

  // A kernel parameter is uniform iff it is a captured scalar (it maps to a
  // closure field). Any other block argument -- a non-captured parameter, or a
  // loop/region-carried value -- is conservatively non-uniform.
  if (mlir::isa<mlir::BlockArgument>(v)) {
    CapturedField cf = closureFieldForArg(v);
    if (cf.name.empty() || !fields.count(cf.name))
      return false;
    if (outField.empty())
      outField = cf.name;
    return true;
  }

  mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return false;

  // A load is uniform only when it reads a captured-scalar field of the closure
  // that parameter stores populated (the divisor `C_` in operator() after
  // inlining). Any other load -- a USM buffer, an accessor -- is memory and so
  // non-uniform.
  if (auto load = mlir::dyn_cast<cir::LoadOp>(def)) {
    auto gm = load.getAddr().getDefiningOp<cir::GetMemberOp>();
    if (!gm)
      return false;
    auto it = fields.find(gm.getName());
    if (it == fields.end() || peelPtrCasts(gm.getAddr()) != it->second)
      return false;
    if (outField.empty())
      outField = gm.getName();
    return true;
  }

  // A call is uniform iff it is a pure (`const`, i.e. reads/writes no memory)
  // function whose callee is NOT an invocation-state builtin, and its data
  // arguments are all uniform. That admits transcendental math (cos/sin/exp/...
  // lower to `__spirv_ocl_*` const calls) and other pure library calls, while
  // rejecting the work-item/index builtins (`__spirv_BuiltIn*`) -- which are
  // also `const` but read per-work-item invocation state, so their result
  // varies across the launch.
  if (auto call = mlir::dyn_cast<cir::CallOp>(def)) {
    if (!isReadNoneCall(call))
      return false;
    std::optional<llvm::StringRef> callee = call.getCallee();
    if (!callee || callee->contains("__spirv_BuiltIn"))
      return false;
    for (mlir::Value arg : call.getArgOperands())
      if (!isUniform(arg, fields, visited, outField, depth + 1))
        return false;
    return true;
  }

  // A pure (memory-effect-free) op is a function of its operands only, so it is
  // uniform iff they all are. This covers cir.const (no operands -> uniform),
  // integral/bool casts, and integer arithmetic (add/sub/mul/shift/and/or/xor/
  // select/...). An impure or unknown op is conservatively non-uniform.
  if (!mlir::isMemoryEffectFree(def))
    return false;
  for (mlir::Value operand : def->getOperands())
    if (!isUniform(operand, fields, visited, outField, depth + 1))
      return false;
  return true;
}

// The expensive transcendental math builtins whose result, when the argument
// is launch-uniform, is a launch-invariant constant worth host-precomputing.
// Cheap library calls (fabs/floor/fmin/...) are intentionally excluded: their
// device cost does not justify a hint. Returns the human name (e.g. "cos") of
// the matched builtin, or empty if `call` is not one.
static llvm::StringRef expensiveMathName(cir::CallOp call) {
  if (!isReadNoneCall(call))
    return {};
  std::optional<llvm::StringRef> callee = call.getCallee();
  if (!callee)
    return {};
  // OpenCL extended-instruction math lowers to `__spirv_ocl_<name><typesuffix>`
  // under an Itanium `_Z<len>` prefix. Match the bare name in the middle.
  static constexpr llvm::StringRef kExpensive[] = {
      "sin",  "cos",   "tan",  "asin", "acos",   "atan",   "atan2",
      "sinh", "cosh",  "tanh", "exp",  "exp2",   "exp10",  "log",
      "log2", "log10", "pow",  "powr", "pown",   "sqrt",   "rsqrt",
      "cbrt", "hypot", "erf",  "erfc", "tgamma", "lgamma", "divide"};
  llvm::StringRef marker = "__spirv_ocl_";
  size_t pos = callee->find(marker);
  if (pos == llvm::StringRef::npos)
    return {};
  llvm::StringRef rest = callee->substr(pos + marker.size());
  for (llvm::StringRef name : kExpensive) {
    // The mangled tail is `<name>` + a type suffix (f/d/Dh/...); require the
    // builtin name to be a prefix and the next char to not extend it into a
    // different identifier (e.g. do not match "cos" inside "cosh").
    if (rest.starts_with(name)) {
      char after = rest.size() > name.size() ? rest[name.size()] : '\0';
      if (!llvm::isAlpha(after) || after == 'f' || after == 'd')
        return name;
    }
  }
  return {};
}

// An op whose device cost is high enough to warrant a host-precompute hint:
// integer/floating-point division or remainder, or a transcendental math call.
static bool isExpensiveOp(mlir::Operation *op) {
  if (mlir::isa<cir::DivOp, cir::RemOp, cir::FDivOp, cir::FRemOp>(op))
    return true;
  if (auto call = mlir::dyn_cast<cir::CallOp>(op))
    return !expensiveMathName(call).empty();
  return false;
}

// Whether `op`'s (uniform) result flows, through pure uniform arithmetic, into
// another expensive uniform op. If so, that outer op is the better report site
// (host-precompute the whole thing), so this inner one is suppressed. This
// collapses `sqrt(a / b)` to a single hint on the sqrt while still reporting a
// lone `a / b` that only feeds cheap uniform arithmetic before reaching varying
// code.
static bool
subsumedByExpensiveUniform(mlir::Operation *op,
                           const llvm::StringMap<mlir::Value> &fields,
                           llvm::SmallPtrSetImpl<mlir::Operation *> &seen) {
  for (mlir::Operation *user : op->getUsers()) {
    if (!seen.insert(user).second)
      continue;
    if (user->getNumResults() != 1)
      continue;
    llvm::SmallPtrSet<mlir::Value, 16> visited;
    llvm::StringRef field;
    if (!isUniform(user->getResult(0), fields, visited, field, 0))
      continue; // varying consumer: not a subsuming uniform expression.
    if (isExpensiveOp(user))
      return true;
    if (subsumedByExpensiveUniform(user, fields, seen))
      return true;
  }
  return false;
}

struct OffloadDetectHostUniformIntDivisorPass
    : public impl::OffloadDetectHostUniformIntDivisorBase<
          OffloadDetectHostUniformIntDivisorPass> {
  void runOnOperation() override;
};

void OffloadDetectHostUniformIntDivisorPass::runOnOperation() {
  mlir::ModuleOp container = getOperation();
  if (!cir::isOffloadContainer(container))
    return;

  // Analysis only: nothing below mutates the IR.
  markAllAnalysesPreserved();

  auto scanKernel = [&](cir::FuncOp kernel) {
    if (kernel.getCallingConv() != cir::CallingConv::SpirKernel)
      return;
    if (kernel.getBody().empty())
      return;

    // Captured-scalar fields of this kernel's closure: each scalar parameter
    // the prologue stores into a closure member. The uniformity oracle treats
    // exactly these fields (and constants, through pure arithmetic) as uniform.
    llvm::StringMap<mlir::Value> fields;
    for (mlir::BlockArgument arg : kernel.getBody().front().getArguments()) {
      if (!mlir::isa<cir::IntType, cir::FPTypeInterface, cir::BoolType>(
              arg.getType()))
        continue;
      CapturedField cf = closureFieldForArg(arg);
      if (!cf.name.empty() && cf.closure)
        fields.try_emplace(cf.name, cf.closure);
    }
    if (fields.empty())
      return;

    // A whole-value-uniform expensive computation is worth reporting only where
    // it crosses into varying code, and only when it actually derives from a
    // runtime capture (a purely constant one is folded by the backend). Reports
    // once, at the boundary, naming the driving capture.
    auto reportFullyUniform = [&](mlir::Operation *op, llvm::StringRef what) {
      llvm::SmallPtrSet<mlir::Operation *, 16> seen;
      if (subsumedByExpensiveUniform(op, fields, seen))
        return; // a larger expensive uniform op is the better report site.
      llvm::SmallPtrSet<mlir::Value, 16> visited;
      llvm::StringRef field;
      for (mlir::Value operand : op->getOperands())
        if (isUniform(operand, fields, visited, field, 0) && !field.empty())
          break;
      if (field.empty())
        return; // no runtime capture involved -> a compile-time constant.
      op->emitRemark()
          << "device kernel '" << kernel.getName() << "': launch-uniform "
          << what << " of captured scalar '" << field
          << "'; result is invariant across the launch -- host-precompute it "
             "and pass the scalar (CMPLRLLVM-76652)";
    };

    kernel.getBody().walk([&](mlir::Operation *op) {
      // Division / remainder (integer and floating-point).
      if (mlir::isa<cir::DivOp, cir::RemOp, cir::FDivOp, cir::FRemOp>(op)) {
        bool isRem = mlir::isa<cir::RemOp, cir::FRemOp>(op);
        bool isFP = mlir::isa<cir::FDivOp, cir::FRemOp>(op);
        mlir::Value dividend = op->getOperand(0);
        mlir::Value divisor = op->getOperand(1);
        // A compile-time-constant divisor is already strength-reduced by the
        // backend; the interesting case is a runtime, launch-uniform value.
        if (divisor.getDefiningOp<cir::ConstantOp>())
          return;

        llvm::SmallPtrSet<mlir::Value, 16> dv, nv;
        llvm::StringRef dField, nField;
        bool divisorUniform = isUniform(divisor, fields, dv, dField, 0);

        // Whole quotient uniform (both operands uniform): host-precompute it.
        if (divisorUniform && isUniform(dividend, fields, nv, nField, 0)) {
          reportFullyUniform(op, isFP ? "floating-point division"
                                      : "integer division");
          return;
        }
        // Only the divisor is uniform (the classic `lid / C_`): strength-reduce
        // to a reciprocal magic-multiply whose constants are a function of the
        // divisor, precomputed on the host.
        if (divisorUniform && !dField.empty()) {
          op->emitRemark() << "device kernel '" << kernel.getName()
                           << "': " << (isFP ? "floating-point " : "integer ")
                           << (isRem ? "remainder" : "division")
                           << " by launch-uniform capture '" << dField
                           << "'; consider host-precomputing a reciprocal "
                           << (isFP ? "multiply (under fast-math)"
                                    : "magic-multiply")
                           << " (CMPLRLLVM-76652)";
        }
        return;
      }

      // Expensive transcendental math call (cos/sin/exp/pow/sqrt/...) whose
      // argument is launch-uniform: the whole result is a launch constant.
      if (auto call = mlir::dyn_cast<cir::CallOp>(op)) {
        llvm::StringRef name = expensiveMathName(call);
        if (name.empty())
          return;
        llvm::SmallPtrSet<mlir::Value, 16> visited;
        llvm::StringRef field;
        for (mlir::Value arg : call.getArgOperands())
          if (!isUniform(arg, fields, visited, field, 0))
            return;
        reportFullyUniform(op, name);
      }
    });
  };

  // Survey mode: walk every device-side spir_kernel directly, without a host
  // binding. Used to scan a device-only module for the pattern's frequency.
  if (scanAllKernels) {
    for (mlir::ModuleOp mod : cir::getOffloadContainerDeviceModules(container))
      for (cir::FuncOp kernel : mod.getOps<cir::FuncOp>())
        scanKernel(kernel);
    return;
  }

  cir::KernelBindingTable &table = getAnalysis<cir::KernelBindingTable>();
  for (const auto &entry : table) {
    const cir::KernelBinding &binding = entry.second;
    // Every SYCL device kernel is a candidate, not only TU-local/optimizable
    // ones. The transform-soundness gate the sibling opts use (in-place edits
    // are only safe when every launch is visible) does not apply to a remark:
    // the host may precompute the reciprocal of a launch-uniform divisor
    // regardless of whether other TUs also launch the kernel -- and the
    // motivating case (CMPLRLLVM-76652) is a named functor, which is not
    // TU-local. The divisor's uniformity is a property of the kernel body and
    // its captured scalars alone.
    if (binding.kind != cir::BindingKind::SYCLName)
      continue;
    for (cir::FuncOp kernel : binding.deviceKernels)
      scanKernel(kernel);
  }
}

} // namespace

std::unique_ptr<Pass> mlir::createOffloadDetectHostUniformIntDivisorPass() {
  return std::make_unique<OffloadDetectHostUniformIntDivisorPass>();
}
