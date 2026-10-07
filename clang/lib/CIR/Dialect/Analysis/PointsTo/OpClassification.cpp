//===- OpClassification.cpp - What points-to does with each CIR op --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The classification table. To classify a new CIR op, add it to the one
// section that fits, keeping the section (or family) sorted:
//  - Handled only if LocalBuilder has a rule for it (an op-specific case or a
//    control-flow interface it implements);
//  - NoPointerSemantics only if the op can neither create, move, store nor
//    publish a pointer. Say why in the family comment;
//  - Pessimistic otherwise, with a TODO naming the task that will model it.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/PointsTo/OpClassification.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

using namespace cir::pointsto;
using llvm::StringRef;

static constexpr OpClass kHandled = OpClass::Handled;
static constexpr OpClass kNoPtr = OpClass::NoPointerSemantics;
static constexpr OpClass kPessimistic = OpClass::Pessimistic;

static const OpClassEntry kOpClassificationTable[] = {
    //===------------------------------------------------------------------===//
    // Handled: LocalBuilder has a rule.
    //===------------------------------------------------------------------===//
    {"cir.alloca", kHandled},              // Stack node.
    {"cir.atomic.clear", kHandled},        // Mod|Ref on an integer.
    {"cir.atomic.cmpxchg", kHandled},      // Load + store.
    {"cir.atomic.fetch", kHandled},        // Load + store.
    {"cir.atomic.test_and_set", kHandled}, // Mod|Ref on an integer.
    {"cir.atomic.xchg", kHandled},         // Load + store.
    {"cir.await", kHandled},               // RegionBranchOpInterface.
    {"cir.base_class_addr", kHandled},     // Constant offset.
    {"cir.br", kHandled},                  // BranchOpInterface.
    {"cir.brcond", kHandled},              // BranchOpInterface.
    {"cir.call", kHandled},                // CallSiteRecord.
    {"cir.call_llvm_intrinsic", kHandled}, // Allow-list, else pessimistic.
    {"cir.case", kHandled},                // RegionBranchOpInterface.
    {"cir.cast", kHandled},          // Per cast kind; the rest is pessimistic.
    {"cir.cleanup.scope", kHandled}, // RegionBranchOpInterface.
    {"cir.clear_padding", kHandled}, // Mod; writes zero bytes.
    {"cir.co_return", kHandled},     // RegionBranchTerminatorOpInterface.
    {"cir.complex.imag_ptr", kHandled}, // Constant offset.
    {"cir.complex.real_ptr", kHandled}, // Constant offset.
    {"cir.condition", kHandled},        // RegionBranchTerminatorOpInterface.
    {"cir.const", kHandled}, // Null: no cell; the rest is pessimistic.
    {"cir.coro.suspend_point", kHandled}, // RegionBranchTerminatorOpInterface.
    {"cir.coroutine", kHandled},          // RegionBranchOpInterface.
    {"cir.derived_class_addr", kHandled}, // Constant offset.
    {"cir.do", kHandled},                 // RegionBranchOpInterface.
    {"cir.dyn_cast", kHandled},           // Collapse.
    {"cir.extract_member", kHandled},     // Aggregate slot.
    {"cir.for", kHandled},                // RegionBranchOpInterface.
    {"cir.freeze", kHandled},             // Identity.
    {"cir.func", kHandled},               // Formals (the builder's root).
    {"cir.get_bitfield", kHandled},       // Ref on an integer.
    {"cir.get_element", kHandled},        // Offset or array.
    {"cir.get_global", kHandled},         // Global/Function node per symbol.
    {"cir.get_member", kHandled},         // Constant offset.
    {"cir.get_runtime_member", kHandled}, // Collapse.
    {"cir.if", kHandled},                 // RegionBranchOpInterface.
    {"cir.indirect_br", kHandled},        // BranchOpInterface.
    {"cir.insert_member", kHandled},      // Aggregate slots.
    {"cir.launder", kHandled},            // Identity.
    {"cir.libc.memset", kHandled},     // Mod; a byte pattern makes no pointer.
    {"cir.load", kHandled},            // Link / aggregate slots.
    {"cir.ptr_mask", kHandled},        // Collapse.
    {"cir.ptr_stride", kHandled},      // Offset or array.
    {"cir.resume", kHandled},          // RegionBranchTerminatorOpInterface.
    {"cir.return", kHandled},          // Return cell.
    {"cir.scope", kHandled},           // RegionBranchOpInterface.
    {"cir.select", kHandled},          // Unify.
    {"cir.set_bitfield", kHandled},    // Mod on an integer.
    {"cir.store", kHandled},           // Link / aggregate slots.
    {"cir.switch", kHandled},          // RegionBranchOpInterface.
    {"cir.switch.flat", kHandled},     // BranchOpInterface.
    {"cir.ternary", kHandled},         // RegionBranchOpInterface.
    {"cir.try", kHandled},             // RegionBranchOpInterface.
    {"cir.try_call", kHandled},        // CallSiteRecord + BranchOpInterface.
    {"cir.vec.masked_load", kHandled}, // Ref; no pointer elements.
    {"cir.while", kHandled},           // RegionBranchOpInterface.
    {"cir.yield", kHandled},           // RegionBranchTerminatorOpInterface.

    //===------------------------------------------------------------------===//
    // NoPointerSemantics, by family.
    //===------------------------------------------------------------------===//
    // Integer arithmetic and bit manipulation: integer (or integer vector)
    // operands and results only.
    {"cir.abs", kNoPtr},
    {"cir.add", kNoPtr},
    {"cir.add.overflow", kNoPtr},
    {"cir.and", kNoPtr},
    {"cir.bitreverse", kNoPtr},
    {"cir.builtin_int_cast", kNoPtr},
    {"cir.byte_swap", kNoPtr},
    {"cir.clrsb", kNoPtr},
    {"cir.clz", kNoPtr},
    {"cir.ctz", kNoPtr},
    {"cir.dec", kNoPtr},
    {"cir.div", kNoPtr},
    {"cir.ffs", kNoPtr},
    {"cir.fshl", kNoPtr},
    {"cir.fshr", kNoPtr},
    {"cir.inc", kNoPtr},
    {"cir.max", kNoPtr},
    {"cir.min", kNoPtr},
    {"cir.minus", kNoPtr},
    {"cir.mul", kNoPtr},
    {"cir.mul.overflow", kNoPtr},
    {"cir.not", kNoPtr},
    {"cir.or", kNoPtr},
    {"cir.parity", kNoPtr},
    {"cir.popcount", kNoPtr},
    {"cir.rem", kNoPtr},
    {"cir.rotate", kNoPtr},
    {"cir.shift", kNoPtr},
    {"cir.sub", kNoPtr},
    {"cir.sub.overflow", kNoPtr},
    {"cir.xor", kNoPtr},
    // Floating-point math: floating-point (or vector) values in, numbers out;
    // none takes an out-pointer (frexp/modf return their parts as results).
    {"cir.acos", kNoPtr},
    {"cir.asin", kNoPtr},
    {"cir.atan", kNoPtr},
    {"cir.atan2", kNoPtr},
    {"cir.ceil", kNoPtr},
    {"cir.copysign", kNoPtr},
    {"cir.cos", kNoPtr},
    {"cir.cosh", kNoPtr},
    {"cir.exp", kNoPtr},
    {"cir.exp10", kNoPtr},
    {"cir.exp2", kNoPtr},
    {"cir.fabs", kNoPtr},
    {"cir.fadd", kNoPtr},
    {"cir.fdiv", kNoPtr},
    {"cir.floor", kNoPtr},
    {"cir.fma", kNoPtr},
    {"cir.fmaximum", kNoPtr},
    {"cir.fmaxnum", kNoPtr},
    {"cir.fminimum", kNoPtr},
    {"cir.fminnum", kNoPtr},
    {"cir.fmod", kNoPtr},
    {"cir.fmul", kNoPtr},
    {"cir.fmuladd", kNoPtr},
    {"cir.fneg", kNoPtr},
    {"cir.frem", kNoPtr},
    {"cir.frexp", kNoPtr},
    {"cir.fsub", kNoPtr},
    {"cir.is_fp_class", kNoPtr},
    {"cir.llrint", kNoPtr},
    {"cir.llround", kNoPtr},
    {"cir.log", kNoPtr},
    {"cir.log10", kNoPtr},
    {"cir.log2", kNoPtr},
    {"cir.lrint", kNoPtr},
    {"cir.lround", kNoPtr},
    {"cir.modf", kNoPtr},
    {"cir.nearbyint", kNoPtr},
    {"cir.pow", kNoPtr},
    {"cir.rint", kNoPtr},
    {"cir.round", kNoPtr},
    {"cir.roundeven", kNoPtr},
    {"cir.signbit", kNoPtr},
    {"cir.sin", kNoPtr},
    {"cir.sinh", kNoPtr},
    {"cir.sqrt", kNoPtr},
    {"cir.tan", kNoPtr},
    {"cir.tanh", kNoPtr},
    {"cir.trunc", kNoPtr},
    // Complex values: the parts are integers or floats.
    {"cir.complex.add", kNoPtr},
    {"cir.complex.conj", kNoPtr},
    {"cir.complex.create", kNoPtr},
    {"cir.complex.div", kNoPtr},
    {"cir.complex.imag", kNoPtr},
    {"cir.complex.mul", kNoPtr},
    {"cir.complex.real", kNoPtr},
    {"cir.complex.sub", kNoPtr},
    // Vectors and matrices: CIR vector and matrix elements are scalars,
    // never pointers.
    {"cir.matrix.transpose", kNoPtr},
    {"cir.vec.cmp", kNoPtr},
    {"cir.vec.create", kNoPtr},
    {"cir.vec.extract", kNoPtr},
    {"cir.vec.insert", kNoPtr},
    {"cir.vec.shuffle", kNoPtr},
    {"cir.vec.shuffle.dynamic", kNoPtr},
    {"cir.vec.splat", kNoPtr},
    {"cir.vec.ternary", kNoPtr},
    // Comparisons and queries: read the address only, produce a number or a
    // bool, never access the pointee.
    {"cir.cmp", kNoPtr},
    {"cir.cmp3way", kNoPtr},
    {"cir.is_constant", kNoPtr},
    {"cir.objsize", kNoPtr},
    {"cir.ptr_diff", kNoPtr},
    // Member-pointer adjustments: offset arithmetic on !cir.data_member and
    // !cir.method values, no object is accessed (get_method is Pessimistic).
    {"cir.base_data_member", kNoPtr},
    {"cir.base_method", kNoPtr},
    {"cir.derived_data_member", kNoPtr},
    {"cir.derived_method", kNoPtr},
    // Markers and hints: lifetime, assumptions, prefetch and cache
    // maintenance, fences; they neither read nor write a pointer slot.
    {"cir.assume", kNoPtr},
    {"cir.atomic.fence", kNoPtr},
    {"cir.clear_cache", kNoPtr},
    {"cir.expect", kNoPtr},
    {"cir.lifetime.end", kNoPtr},
    {"cir.lifetime.start", kNoPtr},
    {"cir.prefetch", kNoPtr},
    {"cir.token.none", kNoPtr},
    // Control flow that carries no values.
    {"cir.break", kNoPtr},
    {"cir.continue", kNoPtr},
    {"cir.goto", kNoPtr},
    {"cir.label", kNoPtr},
    {"cir.trap", kNoPtr},
    {"cir.unreachable", kNoPtr},
    // Exception-handling token plumbing: tokens and type ids only; the ops
    // that produce or consume the exception pointer are Pessimistic.
    {"cir.begin_cleanup", kNoPtr},
    {"cir.eh.dispatch", kNoPtr},
    {"cir.eh.initiate", kNoPtr},
    {"cir.eh.terminate", kNoPtr},
    {"cir.eh.typeid", kNoPtr},
    {"cir.eh.unexpected", kNoPtr},
    {"cir.end_catch", kNoPtr},

    //===------------------------------------------------------------------===//
    // Pessimistic: matters, not modelled yet.
    //===------------------------------------------------------------------===//
    {"cir.address_of_return_address", kPessimistic}, // Frame internals.
    {"cir.alloc.exception", kPessimistic},           // TODO: exception objects.
    {"cir.array.ctor", kPessimistic},            // TODO(T06): per-element ctor.
    {"cir.array.dtor", kPessimistic},            // TODO(T06): per-element dtor.
    {"cir.asm", kPessimistic},                   // Opaque.
    {"cir.begin_catch", kPessimistic},           // TODO: exception objects.
    {"cir.block_address", kPessimistic},         // Label address.
    {"cir.catch_param", kPessimistic},           // TODO: exception objects.
    {"cir.construct_catch_param", kPessimistic}, // TODO: exception objects.
    {"cir.copy", kPessimistic},                  // TODO(T05): cir.copy.
    {"cir.coro.intrinsic.alloc", kPessimistic},  // Coroutines.
    {"cir.coro.intrinsic.begin", kPessimistic},  // Coroutines.
    {"cir.coro.intrinsic.destroy", kPessimistic}, // Coroutines.
    {"cir.coro.intrinsic.done", kPessimistic},    // Coroutines.
    {"cir.coro.intrinsic.end", kPessimistic},     // Coroutines.
    {"cir.coro.intrinsic.free", kPessimistic},    // Coroutines.
    {"cir.coro.intrinsic.id", kPessimistic},      // Coroutines.
    {"cir.coro.intrinsic.promise", kPessimistic}, // Coroutines.
    {"cir.coro.intrinsic.resume", kPessimistic},  // Coroutines.
    {"cir.coro.intrinsic.size", kPessimistic},    // Coroutines.
    {"cir.cpuid", kPessimistic},                  // Rare; writes an int array.
    {"cir.delete_array", kPessimistic},           // TODO(T06): dtor + delete.
    {"cir.eh.inflight_exception", kPessimistic},  // TODO: exception objects.
    {"cir.eh.longjmp", kPessimistic},             // jmp_buf.
    {"cir.eh.setjmp", kPessimistic},              // jmp_buf.
    {"cir.end_cleanup", kPessimistic},            // TODO: exception objects.
    {"cir.frame_address", kPessimistic},          // Frame internals.
    {"cir.get_method", kPessimistic},             // TODO(T09): method pointers.
    {"cir.global", kPessimistic},                 // TODO(T05): module level.
    {"cir.indirect_goto", kPessimistic},          // Label address.
    {"cir.init_catch_param", kPessimistic},       // TODO: exception objects.
    {"cir.libc.memchr", kPessimistic},            // TODO(T06): result into src.
    {"cir.libc.memcpy", kPessimistic},            // TODO(T05): memcpy family.
    {"cir.libc.memmove", kPessimistic},           // TODO(T05): memcpy family.
    {"cir.local_init", kPessimistic},             // TODO(T05): static locals.
    {"cir.resume.flat", kPessimistic},            // TODO: exception objects.
    {"cir.return_address", kPessimistic},         // Frame internals.
    {"cir.stackrestore", kPessimistic},           // Frame internals.
    {"cir.stacksave", kPessimistic},              // Frame internals.
    {"cir.std.find", kPessimistic},               // TODO(T06): library model.
    {"cir.std.strlen", kPessimistic},             // TODO(T06): library model.
    {"cir.throw", kPessimistic},                  // TODO: exception objects.
    {"cir.try_throw", kPessimistic},              // TODO: exception objects.
    {"cir.va_arg", kPessimistic},                 // Varargs are not modelled.
    {"cir.va_copy", kPessimistic},                // Varargs are not modelled.
    {"cir.va_end", kPessimistic},                 // Varargs are not modelled.
    {"cir.va_start", kPessimistic},               // Varargs are not modelled.
    {"cir.vtable.address_point", kPessimistic},   // TODO(T09): vtables.
    {"cir.vtable.get_type_info", kPessimistic},   // TODO(T09): vtables.
    {"cir.vtable.get_virtual_fn_addr", kPessimistic}, // TODO(T09): vtables.
    {"cir.vtable.get_vptr", kPessimistic},            // TODO(T09): vtables.
    {"cir.vtt.address_point", kPessimistic},          // TODO(T09): vtables.
};

StringRef cir::pointsto::getOpClassName(OpClass cls) {
  switch (cls) {
  case OpClass::Handled:
    return "Handled";
  case OpClass::NoPointerSemantics:
    return "NoPointerSemantics";
  case OpClass::Pessimistic:
    return "Pessimistic";
  }
  llvm_unreachable("unknown OpClass");
}

llvm::ArrayRef<OpClassEntry> cir::pointsto::getOpClassificationTable() {
  return kOpClassificationTable;
}

std::optional<OpClass> cir::pointsto::lookupOpClass(StringRef opName) {
  static const llvm::StringMap<OpClass> map = [] {
    llvm::StringMap<OpClass> map;
    for (const OpClassEntry &entry : kOpClassificationTable)
      map.try_emplace(entry.name, entry.cls);
    return map;
  }();
  auto it = map.find(opName);
  if (it == map.end())
    return std::nullopt;
  return it->second;
}

bool cir::pointsto::isPointerFreeLLVMIntrinsic(StringRef name) {
  // Pure value computations on integers, floats and vectors of them.
  static const llvm::StringSet<> names = {
      "fptosi.sat", "fptoui.sat", "frexp", "fshl", "fshr", "ldexp", "powi",
  };
  return names.contains(name) || name.starts_with("vector.reduce.");
}
