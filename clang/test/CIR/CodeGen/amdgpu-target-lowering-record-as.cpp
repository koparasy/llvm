// Tests for LangAddressSpaceAttr -> TargetAddressSpaceAttr conversion of
// pointer members INSIDE records in the cir-target-lowering pass. Without a
// RecordType conversion rule the struct member keeps its lang address space
// while the get_member result pointer is lowered to a target address space,
// tripping GetMemberOp's member-type equality verifier.
//
// AMDGPU address space mapping:
//   offload_global -> 1 (GLOBAL_ADDRESS)

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -emit-cir \
// RUN:   -mmlir -mlir-print-ir-before=cir-target-lowering %s -o %t.cir 2> %t.pre.cir
// RUN: FileCheck --check-prefix=PRE --input-file=%t.pre.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=POST --input-file=%t.cir %s

struct S {
  int [[clang::opencl_global]] *p;
};

int [[clang::opencl_global]] *load_member(S *s) { return s->p; }

// Before target lowering the member and the get_member result both carry the
// lang address space.

// PRE: !rec_S = !cir.struct<"S" {data !cir.ptr<!s32i, lang_address_space(offload_global)>}>
// PRE: cir.func {{.*}}@_Z11load_memberP1S
// PRE:   cir.get_member {{.*}} -> !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_global)>>

// After target lowering the member pointer address space is lowered in lockstep
// with the get_member result, so both are target_address_space(1) and the
// record member type matches the get_member result pointee. The record is
// lowered in place, keeping its name.

// POST: !rec_[[REC:.*]] = !cir.struct<"S" {data !cir.ptr<!s32i, target_address_space(1)>}>
// POST: cir.func {{.*}}@_Z11load_memberP1S
// POST:   cir.get_member {{.*}} !cir.ptr<!rec_[[REC]]> -> !cir.ptr<!cir.ptr<!s32i, target_address_space(1)>>
