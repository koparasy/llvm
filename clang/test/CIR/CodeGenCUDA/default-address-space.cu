#include "Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -fclangir -emit-cir -mmlir -mlir-print-ir-before=cir-target-lowering \
// RUN:   -I%S/Inputs/ %s -o %t.cir 2> %t-pre.cir
// RUN: FileCheck --check-prefixes=CIR-PRE,CIR-PRE-NVPTX --input-file=%t-pre.cir %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -fclangir -emit-llvm -I%S/Inputs/ %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-NVPTX --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -emit-llvm -I%S/Inputs/ %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-NVPTX --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fclangir -emit-cir -mmlir -mlir-print-ir-before=cir-target-lowering \
// RUN:   -I%S/Inputs/ %s -o %t.cir 2> %t-pre-amdgpu.cir
// RUN: FileCheck --check-prefixes=CIR-PRE,CIR-PRE-AMDGPU --input-file=%t-pre-amdgpu.cir %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fclangir -emit-llvm -I%S/Inputs/ %s -o %t-cir-amdgpu.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-AMDGPU --input-file=%t-cir-amdgpu.ll %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -emit-llvm -I%S/Inputs/ %s -o %t-amdgpu.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-AMDGPU --input-file=%t-amdgpu.ll %s

// On an offload device an unqualified pointer is a generic pointer. CIRGen
// spells it offload_generic, so it stays distinct from the address space of
// the object it points to until target lowering.

struct Node {
  Node *next;
  int *data;
};

// LLVM: %struct.Node = type { ptr, ptr }

__device__ int g;
__device__ Node n = {&n, &g};

// The initializer refers to the globals through generic pointers.
// CIR-PRE: cir.global external lang_address_space(offload_global) @n = #cir.const_record<{#cir.global_view<@n> : !cir.ptr<!rec_Node, lang_address_space(offload_generic)>, #cir.global_view<@g> : !cir.ptr<!s32i, lang_address_space(offload_generic)>}> : !rec_Node

// LLVM: @g = addrspace(1) externally_initialized global i32 0
// LLVM: @n = addrspace(1) externally_initialized global %struct.Node { ptr addrspacecast (ptr addrspace(1) @n to ptr), ptr addrspacecast (ptr addrspace(1) @g to ptr) }

__device__ int *addr_of_global() { return &g; }

// Taking the address of a global yields a pointer in the global's address
// space, which is cast to the generic address space of the declared type.
// CIR-PRE-LABEL: cir.func {{.*}} @_Z14addr_of_globalv() -> (!cir.ptr<!s32i, lang_address_space(offload_generic)>
// CIR-PRE:         %[[G:.*]] = cir.get_global @g : !cir.ptr<!s32i, lang_address_space(offload_global)>
// CIR-PRE:         cir.cast address_space %[[G]] : !cir.ptr<!s32i, lang_address_space(offload_global)> -> !cir.ptr<!s32i, lang_address_space(offload_generic)>

// LLVM-LABEL: define {{.*}} ptr @_Z14addr_of_globalv()
// LLVM:         addrspacecast (ptr addrspace(1) @g to ptr)

__device__ int follow(Node *p, bool c) {
  Node local = *p;
  int *q = c ? local.data : p->next->data;
  return *q;
}

// Members of a record reached through a generic pointer are generic too.
// CIR-PRE-LABEL: cir.func {{.*}} @_Z6followP4Nodeb(%arg0: !cir.ptr<!rec_Node, lang_address_space(offload_generic)>
// CIR-PRE-NVPTX:   cir.cast address_space %{{.*}} : !cir.ptr<!rec_Node> -> !cir.ptr<!rec_Node, lang_address_space(offload_generic)>
// CIR-PRE-AMDGPU:  cir.cast address_space %{{.*}} : !cir.ptr<!rec_Node, lang_address_space(offload_private)> -> !cir.ptr<!rec_Node, lang_address_space(offload_generic)>
// CIR-PRE:         cir.get_member %{{.*}}[1] {name = "data"} : !cir.ptr<!rec_Node, lang_address_space(offload_generic)> -> !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_generic)>, lang_address_space(offload_generic)>
// CIR-PRE:         cir.get_member %{{.*}}[0] {name = "next"} : !cir.ptr<!rec_Node, lang_address_space(offload_generic)> -> !cir.ptr<!cir.ptr<!rec_Node, lang_address_space(offload_generic)>, lang_address_space(offload_generic)>

// LLVM-LABEL:  define {{.*}} i32 @_Z6followP4Nodeb(ptr
// LLVM-NVPTX:    alloca %struct.Node, align 8
// LLVM-AMDGPU:   alloca %struct.Node, align 8, addrspace(5)
// LLVM:          getelementptr inbounds nuw %struct.Node, ptr %{{.*}}, i32 0, i32 1
// LLVM:          getelementptr inbounds nuw %struct.Node, ptr %{{.*}}, i32 0, i32 0
// LLVM:          getelementptr inbounds nuw %struct.Node, ptr %{{.*}}, i32 0, i32 1
