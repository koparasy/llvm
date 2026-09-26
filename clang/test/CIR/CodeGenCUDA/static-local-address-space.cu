#include "Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -fclangir -emit-cir -mmlir -mlir-print-ir-before=cir-target-lowering \
// RUN:   -I%S/Inputs/ %s -o %t.cir 2> %t-pre.cir
// RUN: FileCheck --check-prefix=CIR-PRE --input-file=%t-pre.cir %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -fclangir -emit-llvm -I%S/Inputs/ %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -emit-llvm -I%S/Inputs/ %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fclangir -emit-cir -mmlir -mlir-print-ir-before=cir-target-lowering \
// RUN:   -I%S/Inputs/ %s -o %t.cir 2> %t-pre-amdgpu.cir
// RUN: FileCheck --check-prefix=CIR-PRE --input-file=%t-pre-amdgpu.cir %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fclangir -emit-llvm -I%S/Inputs/ %s -o %t-cir-amdgpu.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir-amdgpu.ll %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -emit-llvm -I%S/Inputs/ %s -o %t-amdgpu.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-amdgpu.ll %s

// Static local variables in device code live in the address space of their
// CUDA attribute, like globals at namespace scope, and their uses are cast to
// the generic address space of the declared type.

// CIR-PRE-DAG: cir.global "private" internal dso_local lang_address_space(offload_local) @_ZZ10use_sharediE1s = #cir.undef : !s32i
// CIR-PRE-DAG: cir.global "private" internal dso_local lang_address_space(offload_local) @_ZZ10use_sharediE2sa = #cir.undef : !cir.array<!s32i x 8>
// CIR-PRE-DAG: cir.global "private" internal dso_local lang_address_space(offload_global) @_ZZ10use_sharediE2sd = #cir.int<0> : !s32i
// CIR-PRE-DAG: cir.global "private" internal dso_local lang_address_space(offload_global) @_ZZ13ptr_to_staticvE1x = #cir.int<0> : !s32i

// LLVM-DAG: @_ZZ10use_sharediE1s = internal addrspace(3) global i32 undef
// LLVM-DAG: @_ZZ10use_sharediE2sa = internal addrspace(3) global [8 x i32] undef
// LLVM-DAG: @_ZZ10use_sharediE2sd = internal addrspace(1) global i32 0
// LLVM-DAG: @_ZZ13ptr_to_staticvE1x = internal addrspace(1) global i32 0

__device__ int use_shared(int i) {
  __shared__ int s;
  __shared__ int sa[8];
  static __device__ int sd;
  s = i;
  sa[i] = i;
  sd = i;
  int *q = &s;
  return *q + sa[0] + sd;
}

// CIR-PRE-LABEL: cir.func {{.*}} @_Z10use_sharedi
// CIR-PRE:         %[[S:.*]] = cir.get_global @_ZZ10use_sharediE1s : !cir.ptr<!s32i, lang_address_space(offload_local)>
// CIR-PRE:         cir.cast address_space %[[S]] : !cir.ptr<!s32i, lang_address_space(offload_local)> -> !cir.ptr<!s32i, lang_address_space(offload_generic)>
// CIR-PRE:         %[[SA:.*]] = cir.get_global @_ZZ10use_sharediE2sa : !cir.ptr<!cir.array<!s32i x 8>, lang_address_space(offload_local)>
// CIR-PRE:         cir.cast address_space %[[SA]] : !cir.ptr<!cir.array<!s32i x 8>, lang_address_space(offload_local)> -> !cir.ptr<!cir.array<!s32i x 8>, lang_address_space(offload_generic)>
// CIR-PRE:         %[[SD:.*]] = cir.get_global @_ZZ10use_sharediE2sd : !cir.ptr<!s32i, lang_address_space(offload_global)>
// CIR-PRE:         cir.cast address_space %[[SD]] : !cir.ptr<!s32i, lang_address_space(offload_global)> -> !cir.ptr<!s32i, lang_address_space(offload_generic)>

// LLVM-LABEL: define {{.*}} i32 @_Z10use_sharedi(
// LLVM:         store i32 %{{.*}}, ptr addrspacecast (ptr addrspace(3) @_ZZ10use_sharediE1s to ptr)
// LLVM:         store i32 %{{.*}}, ptr addrspacecast (ptr addrspace(1) @_ZZ10use_sharediE2sd to ptr)
// LLVM:         store ptr addrspacecast (ptr addrspace(3) @_ZZ10use_sharediE1s to ptr), ptr
// LLVM:         load i32, ptr addrspacecast (ptr addrspace(3) @_ZZ10use_sharediE2sa to ptr)
// LLVM:         load i32, ptr addrspacecast (ptr addrspace(1) @_ZZ10use_sharediE2sd to ptr)

__device__ int *ptr_to_static() {
  static __device__ int x;
  return &x;
}

// LLVM-LABEL: define {{.*}} ptr @_Z13ptr_to_staticv()
// LLVM:         addrspacecast (ptr addrspace(1) @_ZZ13ptr_to_staticvE1x to ptr)
