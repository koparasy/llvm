// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fno-gpu-rdc -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fno-gpu-rdc -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fno-gpu-rdc -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// During device compilation a SYCL free-function kernel is outlined into a
// __sycl_kernel_ entry point carrying an implicit device_kernel attribute. It
// must be emitted with the SPIR-V kernel calling convention so that
// sycl-post-link recognizes it as a kernel. Its USM pointer arguments are
// rewritten to the global address space (SPIR-V CrossWorkGroup / addrspace(1)).

__attribute__((sycl_device))
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void ff(int *p) { *p = 42; }

// CIR: cir.func{{.*}} @_Z16__sycl_kernel_ffPi(%arg0: !cir.ptr<!s32i, target_address_space(1)>{{.*}}) {{.*}}cc(spir_kernel)

// LLVM: define {{.*}}spir_kernel void @_Z16__sycl_kernel_ffPi(ptr addrspace(1){{.*}}#[[ATTRS:[0-9]+]]
// LLVM: attributes #[[ATTRS]] = {{.*}}"sycl-module-id"="{{.*}}free-function-kernel-cc.cpp"

// OGCG: define {{.*}}spir_kernel void @_Z16__sycl_kernel_ffPi(ptr addrspace(1){{.*}}#[[ATTRS:[0-9]+]]
// OGCG: attributes #[[ATTRS]] = {{.*}}"sycl-module-id"="{{.*}}free-function-kernel-cc.cpp"
