// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fno-gpu-rdc -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fno-gpu-rdc -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM

// Under -fsycl-is-device on SPIR(-V), an unannotated (Default) pointer pointee
// is placed in the generic address space (SPIR-V Generic / addrspace(4)), so
// the free-function kernel body loads/stores through generic pointers. This
// matches OG codegen and lets llvm-spirv legalize the global->generic cast.

__attribute__((sycl_device)) long __spirv_BuiltInGlobalInvocationId(int);

__attribute__((sycl_device))
void vec_add(const float *a, const float *b, float *c) {
  long i = __spirv_BuiltInGlobalInvocationId(0);
  c[i] = a[i] + b[i];
}

// CIR: cir.func{{.*}} @_Z7vec_addPKfS0_Pf(
// CIR-SAME: !cir.ptr<!cir.float, target_address_space(4)>
// CIR-SAME: !cir.ptr<!cir.float, target_address_space(4)>
// CIR-SAME: !cir.ptr<!cir.float, target_address_space(4)>

// LLVM: define {{.*}} void @_Z7vec_addPKfS0_Pf(ptr addrspace(4){{.*}}, ptr addrspace(4){{.*}}, ptr addrspace(4){{.*}})
// LLVM: call {{.*}} @_Z33__spirv_BuiltInGlobalInvocationIdi
// LLVM: load float, ptr addrspace(4)
// LLVM: load float, ptr addrspace(4)
// LLVM: fadd float
// LLVM: store float {{.*}}, ptr addrspace(4)
