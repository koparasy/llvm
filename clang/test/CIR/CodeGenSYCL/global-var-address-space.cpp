// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fno-gpu-rdc -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fno-gpu-rdc -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM

// Under -fsycl-is-device on SPIR(-V), a namespace-scope / static-storage
// variable with Default address space is placed in the SYCL global address
// space (SPIR-V CrossWorkgroup / addrspace(1)). This mirrors OG codegen
// (CodeGenModule::GetGlobalVarAddressSpace) and exercises the SYCL branch of
// CIRGenModule::getGlobalVarAddressSpace.

__attribute__((sycl_device)) long __spirv_BuiltInGlobalInvocationId(int);

const int table[4] = {1, 2, 3, 4};

__attribute__((sycl_device))
int lookup(int *out) {
  long i = __spirv_BuiltInGlobalInvocationId(0);
  out[0] = table[i & 3];
  return out[0];
}

// CIR: cir.global {{.*}}constant {{.*}}target_address_space(1) @_ZL5table = #cir.const_array<{{.*}}> : !cir.array<!s32i x 4>
// CIR: cir.get_global @_ZL5table : !cir.ptr<!cir.array<!s32i x 4>, target_address_space(1)>

// LLVM: @_ZL5table = {{.*}}addrspace(1) constant [4 x i32] [i32 1, i32 2, i32 3, i32 4]
// LLVM: getelementptr [4 x i32], ptr addrspace(1) @_ZL5table
