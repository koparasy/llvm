// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fno-offload-use-alloca-addrspace-for-srets -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,GENERIC
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fno-offload-use-alloca-addrspace-for-srets -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,GENERIC
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,ALLOCA
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,ALLOCA

// SYCL device functions follow classic SPIR CodeGen's ABI: an aggregate is
// passed byval and returned through an sret pointer. The sret pointer is in
// the generic address space when the driver passes
// -fno-offload-use-alloca-addrspace-for-srets (as it does for SYCL), else in
// the alloca one -- so ClangIR and classic objects link together.

struct P { float x, y, z; int tag; };

// GENERIC: define {{.*}}spir_func void @_Z7combine1PS_(ptr addrspace(4) {{.*}}sret(%struct.P) align 4 %{{.*}}, ptr {{.*}}byval(%struct.P) align 4 %{{.*}}, ptr {{.*}}byval(%struct.P) align 4 %{{.*}})
// ALLOCA: define {{.*}}spir_func void @_Z7combine1PS_(ptr {{.*}}sret(%struct.P) align 4 %{{.*}}, ptr {{.*}}byval(%struct.P) align 4 %{{.*}}, ptr {{.*}}byval(%struct.P) align 4 %{{.*}})
__attribute__((sycl_device)) P combine(P a, P b) {
  return P{a.x + b.x, a.y * b.y, a.z - b.z, a.tag ^ b.tag};
}

// CHECK: define {{.*}}spir_func void @_Z6callerPf(
// GENERIC: call spir_func void @_Z7combine1PS_(ptr addrspace(4) {{.*}}sret(%struct.P) align 4 %{{.*}}, ptr {{.*}}byval(%struct.P) align 4 %{{.*}}, ptr {{.*}}byval(%struct.P) align 4 %{{.*}})
// ALLOCA: call spir_func void @_Z7combine1PS_(ptr {{.*}}sret(%struct.P) align 4 %{{.*}}, ptr {{.*}}byval(%struct.P) align 4 %{{.*}}, ptr {{.*}}byval(%struct.P) align 4 %{{.*}})
__attribute__((sycl_device)) void caller(float *out) {
  P a{1, 2, 3, 4}, b{5, 6, 7, 8};
  *out = combine(a, b).y;
}
