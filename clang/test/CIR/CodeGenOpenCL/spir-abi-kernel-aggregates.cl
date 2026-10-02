// RUN: %clang_cc1 -triple spir64-unknown-unknown -fclangir -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple spir64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// OpenCL on SPIR: an aggregate kernel argument is passed byval, an aggregate
// return goes through an sret pointer in the private (default) address space
// (three-element vectors, passed directly, are covered by as_type.cl).

typedef struct { int a; float b; long c; } S;

// CHECK: define {{.*}}spir_func void @make(ptr {{.*}}sret(%struct.S) align 8 {{.*}}, i32 {{.*}})
S make(int v) { S s = {v, 1.0f, 2}; return s; }

// CHECK: define {{.*}}spir_kernel void @k(ptr addrspace(1) {{.*}}%{{.*}}, ptr {{.*}}byval(%struct.S) align 8 {{.*}})
kernel void k(global int *out, S s) {
  S t = make(s.a);
  out[0] = t.a + (int)s.b;
}
