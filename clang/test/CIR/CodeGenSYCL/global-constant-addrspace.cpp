// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fclangir -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -emit-llvm %s -o - | FileCheck %s

// SYCL device constants and string literals live in the global address space.

// CHECK-DAG: @.str = private {{.*}}addrspace(1) constant [6 x i8] c"hello\00"
// CHECK-DAG: @_ZL5table = internal addrspace(1) constant [4 x i32] [i32 1, i32 2, i32 3, i32 4]

static const int table[4] = {1, 2, 3, 4};
__attribute__((sycl_device)) int f(int i, const char **out) {
  *out = "hello";
  return table[i & 3];
}
