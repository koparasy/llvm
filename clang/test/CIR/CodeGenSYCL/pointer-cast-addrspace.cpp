// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -fdeclare-spirv-builtins \
// RUN:   -triple spir64-unknown-unknown -fclangir -emit-llvm %s -o - | FileCheck %s

// Pointer casts whose operand and result differ in address space and pointee
// are split into an address-space cast and a pointee change, as classic
// CodeGen does with opaque pointers.

struct F4 { float x, y, z, w; };
struct H2 { short a, b; };

// reinterpret_cast of the address of a private local to a generic pointer.
// CHECK-LABEL: define {{.*}}@_Z5reinpPK2F4(
// CHECK: addrspacecast ptr %{{.*}} to ptr addrspace(4)
__attribute__((sycl_device)) int reinp(const F4 *in) {
  F4 v4 = *in;
  const H2 *h2 = reinterpret_cast<const H2 *>(&v4);
  return h2[1].a;
}

// An address-space conversion where both AST address spaces map to the same
// target one and the pointee changes (a SPIR-V builtin returning
// `void __global *`).
// CHECK-LABEL: define {{.*}}@_Z8toGlobalPj(
// CHECK: call spir_func {{.*}}ptr addrspace(1) @_Z{{.*}}GenericCastToPtrExplicit_ToGlobal
__attribute__((sycl_device)) unsigned toGlobal(unsigned *p) {
  using GP = __attribute__((sycl_global)) unsigned *;
  GP g = (GP)__spirv_GenericCastToPtrExplicit_ToGlobal(p, 5);
  return *g;
}
