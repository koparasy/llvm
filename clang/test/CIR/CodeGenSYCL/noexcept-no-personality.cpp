// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o - \
// RUN:   | FileCheck %s

// SYCL device code has no exceptions: a noexcept device function gets no
// terminate scope and no EH personality, matching classic CodeGen.

// CHECK-NOT: personality(
// CHECK-NOT: @__gxx_personality_v0
// CHECK: cir.func {{.*}}@_Z1fi(
__attribute__((sycl_device)) int f(int x) noexcept { return x + 1; }
