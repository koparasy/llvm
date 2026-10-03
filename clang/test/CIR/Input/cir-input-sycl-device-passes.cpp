// A ClangIR input compiled for a SYCL device (as the offload combine does for
// each split device module) runs the SYCL device backend passes, which key on
// -fsycl-is-device.

// RUN: %clang_cc1 -triple spir64-unknown-unknown -fsycl-is-device -fclangir \
// RUN:   -emit-cir %s -o %t.cir
// RUN: %clang_cc1 -triple spir64-unknown-unknown -fsycl-is-device -O2 \
// RUN:   -emit-llvm -x cir %t.cir -o /dev/null -mllvm -print-pipeline-passes \
// RUN:   | FileCheck %s

// CHECK: sycl-propagate-aspects-usage
// CHECK-SAME: compile-time-properties

__attribute__((sycl_device)) int f(int x) { return x + 1; }
