// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -internal-isystem %S/../../CodeGenSYCL/Inputs -fclangir -emit-cir %s \
// RUN:   -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -internal-isystem %S/../../CodeGenSYCL/Inputs -fclangir -emit-llvm %s \
// RUN:   -o - | FileCheck %s --check-prefix=LLVM

// Each SYCL kernel parameter records the closure member SemaSYCL initializes
// from it (cir.sycl_kernel_param), and the kernel's closure object alloca is
// tagged (cir.sycl_kernel_object), so offload passes need not pattern-match
// the kernel prologue. Neither reaches LLVM IR.

#include "sycl.hpp"

struct Pod {
  int x;
  float y;
};

struct Holder {
  sycl::accessor<int, 1, sycl::access::mode::read> acc;
  int k;
};

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

// CHECK-LABEL: cir.func {{.*}}@{{.*}}lambda_kernel(
// CHECK-SAME: %arg0: {{.*}}{cir.sycl_kernel_param = {accessor_part = "ptr", field = "w", init_arg = 0 : i32, kind = "accessor", path = ["w"]}
// CHECK-SAME: %arg1: {{.*}}{cir.sycl_kernel_param = {accessor_part = "access_range", field = "w", init_arg = 1 : i32, kind = "accessor", path = ["w"]}
// CHECK-SAME: %arg2: {{.*}}{cir.sycl_kernel_param = {accessor_part = "mem_range", field = "w", init_arg = 2 : i32, kind = "accessor", path = ["w"]}
// CHECK-SAME: %arg3: {{.*}}{cir.sycl_kernel_param = {accessor_part = "offset", field = "w", init_arg = 3 : i32, kind = "accessor", path = ["w"]}
// CHECK-SAME: %arg4: {{.*}}{cir.sycl_kernel_param = {accessor_part = "ptr", field = "lm", init_arg = 0 : i32, kind = "local_accessor", path = ["lm"]}
// CHECK-SAME: %arg8: {{.*}}{cir.sycl_kernel_param = {field = "p", kind = "pointer", path = ["p"]}
// CHECK-SAME: %arg9: {{.*}}{cir.sycl_kernel_param = {field = "n", kind = "scalar", path = ["n"]}
// CHECK-SAME: %arg10: {{.*}}{cir.sycl_kernel_param = {field = "s", kind = "std_layout", path = ["s"]}
// CHECK-SAME: %arg11: {{.*}}{cir.sycl_kernel_param = {accessor_part = "ptr", field = "h", init_arg = 0 : i32, kind = "accessor", path = ["h", "acc"]}
// CHECK-SAME: %arg15: {{.*}}{cir.sycl_kernel_param = {field = "h", kind = "scalar", path = ["h", "k"]}
// CHECK-SAME: %arg16: {{.*}}{cir.sycl_kernel_param = {accessor_part = "ptr", field = "arr", init_arg = 0 : i32, kind = "accessor", path = ["arr", "[0]"]}
// CHECK-SAME: %arg20: {{.*}}{cir.sycl_kernel_param = {accessor_part = "ptr", field = "arr", init_arg = 0 : i32, kind = "accessor", path = ["arr", "[1]"]}
// CHECK: cir.alloca "__SYCLKernel" {{.*}} {cir.sycl_kernel_object}
void lambdas(sycl::queue &q, sycl::buffer<int, 1> &buf, int *p, int n, Pod s) {
  q.submit([&](sycl::handler &cgh) {
    auto w = buf.get_access<sycl::access::mode::write>(cgh);
    sycl::local_accessor<int, 1> lm;
    Holder h;
    sycl::accessor<int, 1, sycl::access::mode::read> arr[2];
    cgh.single_task<class lambda_kernel>([=] {
      (void)w; (void)lm; (void)p; (void)n; (void)s; (void)h; (void)arr;
    });
  });
}

// A functor kernel object is named after its class, and its fields after
// their declarations.
struct Functor {
  int *out;
  int scale;
  void operator()() const { out[0] = scale; }
};

// CHECK-LABEL: cir.func {{.*}}@{{.*}}functor_kernel(
// CHECK-SAME: %arg0: {{.*}}{cir.sycl_kernel_param = {field = "out", kind = "pointer", path = ["out"]}
// CHECK-SAME: %arg1: {{.*}}{cir.sycl_kernel_param = {field = "scale", kind = "scalar", path = ["scale"]}
// CHECK: cir.alloca "Functor" {{.*}} {cir.sycl_kernel_object}
void functor(int *out) {
  kernel<class functor_kernel>(Functor{out, 2});
}

// LLVM-NOT: sycl_kernel_param
// LLVM-NOT: sycl_kernel_object
