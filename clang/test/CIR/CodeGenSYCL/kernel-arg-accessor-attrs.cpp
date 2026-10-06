// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -internal-isystem %S/../../CodeGenSYCL/Inputs -fclangir -emit-llvm %s \
// RUN:   -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -internal-isystem %S/../../CodeGenSYCL/Inputs -emit-llvm %s \
// RUN:   -o - | FileCheck %s

// SYCL kernel parameters get the same attributes and metadata as in classic
// CodeGen: readonly on read-only accessor pointers, noalias on pointers under
// [[intel::kernel_args_restrict]] or the sycl-unaliased property, and the
// kernel_arg_runtime_aligned/kernel_arg_exclusive_ptr flags on accessor base
// pointers.

#include "sycl.hpp"

struct __attribute__((sycl_special_class))
[[__sycl_detail__::sycl_type(annotated_arg)]] AnnotatedIntPtr {
  void __init([[__sycl_detail__::add_ir_attributes_kernel_parameter(
      "sycl-unaliased", nullptr)]] __attribute__((opencl_global)) int *InPtr) {
    Ptr = InPtr;
  }
  int &operator[](unsigned I) const { return Ptr[I]; }
  __attribute__((opencl_global)) int *Ptr;
};

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}acc_kernel(ptr addrspace(1) noundef align 4 %{{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr addrspace(1) noundef readonly align 4 %{{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}) {{.*}}!kernel_arg_runtime_aligned ![[ACC:[0-9]+]] !kernel_arg_exclusive_ptr ![[ACC]]
void acc(sycl::queue &q, sycl::buffer<int, 1> &in, sycl::buffer<int, 1> &out) {
  q.submit([&](sycl::handler &cgh) {
    auto w = out.get_access<sycl::access::mode::write>(cgh);
    auto r = in.get_access<sycl::access::mode::read>(cgh);
    cgh.single_task<class acc_kernel>([w, r] {});
  });
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}acc_restrict_kernel(ptr addrspace(1) noalias noundef align 4 %{{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr addrspace(1) noalias noundef readonly align 4 %{{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, i32 noundef %{{.*}}) {{.*}}!kernel_arg_runtime_aligned ![[ACCN:[0-9]+]] !kernel_arg_exclusive_ptr ![[ACCN]]
void accRestrict(sycl::queue &q, sycl::buffer<int, 1> &in,
                 sycl::buffer<int, 1> &out, int n) {
  q.submit([&](sycl::handler &cgh) {
    auto w = out.get_access<sycl::access::mode::write>(cgh);
    auto r = in.get_access<sycl::access::mode::read>(cgh);
    cgh.single_task<class acc_restrict_kernel>(
        [w, r, n]() [[intel::kernel_args_restrict]] { (void)n; });
  });
}

void ptrs(int *a, int *b) {
  // CHECK: define {{.*}}spir_kernel void @{{.*}}ptr_restrict_kernel(ptr addrspace(1) noalias noundef align 4 %{{.*}}, ptr addrspace(1) noalias noundef align 4 %{{.*}})
  // CHECK-NOT: !kernel_arg_exclusive_ptr
  // CHECK-SAME: {
  kernel<class ptr_restrict_kernel>(
      [a, b]() [[intel::kernel_args_restrict]] { a[0] = b[0]; });

  // CHECK: define {{.*}}spir_kernel void @{{.*}}ptr_kernel(ptr addrspace(1) noundef align 4 %{{.*}}, ptr addrspace(1) noundef align 4 %{{.*}})
  kernel<class ptr_kernel>([a, b]() { a[0] = b[0]; });

  AnnotatedIntPtr u;
  // CHECK: define {{.*}}spir_kernel void @{{.*}}unaliased_kernel(ptr addrspace(1) noalias noundef align 4 {{.*}}%{{.*}}, ptr addrspace(1) noundef align 4 %{{.*}})
  kernel<class unaliased_kernel>([u, b]() { u[0] = b[0]; });
}

// CHECK-DAG: ![[ACC]] = !{i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false}
// CHECK-DAG: ![[ACCN]] = !{i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 false}
