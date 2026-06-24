// RUN: %clangxx -fsycl -fsyntax-only %s
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only %s
//
// This test exercises the checked sycl::khr free function kernel launch surface
// surfaced in <sycl/khr/free_kernel.hpp> (Step 5):
//   sycl::khr::kernel_function<Func>
//   sycl::khr::nd_launch(queue|handler, nd_range<Dims>, kernel_function<F>, ...)
//   sycl::khr::single_task(queue|handler, kernel_function<F>, ...)
//
// The launchers add a compile-time gate -- kernel kind + dimensionality
// (is_nd_kernel_v / is_single_task_kernel_v) and argument-type validity
// (is_valid_kernel_arg_v) static_asserts -- in front of the experimental free
// function kernel launchers, then forward to them. The checks read the Step-4
// host-readable property traits, so the FIRST (plain -fsycl host) RUN line
// proves they evaluate at host parse time WITHOUT the integration header.
//
// This file covers the POSITIVE (well-formed) launches; the wrong-kind /
// wrong-dimensionality / bad-argument-type rejections live in the companion
// khr_checked_launch_errors.cpp -verify test.

#include <sycl/khr/free_kernel.hpp>
#include <sycl/sycl.hpp>

namespace khr = sycl::khr;

SYCL_KHR_KERNEL(khr::nd_kernel<1>)
void vadd(float *a, float *b, float *c) {}

SYCL_KHR_KERNEL(khr::single_task_kernel)
void init(float *p) {}

template <typename T>
SYCL_KHR_KERNEL(khr::nd_kernel<2>)
void mul(T *a, T *b) {}

void run_queue(sycl::queue q, float *a, float *b, float *c, float *p, int *ia,
               int *ib) {
  khr::nd_launch(q, sycl::nd_range<1>{1024, 32}, khr::kernel_function<vadd>, a,
                 b, c);
  khr::single_task(q, khr::kernel_function<init>, p);
  khr::nd_launch(q, sycl::nd_range<2>{{64, 64}, {8, 8}},
                 khr::kernel_function<mul<int>>, ia, ib);
}

void run_handler(sycl::queue q, float *a, float *b, float *c, float *p,
                 int *ia, int *ib) {
  q.submit([&](sycl::handler &cgh) {
    khr::nd_launch(cgh, sycl::nd_range<1>{1024, 32}, khr::kernel_function<vadd>,
                   a, b, c);
  });
  q.submit([&](sycl::handler &cgh) {
    khr::single_task(cgh, khr::kernel_function<init>, p);
  });
  q.submit([&](sycl::handler &cgh) {
    khr::nd_launch(cgh, sycl::nd_range<2>{{64, 64}, {8, 8}},
                   khr::kernel_function<mul<int>>, ia, ib);
  });
}
