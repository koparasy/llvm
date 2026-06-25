// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
//
// Negative companion to khr_checked_launch.cpp. Each launch below violates a
// compile-time-knowable property of the kernel it targets and must be rejected
// by a static_assert in <sycl/khr/launch.hpp>:
//   * wrong dimensionality (nd_kernel<1> launched with nd_range<2>)
//   * single_task kernel launched via nd_launch
//   * nd kernel launched via single_task
//   * invalid argument type reaching the launcher
// The checks use the Step-4 host-readable property traits, so this is a plain
// -fsycl host compile -- no integration header is needed for the diagnostics.

#include <sycl/khr/free_kernel.hpp>
#include <sycl/sycl.hpp>

namespace khr = sycl::khr;

SYCL_KHR_KERNEL(khr::nd_kernel<1>)
void vadd(float *a, float *b, float *c) {}

SYCL_KHR_KERNEL(khr::single_task_kernel)
void init(float *p) {}

// Not device-copyable: a user-provided copy constructor makes Bad neither
// trivially copyable nor implicitly device copyable, and it is not opted in via
// is_device_copyable, so is_valid_kernel_arg_v<Bad> is false. Dimensionality
// matches the kernel here, so the argument-type static_assert is the one that
// fires.
struct Bad {
  Bad(const Bad &) {}
  int x;
};

SYCL_KHR_KERNEL(khr::nd_kernel<1>)
void kbad(Bad b) {}

void run(sycl::queue q, float *a, float *b, float *c, float *p, Bad bad) {
  // wrong dimensionality: vadd is nd_kernel<1>, launched with nd_range<2>.
  // expected-error@*{{dimensionality does not match}}
  // expected-note@*{{requested here}}
  khr::nd_launch(q, sycl::nd_range<2>{{64, 64}, {8, 8}},
                 khr::kernel_function<vadd>, a, b, c);

  // single_task kernel launched via nd_launch (is_nd_kernel_v<init,1> false).
  // expected-error@*{{dimensionality does not match}}
  // expected-note@*{{requested here}}
  khr::nd_launch(q, sycl::nd_range<1>{1024, 32}, khr::kernel_function<init>, p);

  // nd kernel launched via single_task (is_single_task_kernel_v<vadd> false).
  // expected-error@*{{single_task kernel}}
  // expected-note@*{{requested here}}
  khr::single_task(q, khr::kernel_function<vadd>, a, b, c);

  // invalid argument type reaching the launcher.
  // expected-error@*{{not a valid free function kernel argument}}
  // expected-note@*{{requested here}}
  khr::nd_launch(q, sycl::nd_range<1>{1024, 32}, khr::kernel_function<kbad>,
                 bad);
}
