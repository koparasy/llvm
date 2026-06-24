// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
//
// Negative companion to khr_launch_properties.cpp. A COMPILE-TIME (decoration)
// property passed at launch must be REJECTED -- not silently ignored -- by the
// runtime-only static_assert in <sycl/khr/launch.hpp>. Decoration properties
// (work_group_size, kind/dim, ...) are the kernel's source of truth and belong
// on the SYCL_KHR_KERNEL(...) decoration, never at the launch site.
//
// This is a plain -fsycl host compile: the rejection is a pure header-side
// static_assert over the property list, no integration header needed.

#include <sycl/khr/free_kernel.hpp>
#include <sycl/sycl.hpp>

namespace khr = sycl::khr;
namespace exp = sycl::ext::oneapi::experimental;

SYCL_KHR_KERNEL(khr::nd_kernel<1>)
void vadd(float *a, float *b, float *c) {}

SYCL_KHR_KERNEL(khr::single_task_kernel)
void init(float *p) {}

void run(sycl::queue q, float *a, float *b, float *c, float *p) {
  // work_group_size is a COMPILE-TIME (decoration) property -- must NOT be
  // accepted at launch.
  // expected-error@*{{only runtime launch properties}}
  // expected-note@*{{requested here}}
  khr::nd_launch(q, sycl::nd_range<1>{1024, 32},
                 khr::properties{exp::work_group_size<32>},
                 khr::kernel_function<vadd>, a, b, c);

  // Same rejection for single_task with a compile-time property.
  // expected-error@*{{only runtime launch properties}}
  // expected-note@*{{requested here}}
  khr::single_task(q, khr::properties{exp::sub_group_size<16>},
                   khr::kernel_function<init>, p);
}
