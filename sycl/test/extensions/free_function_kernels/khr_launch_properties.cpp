// RUN: %clangxx -fsycl -fsyntax-only %s
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only %s
//
// POSITIVE companion for the Step-6 launch-property-list overloads of the
// checked sycl::khr free function kernel launch surface:
//   sycl::khr::nd_launch(queue|handler, nd_range<Dims>, properties<...>,
//                        kernel_function<F>, ...)
//   sycl::khr::single_task(queue|handler, properties<...>,
//                          kernel_function<F>, ...)
//
// Step 6 adds a SECOND overload to each Step-5 launcher: one that takes a
// property list (constrained via is_property_list_v) BETWEEN the range/queue
// and the kernel-function handle. The property list may carry ONLY runtime
// (launch-time) properties; compile-time / decoration properties are rejected
// by a static_assert (that rejection is covered by the -verify companion
// khr_launch_properties_errors.cpp).
//
// In v1 the only in-KHR-scope runtime launch property (work_group_scratch_size)
// is deferred, so the realistic exercised list is EMPTY -- that is fine: the
// overload's value is the forward-compatible slot + the reject of misplaced
// compile-time props. This file proves:
//   (1) an EMPTY property list compiles and launches (nd_launch + single_task);
//   (2) the with-properties and without-properties forms COEXIST with no
//       ambiguous-call error (overload disambiguation);
//   (3) a RUNTIME launch property (work_group_scratch_size) is ACCEPTED by the
//       runtime-only static_assert (nd_launch only -- single_task has no
//       prop-carrying forwardee yet).
//
// NOTE: a distinct kernel is used per (kernel, launch-form) call site. The
// integration-header generator emits one KernelInfo per kernel-name wrapper and
// double-emits if the SAME (Func, launch-form) wrapper appears at more than one
// call site in a TU; using a fresh kernel per site keeps each wrapper unique.

#include <sycl/ext/oneapi/work_group_scratch_memory.hpp>
#include <sycl/khr/free_kernel.hpp>
#include <sycl/sycl.hpp>

namespace khr = sycl::khr;
namespace exp = sycl::ext::oneapi::experimental;

SYCL_KHR_KERNEL(khr::nd_kernel<1>) void k_empty_q(float *a) {}
SYCL_KHR_KERNEL(khr::nd_kernel<1>) void k_noprops_q(float *a) {}
SYCL_KHR_KERNEL(khr::nd_kernel<1>) void k_scratch_q(float *a) {}
SYCL_KHR_KERNEL(khr::nd_kernel<1>) void k_empty_h(float *a) {}
SYCL_KHR_KERNEL(khr::nd_kernel<1>) void k_noprops_h(float *a) {}
SYCL_KHR_KERNEL(khr::nd_kernel<1>) void k_scratch_h(float *a) {}

SYCL_KHR_KERNEL(khr::single_task_kernel) void st_empty_q(float *p) {}
SYCL_KHR_KERNEL(khr::single_task_kernel) void st_empty_h(float *p) {}

template <typename T>
SYCL_KHR_KERNEL(khr::nd_kernel<2>) void mul(T *a, T *b) {}

void run_queue(sycl::queue q, float *a, float *b, int *ia, int *ib) {
  const sycl::nd_range<1> r{1024, 32};

  // (1) empty property list -- nd_launch + single_task, queue form.
  khr::nd_launch(q, r, khr::properties{}, khr::kernel_function<k_empty_q>, a);
  khr::single_task(q, khr::properties{}, khr::kernel_function<st_empty_q>, a);

  // templated kernel through the property overload.
  khr::nd_launch(q, sycl::nd_range<2>{{1024, 1024}, {8, 8}}, khr::properties{},
                 khr::kernel_function<mul<int>>, ia, ib);

  // (2) disambiguation: the no-properties Step-5 form still resolves in the
  // SAME TU, no ambiguity.
  khr::nd_launch(q, r, khr::kernel_function<k_noprops_q>, a);

  // (3) a RUNTIME launch property is accepted by the runtime-only assert.
  khr::nd_launch(q, r, khr::properties{exp::work_group_scratch_size(256)},
                 khr::kernel_function<k_scratch_q>, a);
}

void run_handler(sycl::queue q, float *a) {
  const sycl::nd_range<1> r{1024, 32};
  q.submit([&](sycl::handler &cgh) {
    // empty property list -- nd_launch + single_task, handler form.
    khr::nd_launch(cgh, r, khr::properties{}, khr::kernel_function<k_empty_h>,
                   a);
    khr::single_task(cgh, khr::properties{}, khr::kernel_function<st_empty_h>,
                     a);

    // no-properties form coexisting in the same scope.
    khr::nd_launch(cgh, r, khr::kernel_function<k_noprops_h>, a);

    // runtime launch property accepted (handler form).
    khr::nd_launch(cgh, r, khr::properties{exp::work_group_scratch_size(256)},
                   khr::kernel_function<k_scratch_h>, a);
  });
}
