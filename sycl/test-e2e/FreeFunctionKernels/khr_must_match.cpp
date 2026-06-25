// REQUIRES: aspect-usm_shared_allocations
// UNSUPPORTED: target-amd
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/16072

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20142

// FFK KHR prototype, Step 7 -- end-to-end POSITIVE verification that a kernel
// decorated with a required work_group_size, launched through the sycl::khr
// free function kernel launcher (khr::nd_launch -> experimental::nd_launch ->
// SYCL runtime), RUNS on a real device and produces correct output when the
// launch local size MATCHES the decorated work_group_size<32>.
//
// The corresponding NEGATIVE must-match check (mismatched local size should
// throw errc::nd_range) is in khr_must_match_negative.cpp; it is currently
// XFAIL because the kernel_function<Func> launch path enqueues a wrapper kernel
// that does not carry the reqd_work_group_size decoration -- see that file and
// playground/ffk-proto-step-7-RESULT.md.

#include <cassert>
#include <cstddef>

#include <sycl/khr/free_kernel.hpp>
#include <sycl/khr/work_item_queries.hpp>
#include <sycl/usm.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr std::size_t N = 256;
static constexpr std::size_t WGSIZE = 32;

// A required work-group-size decorated nd-range kernel. The khr decoration
// macro takes the kind (khr::nd_kernel<1>) plus the work_group_size requirement
// (the experimental property -- khr has no separate spelling yet).
SYCL_KHR_KERNEL(sycl::khr::nd_kernel<1>, syclexp::work_group_size<WGSIZE>)
void vadd(const float *a, const float *b, float *c) {
  std::size_t i = sycl::khr::this_nd_item<1>().get_global_linear_id();
  c[i] = a[i] + b[i];
}

int main() {
  sycl::queue q;

  float *a = sycl::malloc_shared<float>(N, q);
  float *b = sycl::malloc_shared<float>(N, q);
  float *c = sycl::malloc_shared<float>(N, q);
  for (std::size_t i = 0; i < N; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(2 * i);
    c[i] = 0.0f;
  }

  // POSITIVE: matching local size (32) runs + correct result through the khr
  // launcher forward.
  sycl::khr::nd_launch(
      q, sycl::nd_range<1>{sycl::range<1>{N}, sycl::range<1>{WGSIZE}},
      sycl::khr::kernel_function<vadd>, a, b, c);
  q.wait();

  for (std::size_t i = 0; i < N; ++i) {
    assert(c[i] == a[i] + b[i] &&
           "khr FFK positive launch produced wrong result");
  }

  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(c, q);
  return 0;
}
