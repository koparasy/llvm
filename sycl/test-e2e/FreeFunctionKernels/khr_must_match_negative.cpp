// REQUIRES: aspect-usm_shared_allocations
// UNSUPPORTED: target-amd
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/16072

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// FFK KHR prototype, Step 8 -- end-to-end NEGATIVE must-match verification.
//
// A kernel decorated with work_group_size<32> launched through the sycl::khr
// free function kernel launcher with a MISMATCHED local size (64 != 32) throws
// sycl::exception(errc::nd_range) from the existing runtime reqd-WG-size check
// (sycl/source/detail/error_handling/error_handling.cpp:151-163).
//
// Step 7 found that the kernel_function<Func> launch path enqueues a WRAPPER
// kernel (experimental::detail::NdRangeFreeFunctionKernelWrapper<&vadd,...>,
// enqueue_functions.hpp:444-462) that wraps Func in a lambda, and the wrapper
// did NOT carry the free function's decoration -> the runtime read
// CompileWGSize=0 and skipped the must-match check (the test was XFAIL).
//
// Step 8 (Option 1) propagates the free function's decoration onto the enqueued
// wrapper kernel in Sema (collectSYCLAttributes / VisitCallNode), so the wrapper
// now carries "sycl-work-group-size"="32" and !reqd_work_group_size {32,1,1}.
// The runtime must-match check now fires and the mismatched local size throws,
// so this test PASSES (XFAIL removed).

#include <cassert>
#include <cstddef>

#include <sycl/exception.hpp>
#include <sycl/khr/free_kernel.hpp>
#include <sycl/khr/work_item_queries.hpp>
#include <sycl/usm.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

static constexpr std::size_t N = 256;
static constexpr std::size_t WGSIZE = 32;

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

  // NEGATIVE: mismatched local size (64 != decorated 32) MUST throw
  // errc::nd_range from the existing runtime must-match check.
  bool threw = false;
  try {
    sycl::khr::nd_launch(
        q, sycl::nd_range<1>{sycl::range<1>{N}, sycl::range<1>{64}},
        sycl::khr::kernel_function<vadd>, a, b, c);
    q.wait();
  } catch (const sycl::exception &e) {
    threw = true;
    assert(e.code() == sycl::errc::nd_range &&
           "must-match mismatch should report errc::nd_range");
  }
  assert(threw && "expected a must-match exception for a mismatched local size");

  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(c, q);
  return 0;
}
