// REQUIRES: aspect-usm_shared_allocations
// UNSUPPORTED: target-amd
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/16072

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// XFAIL: *
// XFAIL-TRACKER: FFK-KHR-prototype-step7-must-match-gap

// FFK KHR prototype, Step 7 -- end-to-end NEGATIVE must-match verification.
//
// A kernel decorated with work_group_size<32> launched through the sycl::khr
// free function kernel launcher with a MISMATCHED local size (64 != 32) SHOULD
// throw sycl::exception(errc::nd_range) from the existing runtime reqd-WG-size
// check (sycl/source/detail/error_handling/error_handling.cpp:151-163).
//
// *** KNOWN GAP -- this is why the test is XFAIL. ***
// The kernel_function<Func> launch path (BOTH sycl::khr AND the experimental
// extension it forwards to) enqueues a WRAPPER kernel
// (experimental::detail::NdRangeFreeFunctionKernelWrapper<&vadd,...>,
// enqueue_functions.hpp:444-462) that wraps Func in a lambda. The decoration's
// reqd_work_group_size is emitted only on `vadd` and its native
// `__sycl_kernel_vadd` entry; the WRAPPER device symbol carries NO
// !reqd_work_group_size metadata and NO "sycl-work-group-size" attribute, so
// the runtime reads CompileWGSize=0 and skips the must-match check -> no throw.
// The NATIVE by-name path (handler::parallel_for with the kernel object) DOES
// enforce it. This defect is UPSTREAM of the khr forward, not in the khr path.
// See playground/ffk-proto-step-7-RESULT.md for full IR/runtime evidence.
//
// When the wrapper-codegen gap is fixed this test will start throwing as
// expected and flip to XPASS, signalling that the XFAIL marker should be
// removed.

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
