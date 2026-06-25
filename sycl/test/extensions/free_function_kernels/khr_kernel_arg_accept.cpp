// RUN: %clangxx -fsycl                   -fsyntax-only %s
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only %s

// Acceptance guard: SYCL library numeric wrappers (sycl::half, sycl::vec,
// sycl::marray, and aggregates containing them) are valid free function kernel
// arguments under the spec rule -- each is <<device-copyable>> (trivially
// copyable, or device-copyable through its element type) so the
// is_device_copyable_v alias backing is_valid_kernel_arg_v reports true. These
// stay valid on BOTH passes. (The retained off-spec __is_valid_sycl_kernel_arg
// builtin has a stricter layout-unstable-scalar walk that needs an explicit
// half allowlist to accept these; that walk is NOT on the spec path here.) This
// test uses the real SYCL headers and runs in BOTH passes.
#include <sycl/sycl.hpp>

#include <sycl/khr/kernel_arg_traits.hpp>

using sycl::khr::is_valid_kernel_arg_v;

static_assert(is_valid_kernel_arg_v<sycl::half>);
static_assert(is_valid_kernel_arg_v<sycl::vec<float, 4>>);
static_assert(is_valid_kernel_arg_v<sycl::vec<int, 3>>);
static_assert(is_valid_kernel_arg_v<sycl::marray<float, 8>>);
static_assert(is_valid_kernel_arg_v<sycl::marray<double, 2>>);

// half nested in a user struct stays valid too:
struct HasSyclHalf {
  float f;
  sycl::half h;
};
static_assert(is_valid_kernel_arg_v<HasSyclHalf>);

// a vec of half:
static_assert(is_valid_kernel_arg_v<sycl::vec<sycl::half, 4>>);
