// RUN: %clangxx -fsycl                   -fsyntax-only %s
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only %s

// Over-rejection guard for the __is_valid_sycl_kernel_arg layout-unstable-scalar
// walk. The walk rejects a RAW builtin scalar (e.g. bare _Float16) because its
// representation is not guaranteed stable across toolchains. But a SYCL library
// numeric wrapper (sycl::half, sycl::vec, sycl::marray) has an
// implementation-guaranteed representation and MUST stay valid -- even though
// sycl::half wraps _Float16 internally on the device pass. This test uses the
// real SYCL headers (unlike the clang SemaSYCL test) and runs in BOTH passes.
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
