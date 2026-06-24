// RUN: %clangxx -fsycl -fsyntax-only %s
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only %s
//
// This test exercises the sycl::khr::is_valid_kernel_arg_v<T> trait surfaced in
// <sycl/khr/kernel_arg_traits.hpp>. The trait is backed by the clang builtin
// __is_valid_sycl_kernel_arg. It depends on the SYCL runtime headers but NOT on
// Step 1's <sycl/khr/free_kernel.hpp>; it stays independent of that work.
//
// The second RUN line (-fsycl-device-only) is load-bearing: in the device pass
// sycl::accessor / local_accessor ARE trivially copyable, so the
// trivially-copyable ceiling alone does NOT reject them -- they slip through and
// are rejected only by the builtin's explicit isSyclType exclusion. Without this
// RUN line the accessor static_asserts below would still pass on host (where the
// types are not trivially copyable), masking any regression that drops the
// device-side exclusion.

#include <sycl/khr/kernel_arg_traits.hpp>
#include <sycl/sycl.hpp>

using sycl::khr::is_valid_kernel_arg_v;

// Scalars and pointers are valid kernel arguments.
static_assert(is_valid_kernel_arg_v<int>);
static_assert(is_valid_kernel_arg_v<float *>);

// A trivially-copyable POD aggregate is valid.
struct Pod {
  int a;
  float b;
};
static_assert(is_valid_kernel_arg_v<Pod>);

// SYCL accessor / local_accessor must be rejected in BOTH passes. On host they
// are not trivially copyable (ceiling rejects them); in the device pass they ARE
// trivially copyable and are rejected only by the builtin's explicit isSyclType
// exclusion. The -fsycl-device-only RUN line above guards the latter.
static_assert(!is_valid_kernel_arg_v<sycl::accessor<int, 1>>);
static_assert(!is_valid_kernel_arg_v<sycl::local_accessor<int, 1>>);

int main() { return 0; }
