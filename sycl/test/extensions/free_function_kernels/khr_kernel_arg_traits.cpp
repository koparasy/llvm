// RUN: %clangxx -fsycl                    -fsyntax-only %s
// RUN: %clangxx -fsycl -fsycl-device-only  -fsyntax-only %s
//
// This test exercises the sycl::khr::is_valid_kernel_arg_v<T> trait surfaced in
// <sycl/khr/kernel_arg_traits.hpp>. Per the KHR spec ("Restrictions on kernel
// argument types"), a free function kernel argument must be <<device-copyable>>
// -- the same rule as any other SYCL kernel argument. The trait is therefore an
// ALIAS of the core sycl::is_device_copyable_v and pulls in no FFK-specific
// machinery (it does NOT depend on Step 1's <sycl/khr/free_kernel.hpp>).
//
// NOTE on the accessor cases below: sycl::accessor / local_accessor are NOT
// device-copyable on the HOST pass (not trivially copyable, no
// is_device_copyable specialization). In the isolated -fsycl-device-only pass
// they ARE trivially copyable in their device representation, so the core
// is_device_copyable_v reports true for them there -- a property of the CORE
// trait, not of this header. The stricter, accessor-rejecting behavior on the
// device pass is what the retained (off-spec) __is_valid_sycl_kernel_arg builtin
// provides; it is deliberately not on the spec path. So the accessor REJECTION
// asserts are guarded to the host pass (#ifndef __SYCL_DEVICE_ONLY__). Note both
// RUN lines spawn a device sub-compile under -fsycl, so the guard -- not a -D on
// the command line -- is what keeps the device pass clean.

#include <sycl/khr/kernel_arg_traits.hpp>
#include <sycl/sycl.hpp>

using sycl::khr::is_valid_kernel_arg_v;

// Scalars and pointers are valid kernel arguments.
static_assert(is_valid_kernel_arg_v<int>);
static_assert(is_valid_kernel_arg_v<float *>);

// A trivially-copyable POD aggregate is valid (trivially copyable => implicitly
// device copyable).
struct Pod {
  int a;
  float b;
};
static_assert(is_valid_kernel_arg_v<Pod>);

// A type with a user-provided copy constructor is NOT trivially copyable and is
// NOT device copyable, so it is rejected on BOTH passes.
struct UserCopy {
  UserCopy(const UserCopy &) {}
  int x;
};
static_assert(!is_valid_kernel_arg_v<UserCopy>);

// The same type can be opted in via the core SYCL_DEVICE_COPYABLE mechanism
// (specializing is_device_copyable). Once opted in it is a valid argument.
struct OptIn {
  OptIn(const OptIn &) {}
  int x;
};
template <> struct sycl::is_device_copyable<OptIn> : std::true_type {};
static_assert(is_valid_kernel_arg_v<OptIn>);

// SYCL accessor / local_accessor are special types passed non-positionally for
// ordinary kernels; a free function kernel receives parameters positionally and
// they are NOT device copyable, so they are ill formed as parameters. This is
// observable on the host pass (see the file header for why the device pass of
// the core trait reports them trivially copyable).
#ifndef __SYCL_DEVICE_ONLY__
static_assert(!is_valid_kernel_arg_v<sycl::accessor<int, 1>>);
static_assert(!is_valid_kernel_arg_v<sycl::local_accessor<int, 1>>);
#endif

int main() { return 0; }
