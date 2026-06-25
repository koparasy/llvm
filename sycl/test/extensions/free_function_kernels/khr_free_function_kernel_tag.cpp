// RUN: %clangxx -fsycl -fsyntax-only %s
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only %s
//
// This test exercises the free_function_kernel PROPERTY TAG surfaced in
// <sycl/khr/free_kernel.hpp> (Sync A2). The tag is the concrete artifact the
// spec "Kernel properties" section introduces: an incomplete, never-instantiated
// tag type `sycl::khr::free_function_kernel` that stands in as the `Class`
// operand of sycl_khr_properties' `is_property_for<P, Class>` applicability
// trait (a free function kernel has no class object of its own to key on).
//
// #980-BLOCKED: the spec keys property applicability on
// `is_property_for_v<P, free_function_kernel>` from sycl_khr_properties (#980),
// which is NOT in this tree. So this test ONLY checks that the tag TYPE EXISTS
// and is usable as a type name (it is incomplete, so it is named, never
// instantiated). It does NOT test is_property_for (not available pre-#980).

#include <sycl/khr/free_kernel.hpp>
#include <sycl/sycl.hpp>

#include <type_traits>

namespace khr = sycl::khr;

// The tag type exists and is namable. It is incomplete, so use only traits that
// do NOT require a complete type.

// It is a class type.
static_assert(std::is_class_v<khr::free_function_kernel>);

// It is incomplete -- never instantiated, no size required. We can still form a
// pointer/reference to it and name it in unevaluated contexts.
using tag_ptr = khr::free_function_kernel *;
static_assert(std::is_pointer_v<tag_ptr>);
static_assert(
    std::is_same_v<std::remove_pointer_t<tag_ptr>, khr::free_function_kernel>);

int main() { return 0; }
