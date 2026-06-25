//==--- kernel_arg_traits.hpp --- KHR free-function-kernel arg validity ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// Validity trait for arguments passed to a sycl::khr free function kernel.
//
// SPEC RULE (sycl_khr_free_function_kernels.adoc, "Restrictions on kernel
// argument types"): each free function kernel parameter must be
// <<device-copyable>> -- the SAME rule as every other SYCL kernel argument
// (<<sec:kernel.parameter.passing>>). A free function kernel receives every
// parameter positionally and does not get the special non-positional allowance
// for accessor / local_accessor / image accessors / stream / reducer /
// kernel_handler, so those (and any other non-device-copyable type) are ill
// formed as a parameter.
//
// We therefore reuse the core sycl::is_device_copyable_v trait directly rather
// than introduce a new kernel-arg notion: is_valid_kernel_arg_v is an ALIAS of
// is_device_copyable_v, so the launch.hpp call sites are unchanged.
//
// NOTE: the clang builtin __is_valid_sycl_kernel_arg (a stricter,
// cross-ABI-narrowing trait) is RETAINED in the tree as the off-spec flag-mode
// tier for the native-name / OpenVINO consumer, but it is NOT on the spec path
// and is deliberately not used here.

#include <sycl/detail/is_device_copyable.hpp>

namespace sycl {
inline namespace _V1 {
namespace khr {

template <typename T>
inline constexpr bool is_valid_kernel_arg_v = is_device_copyable_v<T>;

} // namespace khr
} // namespace _V1
} // namespace sycl
