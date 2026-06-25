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
// Trivial-copyability is NECESSARY but NOT SUFFICIENT: an implementation may
// reject even a trivially-copyable type whose representation it cannot
// guarantee stable across the host/device toolchain boundary. A pure-header
// trait cannot introspect struct members (no portable reflection pre-C++26),
// so the trait is backed by the clang type-trait builtin
// __is_valid_sycl_kernel_arg, which Sema folds to a bool constant expression.

namespace sycl {
inline namespace _V1 {
namespace khr {

template <typename T>
inline constexpr bool is_valid_kernel_arg_v = __is_valid_sycl_kernel_arg(T);

} // namespace khr
} // namespace _V1
} // namespace sycl
