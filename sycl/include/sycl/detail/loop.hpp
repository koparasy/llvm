//==---------------- loop.hpp - SYCL loop helpers -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

namespace sycl {
inline namespace _V1 {
namespace detail {

template <size_t... Inds, class F>
constexpr void loop_impl(std::integer_sequence<size_t, Inds...>, F &&f) {
  (f(std::integral_constant<size_t, Inds>{}), ...);
}

template <size_t Count, class F> constexpr void loop(F &&f) {
  loop_impl(std::make_index_sequence<Count>{}, std::forward<F>(f));
}

} // namespace detail
} // namespace _V1
} // namespace sycl