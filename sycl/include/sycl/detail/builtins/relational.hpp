//==------------------- relational.hpp -------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/builtins/base.hpp>
#include <sycl/detail/loop.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

// Relation builtins widen signed-char masks to the required integer element
// type. Keep that conversion local to the relational path so shared builtin
// infrastructure is only carrying helpers used across families.
template <typename NewElemT, int N>
vec<NewElemT, N> relational_mask_widen(vec<signed char, N> X) {
	static_assert(is_scalar_arithmetic_v<NewElemT>);

#ifdef __SYCL_DEVICE_ONLY__
	if constexpr (N > 1) {
		using src_vector_t = signed char __attribute__((ext_vector_type(N)));
		using dst_vector_t = NewElemT __attribute__((ext_vector_type(N)));
		auto OpenCLVec = bit_cast<src_vector_t>(X);
		return bit_cast<vec<NewElemT, N>>(
				__builtin_convertvector(OpenCLVec, dst_vector_t));
	}
#endif

	vec<NewElemT, N> Result{};
	detail::loop<N>([&](size_t I) { Result[I] = static_cast<NewElemT>(X[I]); });
	return Result;
}

} // namespace detail
} // namespace _V1
} // namespace sycl

#include <sycl/detail/builtins/relational_functions.inc>
