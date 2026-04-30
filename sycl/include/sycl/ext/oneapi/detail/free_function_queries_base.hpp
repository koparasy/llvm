//==---- free_function_queries_base.hpp --- Core free function queries -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lightweight definitions of the SYCL_INTEL_free_function_queries extension's
// core queries (get_nd_item, get_work_group, get_sub_group). Intended to be
// included by headers that need the full set of three queries without pulling
// in the deprecated this_* wrappers or their dependencies.
//
// The full extension header, <sycl/ext/oneapi/free_function_queries.hpp>,
// includes this header too so there is a single source of truth for these
// three definitions.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/helpers.hpp>        // for Builder, declptr
#include <sycl/detail/nd_item_base.hpp>   // for nd_item (+ transitively group)
#include <sycl/detail/sub_group_base.hpp> // for sub_group

#ifndef __SYCL_DEVICE_ONLY__
#include <sycl/exception.hpp> // for exception, make_error_code, errc
#endif

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::this_work_item {

template <int Dimensions> nd_item<Dimensions> get_nd_item() {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::Builder::getElement(
      sycl::detail::declptr<nd_item<Dimensions>>());
#else
  throw sycl::exception(
      sycl::make_error_code(sycl::errc::feature_not_supported),
      "Free function calls are not supported on host");
#endif
}

template <int Dimensions> group<Dimensions> get_work_group() {
  return get_nd_item<Dimensions>().get_group();
}

inline sycl::sub_group get_sub_group() {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::sub_group();
#else
  throw sycl::exception(
      sycl::make_error_code(sycl::errc::feature_not_supported),
      "Free function calls are not supported on host");
#endif
}

} // namespace ext::oneapi::this_work_item
} // namespace _V1
} // namespace sycl
