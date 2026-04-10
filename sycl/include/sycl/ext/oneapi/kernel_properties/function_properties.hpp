//==--- function_properties.hpp - SYCL standalone function properties -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename PropertyT> struct FunctionPropertyMetaInfo;

} // namespace detail

template <int Dims> struct nd_range_kernel_t {
  static_assert(Dims >= 1 && Dims <= 3,
                "nd_range_kernel must use dimension 1, 2, or 3.");

  static constexpr int dimensions = Dims;
};

struct single_task_kernel_t {};

template <int Dims>
inline constexpr nd_range_kernel_t<Dims> nd_range_kernel;

inline constexpr single_task_kernel_t single_task_kernel;

namespace detail {

template <int Dims>
struct FunctionPropertyMetaInfo<nd_range_kernel_t<Dims>> {
  static constexpr const char *name = "sycl-nd-range-kernel";
  static constexpr int value = Dims;
};

template <> struct FunctionPropertyMetaInfo<single_task_kernel_t> {
  static constexpr const char *name = "sycl-single-task-kernel";
  static constexpr int value = 0;
};

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(PROP)                                \
  [[__sycl_detail__::add_ir_attributes_function(                               \
      sycl::ext::oneapi::experimental::detail::FunctionPropertyMetaInfo<        \
          sycl::ext::oneapi::experimental::detail::remove_cvref_t<              \
              decltype(PROP)>>::name,                                           \
      sycl::ext::oneapi::experimental::detail::FunctionPropertyMetaInfo<        \
          sycl::ext::oneapi::experimental::detail::remove_cvref_t<              \
              decltype(PROP)>>::value)]]
#else
#define SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(PROP)
#endif