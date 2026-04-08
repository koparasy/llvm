//==---- free_function_queries.hpp -- SYCL_INTEL_free_function_queries ext -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/work_item_context.hpp>
#include <sycl/group.hpp>
#include <sycl/nd_item.hpp>
#include <sycl/sub_group.hpp>

// For deprecated queries:
#include <sycl/id.hpp>
#include <sycl/item.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail::work_item_context {

template <int Dimensions, typename QueryT>
inline id<Dimensions> make_id(const QueryT &Query) {
  if constexpr (Dimensions == 1) {
    return {Query(0)};
  } else if constexpr (Dimensions == 2) {
    return {Query(0), Query(1)};
  } else {
    return {Query(0), Query(1), Query(2)};
  }
}

template <int Dimensions, typename QueryT>
inline range<Dimensions> make_range(const QueryT &Query) {
  if constexpr (Dimensions == 1) {
    return {Query(0)};
  } else if constexpr (Dimensions == 2) {
    return {Query(0), Query(1)};
  } else {
    return {Query(0), Query(1), Query(2)};
  }
}

} // namespace detail::work_item_context

namespace ext::oneapi::this_work_item {
template <int Dimensions> nd_item<Dimensions> get_nd_item() {
  static_assert(sycl::detail::work_item_context::is_valid_dimensions<Dimensions>,
                "invalid dimensions");
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::Builder::createNDItem<Dimensions>();
#else
  sycl::detail::work_item_context::throw_unsupported_query(
      "Free function calls are not supported on host");
#endif
}

template <int Dimensions> group<Dimensions> get_work_group() {
  static_assert(sycl::detail::work_item_context::is_valid_dimensions<Dimensions>,
                "invalid dimensions");
#ifdef __SYCL_DEVICE_ONLY__
  range<Dimensions> GlobalRange = sycl::detail::work_item_context::make_range<Dimensions>(
      [](int Dimension) {
        return sycl::detail::work_item_context::get_global_range<Dimensions>(
            Dimension);
      });
  range<Dimensions> LocalRange = sycl::detail::work_item_context::make_range<Dimensions>(
      [](int Dimension) {
        return sycl::detail::work_item_context::get_local_range<Dimensions>(
            Dimension);
      });
  range<Dimensions> GroupRange = sycl::detail::work_item_context::make_range<Dimensions>(
      [](int Dimension) {
        return sycl::detail::work_item_context::get_group_range<Dimensions>(
            Dimension);
      });
  id<Dimensions> GroupId = sycl::detail::work_item_context::make_id<Dimensions>(
      [](int Dimension) {
        return sycl::detail::work_item_context::get_group_id<Dimensions>(
            Dimension);
      });
  return sycl::detail::Builder::createGroup<Dimensions>(
      GlobalRange, LocalRange, GroupRange, GroupId);
#else
  sycl::detail::work_item_context::throw_unsupported_query(
      "Free function calls are not supported on host");
#endif
}

inline sycl::sub_group get_sub_group() {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::sub_group();
#else
  sycl::detail::work_item_context::throw_unsupported_query(
      "Free function calls are not supported on host");
#endif
}
} // namespace ext::oneapi::this_work_item

namespace ext::oneapi::experimental {
template <int Dims>
__SYCL_DEPRECATED(
    "use sycl::ext::oneapi::this_work_item::get_nd_item() instead")
nd_item<Dims> this_nd_item() {
  return ext::oneapi::this_work_item::get_nd_item<Dims>();
}

template <int Dims>
__SYCL_DEPRECATED(
    "use sycl::ext::oneapi::this_work_item::get_work_group() instead")
group<Dims> this_group() {
  return ext::oneapi::this_work_item::get_work_group<Dims>();
}

__SYCL_DEPRECATED(
    "use sycl::ext::oneapi::this_work_item::get_sub_group() instead")
inline sycl::sub_group this_sub_group() {
  return ext::oneapi::this_work_item::get_sub_group();
}

template <int Dims>
__SYCL_DEPRECATED("use nd_range kernel and "
                  "sycl::ext::oneapi::this_work_item::get_nd_item() instead")
item<Dims> this_item() {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::Builder::getElement(sycl::detail::declptr<item<Dims>>());
#else
  throw sycl::exception(
      sycl::make_error_code(sycl::errc::feature_not_supported),
      "Free function calls are not supported on host");
#endif
}

template <int Dims>
__SYCL_DEPRECATED("use nd_range kernel and "
                  "sycl::ext::oneapi::this_work_item::get_nd_item() instead")
id<Dims> this_id() {
  return this_item<Dims>().get_id();
}
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
