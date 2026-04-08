//==----------- work_item_context.hpp - SYCL work-item context ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>

#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/__spirv/spirv_vars.hpp>
#else
#include <sycl/exception.hpp>
#endif

#include <cstddef>
#include <cstdint>

namespace sycl {
inline namespace _V1 {
namespace detail::work_item_context {

template <int Dimensions>
inline constexpr bool is_valid_dimensions = (Dimensions > 0) && (Dimensions < 4);

inline constexpr bool is_device_execution_scope() {
#ifdef __SYCL_DEVICE_ONLY__
  return true;
#else
  return false;
#endif
}

#ifndef __SYCL_DEVICE_ONLY__
[[noreturn]] inline void throw_unsupported_query(const char *Message) {
  throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported),
                        Message);
}
#endif

template <int Dimensions>
__SYCL_ALWAYS_INLINE constexpr int get_spirv_dim(int Dimension) {
  return Dimensions - Dimension - 1;
}

template <int Dimensions>
__SYCL_ALWAYS_INLINE size_t get_global_id([[maybe_unused]] int Dimension) {
  static_assert(is_valid_dimensions<Dimensions>, "invalid dimensions");
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_BuiltInGlobalInvocationId(get_spirv_dim<Dimensions>(Dimension));
#else
  throw_unsupported_query("Free function calls are not supported on host");
#endif
}

template <int Dimensions>
__SYCL_ALWAYS_INLINE size_t get_global_range([[maybe_unused]] int Dimension) {
  static_assert(is_valid_dimensions<Dimensions>, "invalid dimensions");
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_BuiltInGlobalSize(get_spirv_dim<Dimensions>(Dimension));
#else
  throw_unsupported_query("Free function calls are not supported on host");
#endif
}

template <int Dimensions>
__SYCL_ALWAYS_INLINE size_t get_global_offset([[maybe_unused]] int Dimension) {
  static_assert(is_valid_dimensions<Dimensions>, "invalid dimensions");
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_BuiltInGlobalOffset(get_spirv_dim<Dimensions>(Dimension));
#else
  throw_unsupported_query("Free function calls are not supported on host");
#endif
}

template <int Dimensions>
__SYCL_ALWAYS_INLINE size_t get_local_id([[maybe_unused]] int Dimension) {
  static_assert(is_valid_dimensions<Dimensions>, "invalid dimensions");
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_BuiltInLocalInvocationId(get_spirv_dim<Dimensions>(Dimension));
#else
  throw_unsupported_query("Free function calls are not supported on host");
#endif
}

template <int Dimensions>
__SYCL_ALWAYS_INLINE size_t get_local_range([[maybe_unused]] int Dimension) {
  static_assert(is_valid_dimensions<Dimensions>, "invalid dimensions");
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_BuiltInWorkgroupSize(get_spirv_dim<Dimensions>(Dimension));
#else
  throw_unsupported_query("Free function calls are not supported on host");
#endif
}

template <int Dimensions>
__SYCL_ALWAYS_INLINE size_t get_group_id([[maybe_unused]] int Dimension) {
  static_assert(is_valid_dimensions<Dimensions>, "invalid dimensions");
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_BuiltInWorkgroupId(get_spirv_dim<Dimensions>(Dimension));
#else
  throw_unsupported_query("Free function calls are not supported on host");
#endif
}

template <int Dimensions>
__SYCL_ALWAYS_INLINE size_t get_group_range([[maybe_unused]] int Dimension) {
  static_assert(is_valid_dimensions<Dimensions>, "invalid dimensions");
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_BuiltInNumWorkgroups(get_spirv_dim<Dimensions>(Dimension));
#else
  throw_unsupported_query("Free function calls are not supported on host");
#endif
}

template <int Dimensions, typename IndexQueryT, typename RangeQueryT>
__SYCL_ALWAYS_INLINE size_t get_linear_id(IndexQueryT IndexQuery,
                                          RangeQueryT RangeQuery) {
  static_assert(is_valid_dimensions<Dimensions>, "invalid dimensions");
  if constexpr (Dimensions == 1) {
    return IndexQuery(0);
  } else if constexpr (Dimensions == 2) {
    return IndexQuery(0) * RangeQuery(1) + IndexQuery(1);
  } else {
    return IndexQuery(0) * RangeQuery(1) * RangeQuery(2) +
           IndexQuery(1) * RangeQuery(2) + IndexQuery(2);
  }
}

template <int Dimensions> __SYCL_ALWAYS_INLINE size_t get_global_linear_id() {
  static_assert(is_valid_dimensions<Dimensions>, "invalid dimensions");
  if constexpr (Dimensions == 1) {
    return get_global_id<Dimensions>(0) - get_global_offset<Dimensions>(0);
  } else if constexpr (Dimensions == 2) {
    return (get_global_id<Dimensions>(0) - get_global_offset<Dimensions>(0)) *
               get_global_range<Dimensions>(1) +
           get_global_id<Dimensions>(1) - get_global_offset<Dimensions>(1);
  } else {
    return (get_global_id<Dimensions>(0) - get_global_offset<Dimensions>(0)) *
               get_global_range<Dimensions>(1) *
               get_global_range<Dimensions>(2) +
           (get_global_id<Dimensions>(1) - get_global_offset<Dimensions>(1)) *
               get_global_range<Dimensions>(2) +
           get_global_id<Dimensions>(2) - get_global_offset<Dimensions>(2);
  }
}

template <int Dimensions> __SYCL_ALWAYS_INLINE size_t get_local_linear_id() {
  return get_linear_id<Dimensions>(
      [](int Dimension) { return get_local_id<Dimensions>(Dimension); },
      [](int Dimension) { return get_local_range<Dimensions>(Dimension); });
}

template <int Dimensions> __SYCL_ALWAYS_INLINE size_t get_group_linear_id() {
  return get_linear_id<Dimensions>(
      [](int Dimension) { return get_group_id<Dimensions>(Dimension); },
      [](int Dimension) { return get_group_range<Dimensions>(Dimension); });
}

template <int Dimensions> __SYCL_ALWAYS_INLINE size_t get_global_linear_range() {
  static_assert(is_valid_dimensions<Dimensions>, "invalid dimensions");
  if constexpr (Dimensions == 1) {
    return get_global_range<Dimensions>(0);
  } else if constexpr (Dimensions == 2) {
    return get_global_range<Dimensions>(0) * get_global_range<Dimensions>(1);
  } else {
    return get_global_range<Dimensions>(0) * get_global_range<Dimensions>(1) *
           get_global_range<Dimensions>(2);
  }
}

template <int Dimensions> __SYCL_ALWAYS_INLINE size_t get_local_linear_range() {
  static_assert(is_valid_dimensions<Dimensions>, "invalid dimensions");
  if constexpr (Dimensions == 1) {
    return get_local_range<Dimensions>(0);
  } else if constexpr (Dimensions == 2) {
    return get_local_range<Dimensions>(0) * get_local_range<Dimensions>(1);
  } else {
    return get_local_range<Dimensions>(0) * get_local_range<Dimensions>(1) *
           get_local_range<Dimensions>(2);
  }
}

template <int Dimensions> __SYCL_ALWAYS_INLINE size_t get_group_linear_range() {
  static_assert(is_valid_dimensions<Dimensions>, "invalid dimensions");
  if constexpr (Dimensions == 1) {
    return get_group_range<Dimensions>(0);
  } else if constexpr (Dimensions == 2) {
    return get_group_range<Dimensions>(0) * get_group_range<Dimensions>(1);
  } else {
    return get_group_range<Dimensions>(0) * get_group_range<Dimensions>(1) *
           get_group_range<Dimensions>(2);
  }
}

inline __SYCL_ALWAYS_INLINE uint32_t get_sub_group_local_id() {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_BuiltInSubgroupLocalInvocationId();
#else
  throw_unsupported_query("Free function calls are not supported on host");
#endif
}

inline __SYCL_ALWAYS_INLINE uint32_t get_sub_group_id() {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_BuiltInSubgroupId();
#else
  throw_unsupported_query("Free function calls are not supported on host");
#endif
}

inline __SYCL_ALWAYS_INLINE uint32_t get_sub_group_local_range() {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_BuiltInSubgroupSize();
#else
  throw_unsupported_query("Free function calls are not supported on host");
#endif
}

inline __SYCL_ALWAYS_INLINE uint32_t get_sub_group_max_local_range() {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_BuiltInSubgroupMaxSize();
#else
  throw_unsupported_query("Free function calls are not supported on host");
#endif
}

inline __SYCL_ALWAYS_INLINE uint32_t get_sub_group_group_range() {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_BuiltInNumSubgroups();
#else
  throw_unsupported_query("Free function calls are not supported on host");
#endif
}

} // namespace detail::work_item_context
} // namespace _V1
} // namespace sycl