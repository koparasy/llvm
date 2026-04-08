//==-------- work_item_primitives.hpp - SYCL free work-item primitives ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/work_item_context.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::this_work_item {

template <int Dimensions>
inline size_t get_global_id(int Dimension) {
  return sycl::detail::work_item_context::get_global_id<Dimensions>(Dimension);
}

template <int Dimensions>
inline size_t get_global_range(int Dimension) {
  return sycl::detail::work_item_context::get_global_range<Dimensions>(
      Dimension);
}

template <int Dimensions>
inline size_t get_global_offset(int Dimension) {
  return sycl::detail::work_item_context::get_global_offset<Dimensions>(
      Dimension);
}

template <int Dimensions>
inline size_t get_local_id(int Dimension) {
  return sycl::detail::work_item_context::get_local_id<Dimensions>(Dimension);
}

template <int Dimensions>
inline size_t get_local_range(int Dimension) {
  return sycl::detail::work_item_context::get_local_range<Dimensions>(Dimension);
}

template <int Dimensions>
inline size_t get_group_id(int Dimension) {
  return sycl::detail::work_item_context::get_group_id<Dimensions>(Dimension);
}

template <int Dimensions>
inline size_t get_group_range(int Dimension) {
  return sycl::detail::work_item_context::get_group_range<Dimensions>(Dimension);
}

template <int Dimensions> inline size_t get_global_linear_id() {
  return sycl::detail::work_item_context::get_global_linear_id<Dimensions>();
}

template <int Dimensions> inline size_t get_local_linear_id() {
  return sycl::detail::work_item_context::get_local_linear_id<Dimensions>();
}

template <int Dimensions> inline size_t get_group_linear_id() {
  return sycl::detail::work_item_context::get_group_linear_id<Dimensions>();
}

template <int Dimensions> inline size_t get_global_linear_range() {
  return sycl::detail::work_item_context::get_global_linear_range<Dimensions>();
}

template <int Dimensions> inline size_t get_local_linear_range() {
  return sycl::detail::work_item_context::get_local_linear_range<Dimensions>();
}

template <int Dimensions> inline size_t get_group_linear_range() {
  return sycl::detail::work_item_context::get_group_linear_range<Dimensions>();
}

inline uint32_t get_sub_group_local_id() {
  return sycl::detail::work_item_context::get_sub_group_local_id();
}

inline uint32_t get_sub_group_id() {
  return sycl::detail::work_item_context::get_sub_group_id();
}

inline uint32_t get_sub_group_local_range() {
  return sycl::detail::work_item_context::get_sub_group_local_range();
}

inline uint32_t get_sub_group_max_local_range() {
  return sycl::detail::work_item_context::get_sub_group_max_local_range();
}

inline uint32_t get_sub_group_group_range() {
  return sycl::detail::work_item_context::get_sub_group_group_range();
}

} // namespace ext::oneapi::this_work_item
} // namespace _V1
} // namespace sycl