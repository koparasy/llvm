//==------- sub_group_base.hpp --- SYCL sub-group declarations -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>             // for fence_space
#include <sycl/detail/defines_elementary.hpp> // for __SYCL_DEPRECATED
#include <sycl/id.hpp>                        // for id
#include <sycl/memory_enums.hpp>              // for memory_scope
#include <sycl/range.hpp>                     // for range

#include <stdint.h> // for uint32_t

namespace sycl {
inline namespace _V1 {

template <int Dimensions> class nd_item;

struct sub_group;
namespace ext::oneapi::this_work_item {
inline sycl::sub_group get_sub_group();
} // namespace ext::oneapi::this_work_item

struct sub_group {

  using id_type = id<1>;
  using range_type = range<1>;
  using linear_id_type = uint32_t;
  static constexpr int dimensions = 1;
  static constexpr sycl::memory_scope fence_scope =
      sycl::memory_scope::sub_group;

  /* --- common interface members --- */

  id_type get_local_id() const;

  linear_id_type get_local_linear_id() const;

  range_type get_local_range() const;

  range_type get_max_local_range() const;

  id_type get_group_id() const;

  linear_id_type get_group_linear_id() const;

  range_type get_group_range() const;

  /* --- synchronization functions --- */
  __SYCL_DEPRECATED(
      "Sub-group barrier with no arguments is deprecated."
      "Use sycl::group_barrier with the sub-group as the argument instead.")
  void barrier() const;

  __SYCL_DEPRECATED(
      "Sub-group barrier accepting fence_space is deprecated."
      "Use sycl::group_barrier with the sub-group as the argument instead.")
  void barrier(access::fence_space accessSpace) const;

  linear_id_type get_group_linear_range() const;

  linear_id_type get_local_linear_range() const;

  bool leader() const;

  // Common member functions for by-value semantics
  friend bool operator==(const sub_group &lhs, const sub_group &rhs);

  friend bool operator!=(const sub_group &lhs, const sub_group &rhs);

protected:
  template <int dimensions> friend class sycl::nd_item;
  friend sub_group ext::oneapi::this_work_item::get_sub_group();
  sub_group() = default;
};

} // namespace _V1
} // namespace sycl
