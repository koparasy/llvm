//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __SYCL_KHR_INCLUDES_WORK_ITEM_QUERIES
#define __SYCL_KHR_INCLUDES_WORK_ITEM_QUERIES

#include "version.hpp"

#include <sycl/nd_item.hpp>

namespace sycl {
inline namespace _V1 {
struct sub_group;

namespace khr {

template <int Dimensions> nd_item<Dimensions> this_nd_item() {
  return sycl::detail::getFreeFunctionQueryNDItem<Dimensions>();
}

template <int Dimensions> group<Dimensions> this_group() {
  return sycl::detail::getFreeFunctionQueryGroup<Dimensions>();
}

sycl::sub_group this_sub_group();

} // namespace khr
} // namespace _V1
} // namespace sycl

#endif // __SYCL_KHR_INCLUDES_WORK_ITEM_QUERIES