//==------ nd_item_extra.hpp --- SYCL iteration nd_item extra ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/nd_item_core.hpp>
#include <sycl/nd_range.hpp>

namespace sycl {
inline namespace _V1 {

template <int Dimensions>
nd_range<Dimensions> nd_item<Dimensions>::get_nd_range() const {
  return nd_range<Dimensions>(get_global_range(), get_local_range(),
                              get_offset());
}

} // namespace _V1
} // namespace sycl