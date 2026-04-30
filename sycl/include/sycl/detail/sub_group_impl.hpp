//==------- sub_group_impl.hpp --- SYCL sub-group implementation -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_ops_subgroup.hpp>
#include <sycl/detail/address_space_cast.hpp>
#include <sycl/detail/helpers.hpp> // for getSPIRVMemorySemanticsMask
#include <sycl/detail/sub_group_base.hpp>
#include <sycl/nd_item.hpp>

#ifndef __SYCL_DEVICE_ONLY__
#include <sycl/exception.hpp> // for exception, make_error...
#endif

namespace sycl {
inline namespace _V1 {

inline sub_group::id_type sub_group::get_local_id() const {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_BuiltInSubgroupLocalInvocationId();
#else
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

inline sub_group::linear_id_type sub_group::get_local_linear_id() const {
#ifdef __SYCL_DEVICE_ONLY__
  return static_cast<linear_id_type>(get_local_id()[0]);
#else
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

inline sub_group::range_type sub_group::get_local_range() const {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_BuiltInSubgroupSize();
#else
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

inline sub_group::range_type sub_group::get_max_local_range() const {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_BuiltInSubgroupMaxSize();
#else
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

inline sub_group::id_type sub_group::get_group_id() const {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_BuiltInSubgroupId();
#else
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

inline sub_group::linear_id_type sub_group::get_group_linear_id() const {
#ifdef __SYCL_DEVICE_ONLY__
  return static_cast<linear_id_type>(get_group_id()[0]);
#else
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

inline sub_group::range_type sub_group::get_group_range() const {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_BuiltInNumSubgroups();
#else
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

inline void sub_group::barrier() const {
#ifdef __SYCL_DEVICE_ONLY__
  __spirv_ControlBarrier(
      __spv::Scope::Subgroup, __spv::Scope::Subgroup,
      __spv::MemorySemanticsMask::AcquireRelease |
          __spv::MemorySemanticsMask::SubgroupMemory |
          __spv::MemorySemanticsMask::WorkgroupMemory |
          __spv::MemorySemanticsMask::CrossWorkgroupMemory);
#else
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

inline void sub_group::barrier(access::fence_space accessSpace) const {
#ifdef __SYCL_DEVICE_ONLY__
  int32_t flags = sycl::detail::getSPIRVMemorySemanticsMask(accessSpace);
  __spirv_ControlBarrier(__spv::Scope::Subgroup, __spv::Scope::Subgroup, flags);
#else
  (void)accessSpace;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

inline sub_group::linear_id_type sub_group::get_group_linear_range() const {
#ifdef __SYCL_DEVICE_ONLY__
  return static_cast<linear_id_type>(get_group_range()[0]);
#else
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

inline sub_group::linear_id_type sub_group::get_local_linear_range() const {
#ifdef __SYCL_DEVICE_ONLY__
  return static_cast<linear_id_type>(get_local_range()[0]);
#else
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

inline bool sub_group::leader() const {
#ifdef __SYCL_DEVICE_ONLY__
  return get_local_linear_id() == 0;
#else
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

inline bool operator==(const sub_group &lhs, const sub_group &rhs) {
#ifdef __SYCL_DEVICE_ONLY__
  return lhs.get_group_id() == rhs.get_group_id();
#else
  (void)lhs;
  (void)rhs;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

inline bool operator!=(const sub_group &lhs, const sub_group &rhs) {
#ifdef __SYCL_DEVICE_ONLY__
  return !(lhs == rhs);
#else
  (void)lhs;
  (void)rhs;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

template <int Dimensions> sub_group nd_item<Dimensions>::get_sub_group() const {
  return sub_group();
}

} // namespace _V1
} // namespace sycl
