//==-------- sub_group_core.hpp --- SYCL sub-group core ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_ops_subgroup.hpp>
#include <sycl/detail/defines_elementary.hpp> // for __SYCL_DEPRECATED
#include <sycl/detail/fwd/multi_ptr.hpp>
#include <sycl/detail/nd_item_core.hpp>
#include <sycl/detail/type_traits/integer_traits.hpp>
#include <sycl/memory_enums.hpp>

#include <stdint.h>    // for uint32_t
#include <type_traits> // for enable_if_t, remove_cv_t

#ifndef __SYCL_DEVICE_ONLY__
#include <sycl/exception.hpp> // for exception, make_error...
#endif

namespace sycl {
inline namespace _V1 {

template <typename T, int N> class __SYCL_EBO vec;

namespace detail {
namespace sub_group {

template <typename T>
using SelectBlockT = fixed_width_unsigned<sizeof(T)>;

template <typename T, access::address_space Space>
using AcceptableForGlobalLoadStore =
    std::bool_constant<!std::is_same_v<void, SelectBlockT<T>> &&
                       Space == access::address_space::global_space>;

template <typename T, access::address_space Space>
using AcceptableForLocalLoadStore =
    std::bool_constant<!std::is_same_v<void, SelectBlockT<T>> &&
                       Space == access::address_space::local_space>;

} // namespace sub_group

template <typename CVT, access::address_space Space,
          access::decorated IsDecorated, typename T = std::remove_cv_t<CVT>>
multi_ptr<T, Space, IsDecorated>
GetUnqualMultiPtr(const multi_ptr<CVT, Space, IsDecorated> &Mptr);

} // namespace detail

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

  id_type get_local_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_BuiltInSubgroupLocalInvocationId();
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  linear_id_type get_local_linear_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_local_id()[0]);
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  range_type get_local_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_BuiltInSubgroupSize();
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  range_type get_max_local_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_BuiltInSubgroupMaxSize();
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  id_type get_group_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_BuiltInSubgroupId();
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  linear_id_type get_group_linear_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_group_id()[0]);
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  range_type get_group_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv_BuiltInNumSubgroups();
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  /* --- sub_group load/stores --- */
  /* these can map to SIMD or block read/write hardware where available */
#ifdef __SYCL_DEVICE_ONLY__
  template <typename CVT, typename T = std::remove_cv_t<CVT>>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_load instead.")
  std::enable_if_t<!std::is_same<remove_decoration_t<T>, T>::value, T>
  load(CVT *cv_src) const;

  template <typename CVT, typename T = std::remove_cv_t<CVT>>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_load instead.")
  std::enable_if_t<std::is_same<remove_decoration_t<T>, T>::value, T>
  load(CVT *cv_src) const;
#else  //__SYCL_DEVICE_ONLY__
  template <typename CVT, typename T = std::remove_cv_t<CVT>>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_load instead.")
  T load(CVT *src) const;
#endif //__SYCL_DEVICE_ONLY__

  template <typename CVT, access::address_space Space,
            access::decorated IsDecorated, typename T = std::remove_cv_t<CVT>>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_load instead.")
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value,
      T>
  load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const;

  template <typename CVT, access::address_space Space,
            access::decorated IsDecorated, typename T = std::remove_cv_t<CVT>>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_load instead.")
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForLocalLoadStore<T, Space>::value,
      T>
  load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const;

#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__) || defined(__AMDGCN__)
  template <int N, typename CVT, access::address_space Space,
            access::decorated IsDecorated, typename T = std::remove_cv_t<CVT>>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_load instead.")
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value,
      vec<T, N>>
  load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const;
#else  // __NVPTX__ || __AMDGCN__
  template <int N, typename CVT, access::address_space Space,
            access::decorated IsDecorated, typename T = std::remove_cv_t<CVT>>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_load instead.")
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
          N != 1 && N != 3 && N != 16,
      vec<T, N>>
  load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const;

  template <int N, typename CVT, access::address_space Space,
            access::decorated IsDecorated, typename T = std::remove_cv_t<CVT>>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_load instead.")
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
          N == 16,
      vec<T, 16>>
  load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const;

  template <int N, typename CVT, access::address_space Space,
            access::decorated IsDecorated, typename T = std::remove_cv_t<CVT>>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_load instead.")
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
          N == 3,
      vec<T, 3>>
  load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const;

  template <int N, typename CVT, access::address_space Space,
            access::decorated IsDecorated, typename T = std::remove_cv_t<CVT>>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_load instead.")
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
          N == 1,
      vec<T, 1>>
  load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const;
#endif // ___NVPTX___
#else  // __SYCL_DEVICE_ONLY__
  template <int N, typename CVT, access::address_space Space,
            access::decorated IsDecorated, typename T = std::remove_cv_t<CVT>>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_load instead.")
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value,
      vec<T, N>>
  load(const multi_ptr<CVT, Space, IsDecorated> src) const;
#endif // __SYCL_DEVICE_ONLY__

  template <int N, typename CVT, access::address_space Space,
            access::decorated IsDecorated, typename T = std::remove_cv_t<CVT>>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_load instead.")
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForLocalLoadStore<T, Space>::value,
      vec<T, N>>
  load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const;

#ifdef __SYCL_DEVICE_ONLY__
  template <typename T>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_store instead.")
  std::enable_if_t<!std::is_same<remove_decoration_t<T>, T>::value>
  store(T *dst, const remove_decoration_t<T> &x) const;

  template <typename T>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_store instead.")
  std::enable_if_t<std::is_same<remove_decoration_t<T>, T>::value>
  store(T *dst, const remove_decoration_t<T> &x) const;
#else  //__SYCL_DEVICE_ONLY__
  template <typename T>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_store instead.")
  void store(T *dst, const T &x) const;
#endif //__SYCL_DEVICE_ONLY__

  template <typename T, access::address_space Space,
            access::decorated DecorateAddress>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_store instead.")
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value>
  store(multi_ptr<T, Space, DecorateAddress> dst, const T &x) const;

  template <typename T, access::address_space Space,
            access::decorated DecorateAddress>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_store instead.")
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForLocalLoadStore<T, Space>::value>
  store(multi_ptr<T, Space, DecorateAddress> dst, const T &x) const;

#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__) || defined(__AMDGCN__)
  template <int N, typename T, access::address_space Space,
            access::decorated DecorateAddress>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_store instead.")
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value>
  store(multi_ptr<T, Space, DecorateAddress> dst, const vec<T, N> &x) const;
#else // __NVPTX__ || __AMDGCN__
  template <int N, typename T, access::address_space Space,
            access::decorated DecorateAddress>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_store instead.")
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
      N != 1 && N != 3 && N != 16>
  store(multi_ptr<T, Space, DecorateAddress> dst, const vec<T, N> &x) const;

  template <int N, typename T, access::address_space Space,
            access::decorated DecorateAddress>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_store instead.")
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
      N == 1>
  store(multi_ptr<T, Space, DecorateAddress> dst, const vec<T, 1> &x) const;

  template <int N, typename T, access::address_space Space,
            access::decorated DecorateAddress>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_store instead.")
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
      N == 3>
  store(multi_ptr<T, Space, DecorateAddress> dst, const vec<T, 3> &x) const;

  template <int N, typename T, access::address_space Space,
            access::decorated DecorateAddress>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_store instead.")
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
      N == 16>
  store(multi_ptr<T, Space, DecorateAddress> dst, const vec<T, 16> &x) const;

#endif // __NVPTX__ || __AMDGCN__
#else  // __SYCL_DEVICE_ONLY__
  template <int N, typename T, access::address_space Space,
            access::decorated DecorateAddress>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_store instead.")
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value>
  store(multi_ptr<T, Space, DecorateAddress> dst, const vec<T, N> &x) const;
#endif // __SYCL_DEVICE_ONLY__

  template <int N, typename T, access::address_space Space,
            access::decorated DecorateAddress>
  __SYCL_DEPRECATED("Use sycl::ext::oneapi::experimental::group_store instead.")
  std::enable_if_t<
      sycl::detail::sub_group::AcceptableForLocalLoadStore<T, Space>::value>
  store(multi_ptr<T, Space, DecorateAddress> dst, const vec<T, N> &x) const;

  /* --- synchronization functions --- */
  __SYCL_DEPRECATED(
      "Sub-group barrier with no arguments is deprecated."
      "Use sycl::group_barrier with the sub-group as the argument instead.")
  void barrier() const {
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

  __SYCL_DEPRECATED(
      "Sub-group barrier accepting fence_space is deprecated."
      "Use sycl::group_barrier with the sub-group as the argument instead.")
  void barrier(access::fence_space accessSpace) const {
#ifdef __SYCL_DEVICE_ONLY__
    int32_t flags = sycl::detail::getSPIRVMemorySemanticsMask(accessSpace);
    __spirv_ControlBarrier(__spv::Scope::Subgroup, __spv::Scope::Subgroup,
                           flags);
#else
    (void)accessSpace;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  linear_id_type get_group_linear_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_group_range()[0]);
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  linear_id_type get_local_linear_range() const {
#ifdef __SYCL_DEVICE_ONLY__
    return static_cast<linear_id_type>(get_local_range()[0]);
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  bool leader() const {
#ifdef __SYCL_DEVICE_ONLY__
    return get_local_linear_id() == 0;
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  friend bool operator==(const sub_group &lhs, const sub_group &rhs) {
#ifdef __SYCL_DEVICE_ONLY__
    return lhs.get_group_id() == rhs.get_group_id();
#else
    (void)lhs;
    (void)rhs;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

  friend bool operator!=(const sub_group &lhs, const sub_group &rhs) {
#ifdef __SYCL_DEVICE_ONLY__
    return !(lhs == rhs);
#else
    (void)lhs;
    (void)rhs;
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Sub-groups are not supported on host.");
#endif
  }

protected:
  template <int dimensions> friend class sycl::nd_item;
  friend sub_group ext::oneapi::this_work_item::get_sub_group();
  sub_group() = default;
};

template <int Dimensions> inline sub_group nd_item<Dimensions>::get_sub_group() const {
  return sub_group();
}

namespace ext::oneapi::this_work_item {
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

namespace khr {
inline sycl::sub_group this_sub_group() {
  return ext::oneapi::this_work_item::get_sub_group();
}
} // namespace khr

} // namespace _V1
} // namespace sycl