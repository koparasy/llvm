//==------ nd_item_extra.hpp --- SYCL iteration nd_item extra ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/async_work_group_copy_ptr.hpp>
#include <sycl/detail/nd_item_core.hpp>
#include <sycl/nd_range.hpp>

namespace sycl {
inline namespace _V1 {

template <int Dimensions>
nd_range<Dimensions> nd_item<Dimensions>::get_nd_range() const {
  return nd_range<Dimensions>(get_global_range(), get_local_range(),
                              get_offset());
}

template <int Dimensions>
template <typename dataT>
std::enable_if_t<!detail::is_bool<dataT>::value, device_event>
nd_item<Dimensions>::async_work_group_copy(local_ptr<dataT> dest,
                                           global_ptr<dataT> src,
                                           size_t numElements,
                                           size_t srcStride) const {
#ifdef __SYCL_DEVICE_ONLY__
  __ocl_event_t E = __spirv_GroupAsyncCopy(
      __spv::Scope::Workgroup, detail::convertToOpenCLGroupAsyncCopyPtr(dest),
      detail::convertToOpenCLGroupAsyncCopyPtr(src), numElements, srcStride,
      0);
  return device_event(E);
#else
  return nullptr;
#endif
}

template <int Dimensions>
template <typename dataT>
std::enable_if_t<!detail::is_bool<dataT>::value, device_event>
nd_item<Dimensions>::async_work_group_copy(global_ptr<dataT> dest,
                                           local_ptr<dataT> src,
                                           size_t numElements,
                                           size_t destStride) const {
#ifdef __SYCL_DEVICE_ONLY__
  __ocl_event_t E = __spirv_GroupAsyncCopy(
      __spv::Scope::Workgroup, detail::convertToOpenCLGroupAsyncCopyPtr(dest),
      detail::convertToOpenCLGroupAsyncCopyPtr(src), numElements, destStride,
      0);
  return device_event(E);
#else
  return nullptr;
#endif
}

template <int Dimensions>
template <typename DestDataT, typename SrcDataT>
std::enable_if_t<!detail::is_bool<DestDataT>::value &&
                     std::is_same_v<std::remove_const_t<SrcDataT>, DestDataT>,
                 device_event>
nd_item<Dimensions>::async_work_group_copy(decorated_local_ptr<DestDataT> dest,
                                           decorated_global_ptr<SrcDataT> src,
                                           size_t numElements,
                                           size_t srcStride) const {
#ifdef __SYCL_DEVICE_ONLY__
  __ocl_event_t E = __spirv_GroupAsyncCopy(
      __spv::Scope::Workgroup, detail::convertToOpenCLGroupAsyncCopyPtr(dest),
      detail::convertToOpenCLGroupAsyncCopyPtr(src), numElements, srcStride,
      0);
  return device_event(E);
#else
  return nullptr;
#endif
}

template <int Dimensions>
template <typename DestDataT, typename SrcDataT>
std::enable_if_t<!detail::is_bool<DestDataT>::value &&
                     std::is_same_v<std::remove_const_t<SrcDataT>, DestDataT>,
                 device_event>
nd_item<Dimensions>::async_work_group_copy(decorated_global_ptr<DestDataT> dest,
                                           decorated_local_ptr<SrcDataT> src,
                                           size_t numElements,
                                           size_t destStride) const {
#ifdef __SYCL_DEVICE_ONLY__
  __ocl_event_t E = __spirv_GroupAsyncCopy(
      __spv::Scope::Workgroup, detail::convertToOpenCLGroupAsyncCopyPtr(dest),
      detail::convertToOpenCLGroupAsyncCopyPtr(src), numElements, destStride,
      0);
  return device_event(E);
#else
  return nullptr;
#endif
}

template <int Dimensions>
template <typename T, access::address_space DestS, access::address_space SrcS>
std::enable_if_t<detail::is_scalar_bool<T>::value, device_event>
nd_item<Dimensions>::async_work_group_copy(
    multi_ptr<T, DestS, access::decorated::legacy> Dest,
    multi_ptr<T, SrcS, access::decorated::legacy> Src, size_t NumElements,
    size_t Stride) const {
  static_assert(sizeof(bool) == sizeof(uint8_t),
                "Async copy to/from bool memory is not supported.");
  auto DestP = multi_ptr<uint8_t, DestS, access::decorated::legacy>(
      reinterpret_cast<uint8_t *>(Dest.get()));
  auto SrcP = multi_ptr<uint8_t, SrcS, access::decorated::legacy>(
      reinterpret_cast<uint8_t *>(Src.get()));
  return async_work_group_copy(DestP, SrcP, NumElements, Stride);
}

template <int Dimensions>
template <typename T, access::address_space DestS, access::address_space SrcS>
std::enable_if_t<detail::is_vector_bool<T>::value, device_event>
nd_item<Dimensions>::async_work_group_copy(
    multi_ptr<T, DestS, access::decorated::legacy> Dest,
    multi_ptr<T, SrcS, access::decorated::legacy> Src, size_t NumElements,
    size_t Stride) const {
  static_assert(sizeof(bool) == sizeof(uint8_t),
                "Async copy to/from bool memory is not supported.");
  using VecT = detail::change_base_type_t<T, uint8_t>;
  auto DestP = address_space_cast<DestS, access::decorated::legacy>(
      reinterpret_cast<VecT *>(Dest.get()));
  auto SrcP = address_space_cast<SrcS, access::decorated::legacy>(
      reinterpret_cast<VecT *>(Src.get()));
  return async_work_group_copy(DestP, SrcP, NumElements, Stride);
}

template <int Dimensions>
template <typename DestT, access::address_space DestS, typename SrcT,
          access::address_space SrcS>
std::enable_if_t<detail::is_scalar_bool<DestT>::value &&
                     std::is_same_v<std::remove_const_t<SrcT>, DestT>,
                 device_event>
nd_item<Dimensions>::async_work_group_copy(
    multi_ptr<DestT, DestS, access::decorated::yes> Dest,
    multi_ptr<SrcT, SrcS, access::decorated::yes> Src, size_t NumElements,
    size_t Stride) const {
  static_assert(sizeof(bool) == sizeof(uint8_t),
                "Async copy to/from bool memory is not supported.");
  using QualSrcT =
      std::conditional_t<std::is_const_v<SrcT>, const uint8_t, uint8_t>;
  auto DestP = multi_ptr<uint8_t, DestS, access::decorated::yes>(
      reinterpret_cast<typename multi_ptr<uint8_t, DestS,
                                          access::decorated::yes>::pointer>(
          Dest.get_decorated()));
  auto SrcP = multi_ptr<QualSrcT, SrcS, access::decorated::yes>(
      reinterpret_cast<typename multi_ptr<QualSrcT, SrcS,
                                          access::decorated::yes>::pointer>(
          Src.get_decorated()));
  return async_work_group_copy(DestP, SrcP, NumElements, Stride);
}

template <int Dimensions>
template <typename DestT, access::address_space DestS, typename SrcT,
          access::address_space SrcS>
std::enable_if_t<detail::is_vector_bool<DestT>::value &&
                     std::is_same_v<std::remove_const_t<SrcT>, DestT>,
                 device_event>
nd_item<Dimensions>::async_work_group_copy(
    multi_ptr<DestT, DestS, access::decorated::yes> Dest,
    multi_ptr<SrcT, SrcS, access::decorated::yes> Src, size_t NumElements,
    size_t Stride) const {
  static_assert(sizeof(bool) == sizeof(uint8_t),
                "Async copy to/from bool memory is not supported.");
  using VecT = detail::change_base_type_t<DestT, uint8_t>;
  using QualSrcVecT =
      std::conditional_t<std::is_const_v<SrcT>, std::add_const_t<VecT>, VecT>;
  auto DestP = multi_ptr<VecT, DestS, access::decorated::yes>(
      reinterpret_cast<
          typename multi_ptr<VecT, DestS, access::decorated::yes>::pointer>(
          Dest.get_decorated()));
  auto SrcP = multi_ptr<QualSrcVecT, SrcS, access::decorated::yes>(
      reinterpret_cast<typename multi_ptr<QualSrcVecT, SrcS,
                                          access::decorated::yes>::pointer>(
          Src.get_decorated()));
  return async_work_group_copy(DestP, SrcP, NumElements, Stride);
}

template <int Dimensions>
template <typename dataT>
device_event nd_item<Dimensions>::async_work_group_copy(local_ptr<dataT> dest,
                                                        global_ptr<dataT> src,
                                                        size_t numElements) const {
  return async_work_group_copy(dest, src, numElements, 1);
}

template <int Dimensions>
template <typename dataT>
device_event nd_item<Dimensions>::async_work_group_copy(global_ptr<dataT> dest,
                                                        local_ptr<dataT> src,
                                                        size_t numElements) const {
  return async_work_group_copy(dest, src, numElements, 1);
}

template <int Dimensions>
template <typename DestDataT, typename SrcDataT>
typename std::enable_if_t<
    std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>, device_event>
nd_item<Dimensions>::async_work_group_copy(decorated_local_ptr<DestDataT> dest,
                                           decorated_global_ptr<SrcDataT> src,
                                           size_t numElements) const {
  return async_work_group_copy(dest, src, numElements, 1);
}

template <int Dimensions>
template <typename DestDataT, typename SrcDataT>
typename std::enable_if_t<
    std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>, device_event>
nd_item<Dimensions>::async_work_group_copy(decorated_global_ptr<DestDataT> dest,
                                           decorated_local_ptr<SrcDataT> src,
                                           size_t numElements) const {
  return async_work_group_copy(dest, src, numElements, 1);
}

} // namespace _V1
} // namespace sycl