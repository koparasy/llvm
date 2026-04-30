//==-------- group_impl.hpp --- SYCL work group implementation -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_ops.hpp>
#include <sycl/__spirv/spirv_types.hpp> // for Scope, __ocl_event_t
#include <sycl/__spirv/spirv_vars.hpp>  // for initBuiltInLocalInvocationId
#include <sycl/detail/async_work_group_copy_ptr.hpp> // for convertToOpenCLGroupAsyncCopyPtr
#include <sycl/detail/group_base.hpp>
#include <sycl/detail/helpers.hpp> // for Builder, getSPIRVMemo...
#include <sycl/detail/nd_loop.hpp> // for NDLoop
#include <sycl/item.hpp>           // for item

#ifndef __SYCL_DEVICE_ONLY__
#include <sycl/exception.hpp>
#endif

#include <stddef.h>
#include <stdint.h>
#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace detail {

inline void workGroupBarrier() {
#ifdef __SYCL_DEVICE_ONLY__
  constexpr uint32_t flags =
      static_cast<uint32_t>(
          __spv::MemorySemanticsMask::SequentiallyConsistent) |
      static_cast<uint32_t>(__spv::MemorySemanticsMask::WorkgroupMemory);
  __spirv_ControlBarrier(__spv::Scope::Workgroup, __spv::Scope::Workgroup,
                         flags);
#endif // __SYCL_DEVICE_ONLY__
}

} // namespace detail

template <int Dimensions> id<Dimensions> group<Dimensions>::get_local_id() const {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv::initBuiltInLocalInvocationId<Dimensions, id<Dimensions>>();
#else
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "get_local_id() is not implemented on host");
#endif
}

template <int Dimensions>
template <typename WorkItemFunctionT>
#ifdef __NativeCPU__
__attribute__((__libclc_call__))
#endif
void group<Dimensions>::parallel_for_work_item(WorkItemFunctionT Func) const {
  // need barriers to enforce SYCL semantics for the work item loop -
  // compilers are expected to optimize when possible
  detail::workGroupBarrier();
#ifdef __SYCL_DEVICE_ONLY__
  range<Dimensions> GlobalSize{
      __spirv::initBuiltInGlobalSize<Dimensions, range<Dimensions>>()};
  range<Dimensions> LocalSize{
      __spirv::initBuiltInWorkgroupSize<Dimensions, range<Dimensions>>()};
  id<Dimensions> GlobalId{
      __spirv::initBuiltInGlobalInvocationId<Dimensions, id<Dimensions>>()};
  id<Dimensions> LocalId{
      __spirv::initBuiltInLocalInvocationId<Dimensions, id<Dimensions>>()};

  // no 'iterate' in the device code variant, because
  // (1) this code is already invoked by each work item as a part of the
  //     enclosing parallel_for_work_group kernel
  // (2) the range this pfwi iterates over matches work group size exactly
  item<Dimensions, false> GlobalItem =
      detail::Builder::createItem<Dimensions, false>(GlobalSize, GlobalId);
  item<Dimensions, false> LocalItem =
      detail::Builder::createItem<Dimensions, false>(LocalSize, LocalId);
  h_item<Dimensions> HItem =
      detail::Builder::createHItem<Dimensions>(GlobalItem, LocalItem);

  Func(HItem);
#else
  id<Dimensions> GroupStartID = index * id<Dimensions>{localRange};

  // ... host variant needs explicit 'iterate' because it is serial
  detail::NDLoop<Dimensions>::iterate(
      localRange, [&](const id<Dimensions> &LocalID) {
        item<Dimensions, false> GlobalItem =
            detail::Builder::createItem<Dimensions, false>(
                globalRange, GroupStartID + LocalID);
        item<Dimensions, false> LocalItem =
            detail::Builder::createItem<Dimensions, false>(localRange, LocalID);
        h_item<Dimensions> HItem =
            detail::Builder::createHItem<Dimensions>(GlobalItem, LocalItem);
        Func(HItem);
      });
#endif // __SYCL_DEVICE_ONLY__
  // Need both barriers here - before and after the parallel_for_work_item
  // (PFWI). There can be work group scope code after the PFWI which reads
  // work group local data written within this PFWI. Back Ends are expected to
  // optimize away unneeded barriers (e.g. two barriers in a row).
  detail::workGroupBarrier();
}

template <int Dimensions>
template <typename WorkItemFunctionT>
#ifdef __NativeCPU__
__attribute__((__libclc_call__))
#endif
void group<Dimensions>::parallel_for_work_item(range<Dimensions> flexibleRange,
                                               WorkItemFunctionT Func) const {
  detail::workGroupBarrier();
#ifdef __SYCL_DEVICE_ONLY__
  range<Dimensions> GlobalSize{
      __spirv::initBuiltInGlobalSize<Dimensions, range<Dimensions>>()};
  range<Dimensions> LocalSize{
      __spirv::initBuiltInWorkgroupSize<Dimensions, range<Dimensions>>()};
  id<Dimensions> GlobalId{
      __spirv::initBuiltInGlobalInvocationId<Dimensions, id<Dimensions>>()};
  id<Dimensions> LocalId{
      __spirv::initBuiltInLocalInvocationId<Dimensions, id<Dimensions>>()};

  item<Dimensions, false> GlobalItem =
      detail::Builder::createItem<Dimensions, false>(GlobalSize, GlobalId);
  item<Dimensions, false> LocalItem =
      detail::Builder::createItem<Dimensions, false>(LocalSize, LocalId);
  h_item<Dimensions> HItem = detail::Builder::createHItem<Dimensions>(
      GlobalItem, LocalItem, flexibleRange);

  // iterate over flexible range with work group size stride; each item
  // performs flexibleRange/LocalSize iterations (if the former is divisible
  // by the latter)
  detail::NDLoop<Dimensions>::iterate(
      LocalId, LocalSize, flexibleRange,
      [&](const id<Dimensions> &LogicalLocalID) {
        HItem.setLogicalLocalID(LogicalLocalID);
        Func(HItem);
      });
#else
  id<Dimensions> GroupStartID = index * localRange;

  detail::NDLoop<Dimensions>::iterate(
      localRange, [&](const id<Dimensions> &LocalID) {
        item<Dimensions, false> GlobalItem =
            detail::Builder::createItem<Dimensions, false>(
                globalRange, GroupStartID + LocalID);
        item<Dimensions, false> LocalItem =
            detail::Builder::createItem<Dimensions, false>(localRange, LocalID);
        h_item<Dimensions> HItem = detail::Builder::createHItem<Dimensions>(
            GlobalItem, LocalItem, flexibleRange);

        detail::NDLoop<Dimensions>::iterate(
            LocalID, localRange, flexibleRange,
            [&](const id<Dimensions> &LogicalLocalID) {
              HItem.setLogicalLocalID(LogicalLocalID);
              Func(HItem);
            });
      });
#endif // __SYCL_DEVICE_ONLY__
  detail::workGroupBarrier();
}

template <int Dimensions>
template <access::mode accessMode>
void group<Dimensions>::mem_fence(
    [[maybe_unused]] typename std::enable_if_t<
        accessMode == access::mode::read ||
            accessMode == access::mode::write ||
            accessMode == access::mode::read_write,
        access::fence_space>
        accessSpace) const {
#ifdef __SYCL_DEVICE_ONLY__
  uint32_t flags = detail::getSPIRVMemorySemanticsMask(accessSpace);
  // TODO: currently, there is no good way in SPIR-V to set the memory
  // barrier only for load operations or only for store operations.
  // The full read-and-write barrier is used and the template parameter
  // 'accessMode' is ignored for now. Either SPIR-V or SYCL spec may be
  // changed to address this discrepancy between SPIR-V and SYCL,
  // or if we decide that 'accessMode' is the important feature then
  // we can fix this later, for example, by using OpenCL 1.2 functions
  // read_mem_fence() and write_mem_fence().
  __spirv_MemoryBarrier(__spv::Scope::Workgroup, flags);
#endif
}

template <int Dimensions>
template <typename dataT>
std::enable_if_t<!detail::is_bool<dataT>::value, device_event>
group<Dimensions>::async_work_group_copy(
    [[maybe_unused]] local_ptr<dataT> dest,
    [[maybe_unused]] global_ptr<dataT> src,
    [[maybe_unused]] size_t numElements,
    [[maybe_unused]] size_t srcStride) const {
#ifdef __SYCL_DEVICE_ONLY__
  __ocl_event_t E = __spirv_GroupAsyncCopy(
      __spv::Scope::Workgroup, detail::convertToOpenCLGroupAsyncCopyPtr(dest),
      detail::convertToOpenCLGroupAsyncCopyPtr(src), numElements, srcStride, 0);
  return device_event(E);
#else
  return nullptr;
#endif
}

template <int Dimensions>
template <typename dataT>
std::enable_if_t<!detail::is_bool<dataT>::value, device_event>
group<Dimensions>::async_work_group_copy(
    [[maybe_unused]] global_ptr<dataT> dest,
    [[maybe_unused]] local_ptr<dataT> src,
    [[maybe_unused]] size_t numElements,
    [[maybe_unused]] size_t destStride) const {
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
group<Dimensions>::async_work_group_copy(
    [[maybe_unused]] decorated_local_ptr<DestDataT> dest,
    [[maybe_unused]] decorated_global_ptr<SrcDataT> src,
    [[maybe_unused]] size_t numElements,
    [[maybe_unused]] size_t srcStride) const {
#ifdef __SYCL_DEVICE_ONLY__
  __ocl_event_t E = __spirv_GroupAsyncCopy(
      __spv::Scope::Workgroup, detail::convertToOpenCLGroupAsyncCopyPtr(dest),
      detail::convertToOpenCLGroupAsyncCopyPtr(src), numElements, srcStride, 0);
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
group<Dimensions>::async_work_group_copy(
    [[maybe_unused]] decorated_global_ptr<DestDataT> dest,
    [[maybe_unused]] decorated_local_ptr<SrcDataT> src,
    [[maybe_unused]] size_t numElements,
    [[maybe_unused]] size_t destStride) const {
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
group<Dimensions>::async_work_group_copy(
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
group<Dimensions>::async_work_group_copy(
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
group<Dimensions>::async_work_group_copy(
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
group<Dimensions>::async_work_group_copy(
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

} // namespace _V1
} // namespace sycl
