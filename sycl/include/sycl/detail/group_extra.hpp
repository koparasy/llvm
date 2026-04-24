//==---------- group_extra.hpp --- SYCL work group extra ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/async_work_group_copy_ptr.hpp>
#include <sycl/detail/group_core.hpp>
#include <sycl/detail/nd_loop.hpp> // for NDLoop
#include <sycl/h_item.hpp>

#ifndef __SYCL_DEVICE_ONLY__
#include <memory> // for unique_ptr
#endif

namespace sycl {
inline namespace _V1 {

namespace detail {
// Implements a barrier across work items within a work group.
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

// SYCL 1.2.1rev5, section "4.8.5.3 Parallel For hierarchical invoke":
// Quote:
//   ... To guarantee use of private per-work-item memory, the private_memory
//   class can be used to wrap the data. This class very simply constructs
//   private data for a given group across the entire group.The id of the
//   current work-item is passed to any access to grab the correct data.
template <typename T, int Dimensions = 1>
class __SYCL_TYPE(private_memory) private_memory {
public:
  // Construct based directly off the number of work-items
  private_memory(const group<Dimensions> &G) {
#ifndef __SYCL_DEVICE_ONLY__
    // serial host => one instance per work-group - allocate space for each WI
    // in the group:
    Val.reset(new T[G.get_local_range().size()]);
#endif // __SYCL_DEVICE_ONLY__
    (void)G;
  }

  // Access the instance for the current work-item
  T &operator()(const h_item<Dimensions> &Id) {
#ifndef __SYCL_DEVICE_ONLY__
    // Calculate the linear index of current WI and return reference to the
    // corresponding spot in the value array:
    size_t Ind = Id.get_physical_local().get_linear_id();
    return Val.get()[Ind];
#else
    (void)Id;
    return Val;
#endif // __SYCL_DEVICE_ONLY__
  }

private:
#ifdef __SYCL_DEVICE_ONLY__
  // On SYCL device private_memory<T> instance is created per physical WI, so
  // there is 1:1 correspondence betwen this class instances and per-WI memory.
  T Val;
#else
  // On serial host there is one private_memory<T> instance per work group, so
  // it must have space to hold separate value per WI in the group.
  std::unique_ptr<T[]> Val;
#endif // #ifdef __SYCL_DEVICE_ONLY__
};

template <int Dimensions>
template <typename dataT>
std::enable_if_t<!detail::is_bool<dataT>::value, device_event>
group<Dimensions>::async_work_group_copy(local_ptr<dataT> dest,
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
group<Dimensions>::async_work_group_copy(global_ptr<dataT> dest,
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
group<Dimensions>::async_work_group_copy(decorated_local_ptr<DestDataT> dest,
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
group<Dimensions>::async_work_group_copy(decorated_global_ptr<DestDataT> dest,
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

template <int Dimensions>
template <typename dataT>
device_event group<Dimensions>::async_work_group_copy(local_ptr<dataT> dest,
                                                      global_ptr<dataT> src,
                                                      size_t numElements) const {
  return async_work_group_copy(dest, src, numElements, 1);
}

template <int Dimensions>
template <typename dataT>
device_event group<Dimensions>::async_work_group_copy(global_ptr<dataT> dest,
                                                      local_ptr<dataT> src,
                                                      size_t numElements) const {
  return async_work_group_copy(dest, src, numElements, 1);
}

template <int Dimensions>
template <typename DestDataT, typename SrcDataT>
typename std::enable_if_t<
    std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>, device_event>
group<Dimensions>::async_work_group_copy(decorated_local_ptr<DestDataT> dest,
                                         decorated_global_ptr<SrcDataT> src,
                                         size_t numElements) const {
  return async_work_group_copy(dest, src, numElements, 1);
}

template <int Dimensions>
template <typename DestDataT, typename SrcDataT>
typename std::enable_if_t<
    std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>, device_event>
group<Dimensions>::async_work_group_copy(decorated_global_ptr<DestDataT> dest,
                                         decorated_local_ptr<SrcDataT> src,
                                         size_t numElements) const {
  return async_work_group_copy(dest, src, numElements, 1);
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
  detail::NDLoop<Dimensions>::iterate(localRange,
                                      [&](const id<Dimensions> &LocalID) {
                                        item<Dimensions, false> GlobalItem =
                                            detail::Builder::createItem<
                                                Dimensions, false>(
                                                globalRange,
                                                GroupStartID + LocalID);
                                        item<Dimensions, false> LocalItem =
                                            detail::Builder::createItem<
                                                Dimensions, false>(localRange,
                                                                   LocalID);
                                        h_item<Dimensions> HItem =
                                            detail::Builder::createHItem<
                                                Dimensions>(GlobalItem,
                                                            LocalItem);
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
  h_item<Dimensions> HItem =
      detail::Builder::createHItem<Dimensions>(GlobalItem, LocalItem,
                                               flexibleRange);

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
            detail::Builder::createItem<Dimensions, false>(globalRange,
                                                           GroupStartID + LocalID);
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

} // namespace _V1
} // namespace sycl