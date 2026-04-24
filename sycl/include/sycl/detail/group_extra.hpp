//==---------- group_extra.hpp --- SYCL work group extra ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

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