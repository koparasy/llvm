//==----------- group_core.hpp --- SYCL work group core -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__spirv/spirv_types.hpp> // for Scope, __ocl_event_t
#include <sycl/access/access.hpp>       // for decorated, mode, addr...
#include <sycl/detail/assert.hpp>
#include <sycl/detail/defines.hpp>                   // for __SYCL_TYPE
#include <sycl/detail/defines_elementary.hpp> // for __SYCL2020_DEPRECATED
#include <sycl/detail/helpers.hpp> // for Builder, getSPIRVMemo...
#include <sycl/detail/id_core.hpp>
#include <sycl/detail/type_traits/bool_traits.hpp> // for is_bool, change_base_type_t
#include <sycl/device_event.hpp>                   // for device_event
#include <sycl/memory_enums.hpp>                   // for memory_scope
#include <sycl/pointers.hpp>                       // for decorated_global_ptr
#include <sycl/detail/range_core.hpp>

#ifndef __SYCL_DEVICE_ONLY__
#include <sycl/exception.hpp>
#endif

#include <stddef.h>    // for size_t
#include <stdint.h>    // for uint8_t, uint32_t
#include <type_traits> // for enable_if_t, remove_c...

namespace sycl {
inline namespace _V1 {
template <int Dimensions> class h_item;

/// Encapsulates all functionality required to represent a particular work-group
/// within a parallel execution.
///
/// \ingroup sycl_api
template <int Dimensions = 1> class __SYCL_TYPE(group) group {
public:
#ifndef __DISABLE_SYCL_INTEL_GROUP_ALGORITHMS__
  using id_type = id<Dimensions>;
  using range_type = range<Dimensions>;
  using linear_id_type = size_t;
  static constexpr int dimensions = Dimensions;
#endif // __DISABLE_SYCL_INTEL_GROUP_ALGORITHMS__

  static constexpr sycl::memory_scope fence_scope =
      sycl::memory_scope::work_group;

  group() = delete;

  __SYCL2020_DEPRECATED("use sycl::group::get_group_id() instead")
  id<Dimensions> get_id() const { return index; }

  __SYCL2020_DEPRECATED("use sycl::group::get_group_id() instead")
  size_t get_id(int dimension) const { return index[dimension]; }

  id<Dimensions> get_group_id() const { return index; }

  size_t get_group_id(int dimension) const { return index[dimension]; }

  __SYCL2020_DEPRECATED("calculate sycl::group::get_group_range() * "
                        "sycl::group::get_max_local_range() instead")
  range<Dimensions> get_global_range() const { return globalRange; }

  size_t get_global_range(int dimension) const {
    return globalRange[dimension];
  }

  id<Dimensions> get_local_id() const {
#ifdef __SYCL_DEVICE_ONLY__
    return __spirv::initBuiltInLocalInvocationId<Dimensions, id<Dimensions>>();
#else
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "get_local_id() is not implemented on host");
#endif
  }

  size_t get_local_id(int dimention) const { return get_local_id()[dimention]; }

  size_t get_local_linear_id() const {
    return get_local_linear_id_impl<Dimensions>();
  }

  range<Dimensions> get_local_range() const { return localRange; }

  size_t get_local_range(int dimension) const { return localRange[dimension]; }

  size_t get_local_linear_range() const {
    return get_local_linear_range_impl();
  }

  range<Dimensions> get_group_range() const { return groupRange; }

  size_t get_group_range(int dimension) const {
    return get_group_range()[dimension];
  }

  size_t get_group_linear_range() const {
    return get_group_linear_range_impl();
  }

  range<Dimensions> get_max_local_range() const { return get_local_range(); }

  size_t operator[](int dimension) const { return index[dimension]; }

  __SYCL2020_DEPRECATED("use sycl::group::get_group_linear_id() instead")
  size_t get_linear_id() const { return get_group_linear_id(); }

  size_t get_group_linear_id() const { return get_group_linear_id_impl(); }

  bool leader() const { return (get_local_linear_id() == 0); }

  // Note: These signatures for parallel_for_work_item are intentionally
  // non-conforming. The spec says this should take const WorkItemFunctionT &,
  // but we take it by value, and rely on passing by value being done as passing
  // a copy by reference (ptr byval) to ensure that the special handling in
  // SYCLLowerWGScopePass to mutate the passed functor object works.

  template <typename WorkItemFunctionT>
#ifdef __NativeCPU__
  __attribute__((__libclc_call__))
#endif
  void parallel_for_work_item(WorkItemFunctionT Func) const;

  template <typename WorkItemFunctionT>
#ifdef __NativeCPU__
  __attribute__((__libclc_call__))
#endif
  void parallel_for_work_item(range<Dimensions> flexibleRange,
                              WorkItemFunctionT Func) const;

  /// Executes a work-group mem-fence with memory ordering on the local address
  /// space, global address space or both based on the value of \p accessSpace.
  template <access::mode accessMode = access::mode::read_write>
  void mem_fence(
      [[maybe_unused]]
      typename std::enable_if_t<accessMode == access::mode::read ||
                                    accessMode == access::mode::write ||
                                    accessMode == access::mode::read_write,
                                access::fence_space>
          accessSpace = access::fence_space::global_and_local) const {
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

  /// Asynchronously copies a number of elements specified by \p numElements
  /// from the source pointed by \p src to destination pointed by \p dest
  /// with a source stride specified by \p srcStride, and returns a SYCL
  /// device_event which can be used to wait on the completion of the copy.
  /// Permitted types for dataT are all scalar and vector types, except boolean.
  template <typename dataT>
  __SYCL2020_DEPRECATED("Use decorated multi_ptr arguments instead")
  std::enable_if_t<
      !detail::is_bool<dataT>::value,
      device_event> async_work_group_copy([[maybe_unused]] local_ptr<dataT>
                                              dest,
                                          [[maybe_unused]] global_ptr<dataT>
                                              src,
                                          [[maybe_unused]] size_t numElements,
                        [[maybe_unused]] size_t srcStride)
      const;

  /// Asynchronously copies a number of elements specified by \p numElements
  /// from the source pointed by \p src to destination pointed by \p dest with
  /// the destination stride specified by \p destStride, and returns a SYCL
  /// device_event which can be used to wait on the completion of the copy.
  /// Permitted types for dataT are all scalar and vector types, except boolean.
  template <typename dataT>
  __SYCL2020_DEPRECATED("Use decorated multi_ptr arguments instead")
  std::enable_if_t<
      !detail::is_bool<dataT>::value,
      device_event> async_work_group_copy([[maybe_unused]] global_ptr<dataT>
                                              dest,
                                          [[maybe_unused]] local_ptr<dataT> src,
                                          [[maybe_unused]] size_t numElements,
                        [[maybe_unused]] size_t destStride)
      const;

  /// Asynchronously copies a number of elements specified by \p numElements
  /// from the source pointed by \p src to destination pointed by \p dest
  /// with a source stride specified by \p srcStride, and returns a SYCL
  /// device_event which can be used to wait on the completion of the copy.
  /// Permitted types for DestDataT are all scalar and vector types, except
  /// boolean. SrcDataT must be either the same as DestDataT or const DestDataT.
  template <typename DestDataT, typename SrcDataT>
  std::enable_if_t<!detail::is_bool<DestDataT>::value &&
                       std::is_same_v<std::remove_const_t<SrcDataT>, DestDataT>,
                   device_event>
  async_work_group_copy([[maybe_unused]] decorated_local_ptr<DestDataT> dest,
                        [[maybe_unused]] decorated_global_ptr<SrcDataT> src,
                        [[maybe_unused]] size_t numElements,
              [[maybe_unused]] size_t srcStride) const;

  /// Asynchronously copies a number of elements specified by \p numElements
  /// from the source pointed by \p src to destination pointed by \p dest with
  /// the destination stride specified by \p destStride, and returns a SYCL
  /// device_event which can be used to wait on the completion of the copy.
  /// Permitted types for DestDataT are all scalar and vector types, except
  /// boolean. SrcDataT must be either the same as DestDataT or const DestDataT.
  template <typename DestDataT, typename SrcDataT>
  std::enable_if_t<!detail::is_bool<DestDataT>::value &&
                       std::is_same_v<std::remove_const_t<SrcDataT>, DestDataT>,
                   device_event>
  async_work_group_copy([[maybe_unused]] decorated_global_ptr<DestDataT> dest,
                        [[maybe_unused]] decorated_local_ptr<SrcDataT> src,
                        [[maybe_unused]] size_t numElements,
              [[maybe_unused]] size_t destStride) const;

  /// Specialization for scalar bool type.
  /// Asynchronously copies a number of elements specified by \p NumElements
  /// from the source pointed by \p Src to destination pointed by \p Dest
  /// with a stride specified by \p Stride, and returns a SYCL device_event
  /// which can be used to wait on the completion of the copy.
  template <typename T, access::address_space DestS, access::address_space SrcS>
  __SYCL2020_DEPRECATED("Use decorated multi_ptr arguments instead")
  std::enable_if_t<
      detail::is_scalar_bool<T>::value,
      device_event> async_work_group_copy(multi_ptr<T, DestS,
                                                    access::decorated::legacy>
                                              Dest,
                                          multi_ptr<T, SrcS,
                                                    access::decorated::legacy>
                                              Src,
                                          size_t NumElements,
                        size_t Stride) const;

  /// Specialization for vector bool type.
  /// Asynchronously copies a number of elements specified by \p NumElements
  /// from the source pointed by \p Src to destination pointed by \p Dest
  /// with a stride specified by \p Stride, and returns a SYCL device_event
  /// which can be used to wait on the completion of the copy.
  template <typename T, access::address_space DestS, access::address_space SrcS>
  __SYCL2020_DEPRECATED("Use decorated multi_ptr arguments instead")
  std::enable_if_t<
      detail::is_vector_bool<T>::value,
      device_event> async_work_group_copy(multi_ptr<T, DestS,
                                                    access::decorated::legacy>
                                              Dest,
                                          multi_ptr<T, SrcS,
                                                    access::decorated::legacy>
                                              Src,
                                          size_t NumElements,
                        size_t Stride) const;

  /// Specialization for scalar bool type.
  /// Asynchronously copies a number of elements specified by \p NumElements
  /// from the source pointed by \p Src to destination pointed by \p Dest
  /// with a stride specified by \p Stride, and returns a SYCL device_event
  /// which can be used to wait on the completion of the copy.
  template <typename DestT, access::address_space DestS, typename SrcT,
            access::address_space SrcS>
  std::enable_if_t<detail::is_scalar_bool<DestT>::value &&
                       std::is_same_v<std::remove_const_t<SrcT>, DestT>,
                   device_event>
  async_work_group_copy(multi_ptr<DestT, DestS, access::decorated::yes> Dest,
                        multi_ptr<SrcT, SrcS, access::decorated::yes> Src,
              size_t NumElements, size_t Stride) const;

  /// Specialization for vector bool type.
  /// Asynchronously copies a number of elements specified by \p NumElements
  /// from the source pointed by \p Src to destination pointed by \p Dest
  /// with a stride specified by \p Stride, and returns a SYCL device_event
  /// which can be used to wait on the completion of the copy.
  template <typename DestT, access::address_space DestS, typename SrcT,
            access::address_space SrcS>
  std::enable_if_t<detail::is_vector_bool<DestT>::value &&
                       std::is_same_v<std::remove_const_t<SrcT>, DestT>,
                   device_event>
  async_work_group_copy(multi_ptr<DestT, DestS, access::decorated::yes> Dest,
                        multi_ptr<SrcT, SrcS, access::decorated::yes> Src,
              size_t NumElements, size_t Stride) const;

  /// Asynchronously copies a number of elements specified by \p numElements
  /// from the source pointed by \p src to destination pointed by \p dest and
  /// returns a SYCL device_event which can be used to wait on the completion
  /// of the copy.
  /// Permitted types for dataT are all scalar and vector types.
  template <typename dataT>
  __SYCL2020_DEPRECATED("Use decorated multi_ptr arguments instead")
  device_event async_work_group_copy(local_ptr<dataT> dest, global_ptr<dataT> src,
                                     size_t numElements) const;

  /// Asynchronously copies a number of elements specified by \p numElements
  /// from the source pointed by \p src to destination pointed by \p dest and
  /// returns a SYCL device_event which can be used to wait on the completion
  /// of the copy.
  /// Permitted types for dataT are all scalar and vector types.
  template <typename dataT>
  __SYCL2020_DEPRECATED("Use decorated multi_ptr arguments instead")
  device_event async_work_group_copy(global_ptr<dataT> dest, local_ptr<dataT> src,
                                     size_t numElements) const;

  /// Asynchronously copies a number of elements specified by \p numElements
  /// from the source pointed by \p src to destination pointed by \p dest and
  /// returns a SYCL device_event which can be used to wait on the completion
  /// of the copy.
  /// Permitted types for DestDataT are all scalar and vector types. SrcDataT
  /// must be either the same as DestDataT or const DestDataT.
  template <typename DestDataT, typename SrcDataT>
  typename std::enable_if_t<
      std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>, device_event>
  async_work_group_copy(decorated_local_ptr<DestDataT> dest,
                        decorated_global_ptr<SrcDataT> src,
                        size_t numElements) const;

  /// Asynchronously copies a number of elements specified by \p numElements
  /// from the source pointed by \p src to destination pointed by \p dest and
  /// returns a SYCL device_event which can be used to wait on the completion
  /// of the copy.
  /// Permitted types for DestDataT are all scalar and vector types. SrcDataT
  /// must be either the same as DestDataT or const DestDataT.
  template <typename DestDataT, typename SrcDataT>
  typename std::enable_if_t<
      std::is_same_v<DestDataT, std::remove_const_t<SrcDataT>>, device_event>
  async_work_group_copy(decorated_global_ptr<DestDataT> dest,
                        decorated_local_ptr<SrcDataT> src,
                        size_t numElements) const;

  template <typename... eventTN> void wait_for(eventTN... Events) const {
    waitForHelper(Events...);
  }

  bool operator==(const group<Dimensions> &rhs) const {
    bool Result = (rhs.globalRange == globalRange) &&
                  (rhs.localRange == localRange) && (rhs.index == index);
    __SYCL_ASSERT(rhs.groupRange == groupRange &&
                  "inconsistent group class fields");
    return Result;
  }

  bool operator!=(const group<Dimensions> &rhs) const {
    return !((*this) == rhs);
  }

private:
  range<Dimensions> globalRange;
  range<Dimensions> localRange;
  range<Dimensions> groupRange;
  id<Dimensions> index;

  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 1), size_t>
  get_local_linear_id_impl() const {
    id<Dimensions> localId = get_local_id();
    return localId[0];
  }

  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 2), size_t>
  get_local_linear_id_impl() const {
    id<Dimensions> localId = get_local_id();
    return localId[0] * localRange[1] + localId[1];
  }

  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 3), size_t>
  get_local_linear_id_impl() const {
    id<Dimensions> localId = get_local_id();
    return (localId[0] * localRange[1] * localRange[2]) +
           (localId[1] * localRange[2]) + localId[2];
  }

  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 1), size_t>
  get_local_linear_range_impl() const {
    auto localRange = get_local_range();
    return localRange[0];
  }

  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 2), size_t>
  get_local_linear_range_impl() const {
    auto localRange = get_local_range();
    return localRange[0] * localRange[1];
  }

  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 3), size_t>
  get_local_linear_range_impl() const {
    auto localRange = get_local_range();
    return localRange[0] * localRange[1] * localRange[2];
  }

  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 1), size_t>
  get_group_linear_range_impl() const {
    auto groupRange = get_group_range();
    return groupRange[0];
  }

  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 2), size_t>
  get_group_linear_range_impl() const {
    auto groupRange = get_group_range();
    return groupRange[0] * groupRange[1];
  }

  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 3), size_t>
  get_group_linear_range_impl() const {
    auto groupRange = get_group_range();
    return groupRange[0] * groupRange[1] * groupRange[2];
  }

  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 1), size_t>
  get_group_linear_id_impl() const {
    return index[0];
  }

  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 2), size_t>
  get_group_linear_id_impl() const {
    return index[0] * groupRange[1] + index[1];
  }

  // SYCL specification 1.2.1rev5, section 4.7.6.5 "Buffer accessor":
  //    Whenever a multi-dimensional index is passed to a SYCL accessor the
  //    linear index is calculated based on the index {id1, id2, id3} provided
  //    and the range of the SYCL accessor {r1, r2, r3} according to row-major
  //    ordering as follows:
  //      id3 + (id2 · r3) + (id1 · r3 · r2)            (4.3)
  // section 4.8.1.8 "group class":
  //    size_t get_linear_id()const
  //    Get a linearized version of the work-group id. Calculating a linear
  //    work-group id from a multi-dimensional index follows the equation 4.3.
  template <int dims = Dimensions>
  typename std::enable_if_t<(dims == 3), size_t>
  get_group_linear_id_impl() const {
    return (index[0] * groupRange[1] * groupRange[2]) +
           (index[1] * groupRange[2]) + index[2];
  }

  void waitForHelper() const {}

  void waitForHelper(device_event Event) const { Event.wait(); }

  template <typename T, typename... Ts>
  void waitForHelper(T E, Ts... Es) const {
    waitForHelper(E);
    waitForHelper(Es...);
  }

protected:
  friend class detail::Builder;
  group(const range<Dimensions> &G, const range<Dimensions> &L,
        const range<Dimensions> GroupRange, const id<Dimensions> &I)
      : globalRange(G), localRange(L), groupRange(GroupRange), index(I) {}
};
} // namespace _V1
} // namespace sycl