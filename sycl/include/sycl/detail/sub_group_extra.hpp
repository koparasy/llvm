//==------- sub_group_extra.hpp --- SYCL sub-group extras ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/address_space_cast.hpp>
#include <sycl/detail/generic_type_traits.hpp>
#include <sycl/detail/sub_group_core.hpp>
#include <sycl/multi_ptr.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
namespace sub_group {

template <typename MultiPtrTy> auto convertToBlockPtr(MultiPtrTy MultiPtr) {
  static_assert(is_multi_ptr_v<MultiPtrTy>);
  auto DecoratedPtr = convertToOpenCLType(MultiPtr);
  using DecoratedPtrTy = decltype(DecoratedPtr);
  using ElemTy = remove_decoration_t<std::remove_pointer_t<DecoratedPtrTy>>;

  using TargetElemTy = SelectBlockT<ElemTy>;
  // TODO: Handle cv qualifiers.
#ifdef __SYCL_DEVICE_ONLY__
  using ResultTy =
      typename DecoratedType<TargetElemTy,
                             deduce_AS<DecoratedPtrTy>::value>::type *;
#else
  using ResultTy = TargetElemTy *;
#endif
  return reinterpret_cast<ResultTy>(DecoratedPtr);
}

#ifdef __SYCL_DEVICE_ONLY__
template <typename T, access::address_space Space,
          access::decorated DecorateAddress>
T load(const multi_ptr<T, Space, DecorateAddress> src) {
  using BlockT = SelectBlockT<T>;
  BlockT Ret = __spirv_SubgroupBlockReadINTEL<BlockT>(convertToBlockPtr(src));

  return sycl::bit_cast<T>(Ret);
}

template <int N, typename T, access::address_space Space,
          access::decorated DecorateAddress>
vec<T, N> load(const multi_ptr<T, Space, DecorateAddress> src) {
  using BlockT = SelectBlockT<T>;
  using VecT = sycl::detail::ConvertToOpenCLType_t<vec<BlockT, N>>;
  VecT Ret = __spirv_SubgroupBlockReadINTEL<VecT>(convertToBlockPtr(src));

  return sycl::bit_cast<vec<T, N>>(Ret);
}

template <typename T, access::address_space Space,
          access::decorated DecorateAddress>
void store(multi_ptr<T, Space, DecorateAddress> dst, const T &x) {
  using BlockT = SelectBlockT<T>;

  __spirv_SubgroupBlockWriteINTEL(convertToBlockPtr(dst),
                                  sycl::bit_cast<BlockT>(x));
}

template <int N, typename T, access::address_space Space,
          access::decorated DecorateAddress>
void store(multi_ptr<T, Space, DecorateAddress> dst, const vec<T, N> &x) {
  using BlockT = SelectBlockT<T>;
  using VecT = sycl::detail::ConvertToOpenCLType_t<vec<BlockT, N>>;

  __spirv_SubgroupBlockWriteINTEL(convertToBlockPtr(dst),
                                  sycl::bit_cast<VecT>(x));
}
#endif // __SYCL_DEVICE_ONLY__

} // namespace sub_group

template <typename CVT, access::address_space Space,
          access::decorated IsDecorated, typename T>
inline multi_ptr<T, Space, IsDecorated>
GetUnqualMultiPtr(const multi_ptr<CVT, Space, IsDecorated> &Mptr) {
  if constexpr (IsDecorated == access::decorated::legacy) {
    return multi_ptr<T, Space, IsDecorated>{
        const_cast<typename multi_ptr<T, Space, IsDecorated>::pointer_t>(
            Mptr.get())};
  } else {
    return multi_ptr<T, Space, IsDecorated>{
        const_cast<typename multi_ptr<T, Space, IsDecorated>::pointer>(
            Mptr.get_decorated())};
  }
}

} // namespace detail

#ifdef __SYCL_DEVICE_ONLY__
template <typename CVT, typename T>
inline std::enable_if_t<!std::is_same<remove_decoration_t<T>, T>::value, T>
sub_group::load(CVT *cv_src) const {
  T *src = const_cast<T *>(cv_src);
  return load(sycl::multi_ptr<remove_decoration_t<T>,
                              sycl::detail::deduce_AS<T>::value,
                              sycl::access::decorated::yes>(src));
}

template <typename CVT, typename T>
inline std::enable_if_t<std::is_same<remove_decoration_t<T>, T>::value, T>
sub_group::load(CVT *cv_src) const {
  T *src = const_cast<T *>(cv_src);

#if defined(__NVPTX__) || defined(__AMDGCN__)
  return src[get_local_id()[0]];
#else  // __NVPTX__ || __AMDGCN__
  if (auto l =
          detail::dynamic_address_cast<access::address_space::local_space>(
              src))
    return load(l);

  if (auto g =
          detail::dynamic_address_cast<access::address_space::global_space>(
              src))
    return load(g);

  // Sub-group load() is supported for local or global pointers only.
  return {};
#endif // __NVPTX__ || __AMDGCN__
}
#else  //__SYCL_DEVICE_ONLY__
template <typename CVT, typename T> inline T sub_group::load(CVT *src) const {
  (void)src;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
}
#endif //__SYCL_DEVICE_ONLY__

template <typename CVT, access::address_space Space,
          access::decorated IsDecorated, typename T>
inline std::enable_if_t<
    sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value, T>
sub_group::load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const {
  multi_ptr<T, Space, IsDecorated> src = sycl::detail::GetUnqualMultiPtr(cv_src);
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__) || defined(__AMDGCN__)
  return src.get()[get_local_id()[0]];
#else
  return sycl::detail::sub_group::load(src);
#endif // __NVPTX__ || __AMDGCN__
#else
  (void)src;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif // __SYCL_DEVICE_ONLY__
}

template <typename CVT, access::address_space Space,
          access::decorated IsDecorated, typename T>
inline std::enable_if_t<
    sycl::detail::sub_group::AcceptableForLocalLoadStore<T, Space>::value, T>
sub_group::load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const {
  multi_ptr<T, Space, IsDecorated> src = sycl::detail::GetUnqualMultiPtr(cv_src);
#ifdef __SYCL_DEVICE_ONLY__
  return src.get()[get_local_id()[0]];
#else
  (void)src;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__) || defined(__AMDGCN__)
template <int N, typename CVT, access::address_space Space,
          access::decorated IsDecorated, typename T>
inline std::enable_if_t<
    sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value,
    vec<T, N>>
sub_group::load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const {
  multi_ptr<T, Space, IsDecorated> src = sycl::detail::GetUnqualMultiPtr(cv_src);
  vec<T, N> res;
  for (int i = 0; i < N; ++i) {
    res[i] = *(src.get() + i * get_max_local_range()[0] + get_local_id()[0]);
  }
  return res;
}
#else  // __NVPTX__ || __AMDGCN__
template <int N, typename CVT, access::address_space Space,
          access::decorated IsDecorated, typename T>
inline std::enable_if_t<
    sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
        N != 1 && N != 3 && N != 16,
    vec<T, N>>
sub_group::load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const {
  multi_ptr<T, Space, IsDecorated> src = sycl::detail::GetUnqualMultiPtr(cv_src);
  return sycl::detail::sub_group::load<N, T>(src);
}

template <int N, typename CVT, access::address_space Space,
          access::decorated IsDecorated, typename T>
inline std::enable_if_t<
    sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
        N == 16,
    vec<T, 16>>
sub_group::load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const {
  multi_ptr<T, Space, IsDecorated> src = sycl::detail::GetUnqualMultiPtr(cv_src);
  return {sycl::detail::sub_group::load<8, T>(src),
          sycl::detail::sub_group::load<8, T>(src + 8 * get_max_local_range()[0])};
}

template <int N, typename CVT, access::address_space Space,
          access::decorated IsDecorated, typename T>
inline std::enable_if_t<
    sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
        N == 3,
    vec<T, 3>>
sub_group::load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const {
  multi_ptr<T, Space, IsDecorated> src = sycl::detail::GetUnqualMultiPtr(cv_src);
  return {sycl::detail::sub_group::load<1, T>(src),
          sycl::detail::sub_group::load<2, T>(src + get_max_local_range()[0])};
}

template <int N, typename CVT, access::address_space Space,
          access::decorated IsDecorated, typename T>
inline std::enable_if_t<
    sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
        N == 1,
    vec<T, 1>>
sub_group::load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const {
  multi_ptr<T, Space, IsDecorated> src = sycl::detail::GetUnqualMultiPtr(cv_src);
  return sycl::detail::sub_group::load(src);
}
#endif // ___NVPTX___
#else  // __SYCL_DEVICE_ONLY__
template <int N, typename CVT, access::address_space Space,
          access::decorated IsDecorated, typename T>
inline std::enable_if_t<
    sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value,
    vec<T, N>>
sub_group::load(const multi_ptr<CVT, Space, IsDecorated> src) const {
  (void)src;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
}
#endif // __SYCL_DEVICE_ONLY__

template <int N, typename CVT, access::address_space Space,
          access::decorated IsDecorated, typename T>
inline std::enable_if_t<
    sycl::detail::sub_group::AcceptableForLocalLoadStore<T, Space>::value,
    vec<T, N>>
sub_group::load(const multi_ptr<CVT, Space, IsDecorated> cv_src) const {
  multi_ptr<T, Space, IsDecorated> src = sycl::detail::GetUnqualMultiPtr(cv_src);
#ifdef __SYCL_DEVICE_ONLY__
  vec<T, N> res;
  for (int i = 0; i < N; ++i) {
    res[i] = *(src.get() + i * get_max_local_range()[0] + get_local_id()[0]);
  }
  return res;
#else
  (void)src;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

#ifdef __SYCL_DEVICE_ONLY__
template <typename T>
inline std::enable_if_t<!std::is_same<remove_decoration_t<T>, T>::value>
sub_group::store(T *dst, const remove_decoration_t<T> &x) const {
  store(sycl::multi_ptr<remove_decoration_t<T>,
                        sycl::detail::deduce_AS<T>::value,
                        sycl::access::decorated::yes>(dst),
        x);
}

template <typename T>
inline std::enable_if_t<std::is_same<remove_decoration_t<T>, T>::value>
sub_group::store(T *dst, const remove_decoration_t<T> &x) const {

#if defined(__NVPTX__) || defined(__AMDGCN__)
  dst[get_local_id()[0]] = x;
#else  // __NVPTX__ || __AMDGCN__
  if (auto l =
          detail::dynamic_address_cast<access::address_space::local_space>(dst)) {
    store(l, x);
    return;
  }

  if (auto g =
          detail::dynamic_address_cast<access::address_space::global_space>(dst)) {
    store(g, x);
    return;
  }

  // Sub-group store() is supported for local or global pointers only.
  return;
#endif // __NVPTX__ || __AMDGCN__
}
#else  //__SYCL_DEVICE_ONLY__
template <typename T> inline void sub_group::store(T *dst, const T &x) const {
  (void)dst;
  (void)x;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
}
#endif //__SYCL_DEVICE_ONLY__

template <typename T, access::address_space Space,
          access::decorated DecorateAddress>
inline std::enable_if_t<
    sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value>
sub_group::store(multi_ptr<T, Space, DecorateAddress> dst, const T &x) const {
#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__) || defined(__AMDGCN__)
  dst.get()[get_local_id()[0]] = x;
#else
  sycl::detail::sub_group::store(dst, x);
#endif // __NVPTX__ || __AMDGCN__
#else
  (void)dst;
  (void)x;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

template <typename T, access::address_space Space,
          access::decorated DecorateAddress>
inline std::enable_if_t<
    sycl::detail::sub_group::AcceptableForLocalLoadStore<T, Space>::value>
sub_group::store(multi_ptr<T, Space, DecorateAddress> dst, const T &x) const {
#ifdef __SYCL_DEVICE_ONLY__
  dst.get()[get_local_id()[0]] = x;
#else
  (void)dst;
  (void)x;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

#ifdef __SYCL_DEVICE_ONLY__
#if defined(__NVPTX__) || defined(__AMDGCN__)
template <int N, typename T, access::address_space Space,
          access::decorated DecorateAddress>
inline std::enable_if_t<
    sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value>
sub_group::store(multi_ptr<T, Space, DecorateAddress> dst,
                 const vec<T, N> &x) const {
  for (int i = 0; i < N; ++i) {
    *(dst.get() + i * get_max_local_range()[0] + get_local_id()[0]) = x[i];
  }
}
#else // __NVPTX__ || __AMDGCN__
template <int N, typename T, access::address_space Space,
          access::decorated DecorateAddress>
inline std::enable_if_t<
    sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
    N != 1 && N != 3 && N != 16>
sub_group::store(multi_ptr<T, Space, DecorateAddress> dst,
                 const vec<T, N> &x) const {
  sycl::detail::sub_group::store(dst, x);
}

template <int N, typename T, access::address_space Space,
          access::decorated DecorateAddress>
inline std::enable_if_t<
    sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
    N == 1>
sub_group::store(multi_ptr<T, Space, DecorateAddress> dst,
                 const vec<T, 1> &x) const {
  sycl::detail::sub_group::store(dst, x);
}

template <int N, typename T, access::address_space Space,
          access::decorated DecorateAddress>
inline std::enable_if_t<
    sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
    N == 3>
sub_group::store(multi_ptr<T, Space, DecorateAddress> dst,
                 const vec<T, 3> &x) const {
  store<1, T, Space, DecorateAddress>(dst, x.s0());
  store<2, T, Space, DecorateAddress>(dst + get_max_local_range()[0],
                                      {x.s1(), x.s2()});
}

template <int N, typename T, access::address_space Space,
          access::decorated DecorateAddress>
inline std::enable_if_t<
    sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value &&
    N == 16>
sub_group::store(multi_ptr<T, Space, DecorateAddress> dst,
                 const vec<T, 16> &x) const {
  store<8, T, Space, DecorateAddress>(dst, x.lo());
  store<8, T, Space, DecorateAddress>(dst + 8 * get_max_local_range()[0],
                                      x.hi());
}

#endif // __NVPTX__ || __AMDGCN__
#else  // __SYCL_DEVICE_ONLY__
template <int N, typename T, access::address_space Space,
          access::decorated DecorateAddress>
inline std::enable_if_t<
    sycl::detail::sub_group::AcceptableForGlobalLoadStore<T, Space>::value>
sub_group::store(multi_ptr<T, Space, DecorateAddress> dst,
                 const vec<T, N> &x) const {
  (void)dst;
  (void)x;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
}
#endif // __SYCL_DEVICE_ONLY__

template <int N, typename T, access::address_space Space,
          access::decorated DecorateAddress>
inline std::enable_if_t<
    sycl::detail::sub_group::AcceptableForLocalLoadStore<T, Space>::value>
sub_group::store(multi_ptr<T, Space, DecorateAddress> dst,
                 const vec<T, N> &x) const {
#ifdef __SYCL_DEVICE_ONLY__
  for (int i = 0; i < N; ++i) {
    *(dst.get() + i * get_max_local_range()[0] + get_local_id()[0]) = x[i];
  }
#else
  (void)dst;
  (void)x;
  throw sycl::exception(make_error_code(errc::feature_not_supported),
                        "Sub-groups are not supported on host.");
#endif
}

} // namespace _V1
} // namespace sycl