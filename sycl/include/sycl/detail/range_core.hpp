//==------------ range_core.hpp --- SYCL iteration range core ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/array.hpp> // for array

#include <stddef.h>    // for size_t
#include <type_traits> // for enable_if_t

namespace sycl {
inline namespace _V1 {
template <int Dimensions> class id;

namespace detail {
class Builder;
}

/// Defines the iteration domain of either a single work-group in a parallel
/// dispatch, or the overall Dimensions of the dispatch.
///
/// \ingroup sycl_api
template <int Dimensions = 1> class range : public detail::array<Dimensions> {
public:
  static constexpr int dimensions = Dimensions;

private:
  static_assert(Dimensions >= 1 && Dimensions <= 3,
                "range can only be 1, 2, or 3 Dimensional.");
  using base = detail::array<Dimensions>;
  template <typename N, typename T>
  using IntegralType = std::enable_if_t<std::is_integral_v<N>, T>;

public:
  /* The following constructor is only available in the range class
  specialization where: Dimensions==1 */
  template <int N = Dimensions>
  range(typename std::enable_if_t<(N == 1), size_t> dim0) : base(dim0) {}

  /* The following constructor is only available in the range class
  specialization where: Dimensions==2 */
  template <int N = Dimensions>
  range(typename std::enable_if_t<(N == 2), size_t> dim0, size_t dim1)
      : base(dim0, dim1) {}

  /* The following constructor is only available in the range class
  specialization where: Dimensions==3 */
  template <int N = Dimensions>
  range(typename std::enable_if_t<(N == 3), size_t> dim0, size_t dim1,
        size_t dim2)
      : base(dim0, dim1, dim2) {}

  size_t size() const {
    size_t size = 1;
    for (int i = 0; i < Dimensions; ++i) {
      size *= this->common_array[i];
    }
    return size;
  }

  range(const range<Dimensions> &rhs) = default;
  range(range<Dimensions> &&rhs) = default;
  range<Dimensions> &operator=(const range<Dimensions> &rhs) = default;
  range<Dimensions> &operator=(range<Dimensions> &&rhs) = default;
  range() = default;

#define __SYCL_GEN_FRIEND_OP_DECL(op)                                          \
  template <int Dims>                                                          \
  friend range<Dims> operator op(const range<Dims> &lhs,                      \
                                 const range<Dims> &rhs);                     \
  template <int Dims, typename T>                                              \
  friend std::enable_if_t<std::is_integral_v<T>, range<Dims>> operator op(    \
      const range<Dims> &lhs, const T &rhs);                                   \
  template <int Dims, typename T>                                              \
  friend std::enable_if_t<std::is_integral_v<T>, range<Dims>> operator op(    \
      const T &lhs, const range<Dims> &rhs)

#ifndef __SYCL_DISABLE_ID_TO_INT_CONV__
  __SYCL_GEN_FRIEND_OP_DECL(+);
  __SYCL_GEN_FRIEND_OP_DECL(-);
  __SYCL_GEN_FRIEND_OP_DECL(*);
  __SYCL_GEN_FRIEND_OP_DECL(/);
  __SYCL_GEN_FRIEND_OP_DECL(%);
  __SYCL_GEN_FRIEND_OP_DECL(<<);
  __SYCL_GEN_FRIEND_OP_DECL(>>);
  __SYCL_GEN_FRIEND_OP_DECL(&);
  __SYCL_GEN_FRIEND_OP_DECL(|);
  __SYCL_GEN_FRIEND_OP_DECL(^);
  __SYCL_GEN_FRIEND_OP_DECL(&&);
  __SYCL_GEN_FRIEND_OP_DECL(||);
  __SYCL_GEN_FRIEND_OP_DECL(<);
  __SYCL_GEN_FRIEND_OP_DECL(>);
  __SYCL_GEN_FRIEND_OP_DECL(<=);
  __SYCL_GEN_FRIEND_OP_DECL(>=);
#else
  template <int Dims>
  friend range<Dims> operator+(const range<Dims> &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator+(const range<Dims> &lhs, const size_t &rhs);
  template <int Dims>
  friend range<Dims> operator+(const size_t &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator-(const range<Dims> &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator-(const range<Dims> &lhs, const size_t &rhs);
  template <int Dims>
  friend range<Dims> operator-(const size_t &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator*(const range<Dims> &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator*(const range<Dims> &lhs, const size_t &rhs);
  template <int Dims>
  friend range<Dims> operator*(const size_t &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator/(const range<Dims> &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator/(const range<Dims> &lhs, const size_t &rhs);
  template <int Dims>
  friend range<Dims> operator/(const size_t &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator%(const range<Dims> &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator%(const range<Dims> &lhs, const size_t &rhs);
  template <int Dims>
  friend range<Dims> operator%(const size_t &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator<<(const range<Dims> &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator<<(const range<Dims> &lhs, const size_t &rhs);
  template <int Dims>
  friend range<Dims> operator<<(const size_t &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator>>(const range<Dims> &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator>>(const range<Dims> &lhs, const size_t &rhs);
  template <int Dims>
  friend range<Dims> operator>>(const size_t &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator&(const range<Dims> &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator&(const range<Dims> &lhs, const size_t &rhs);
  template <int Dims>
  friend range<Dims> operator&(const size_t &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator|(const range<Dims> &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator|(const range<Dims> &lhs, const size_t &rhs);
  template <int Dims>
  friend range<Dims> operator|(const size_t &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator^(const range<Dims> &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator^(const range<Dims> &lhs, const size_t &rhs);
  template <int Dims>
  friend range<Dims> operator^(const size_t &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator&&(const range<Dims> &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator&&(const range<Dims> &lhs, const size_t &rhs);
  template <int Dims>
  friend range<Dims> operator&&(const size_t &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator||(const range<Dims> &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator||(const range<Dims> &lhs, const size_t &rhs);
  template <int Dims>
  friend range<Dims> operator||(const size_t &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator<(const range<Dims> &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator<(const range<Dims> &lhs, const size_t &rhs);
  template <int Dims>
  friend range<Dims> operator<(const size_t &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator>(const range<Dims> &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator>(const range<Dims> &lhs, const size_t &rhs);
  template <int Dims>
  friend range<Dims> operator>(const size_t &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator<=(const range<Dims> &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator<=(const range<Dims> &lhs, const size_t &rhs);
  template <int Dims>
  friend range<Dims> operator<=(const size_t &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator>=(const range<Dims> &lhs, const range<Dims> &rhs);
  template <int Dims>
  friend range<Dims> operator>=(const range<Dims> &lhs, const size_t &rhs);
  template <int Dims>
  friend range<Dims> operator>=(const size_t &lhs, const range<Dims> &rhs);
#endif

#undef __SYCL_GEN_FRIEND_OP_DECL

#define __SYCL_GEN_FRIEND_ASSIGN_DECL(op)                                      \
  template <int Dims>                                                          \
  friend range<Dims> &operator op(range<Dims> &lhs, const range<Dims> &rhs); \
  template <int Dims>                                                          \
  friend range<Dims> &operator op(range<Dims> &lhs, const size_t &rhs)

  __SYCL_GEN_FRIEND_ASSIGN_DECL(+=);
  __SYCL_GEN_FRIEND_ASSIGN_DECL(-=);
  __SYCL_GEN_FRIEND_ASSIGN_DECL(*=);
  __SYCL_GEN_FRIEND_ASSIGN_DECL(/=);
  __SYCL_GEN_FRIEND_ASSIGN_DECL(%=);
  __SYCL_GEN_FRIEND_ASSIGN_DECL(<<=);
  __SYCL_GEN_FRIEND_ASSIGN_DECL(>>=);
  __SYCL_GEN_FRIEND_ASSIGN_DECL(&=);
  __SYCL_GEN_FRIEND_ASSIGN_DECL(|=);
  __SYCL_GEN_FRIEND_ASSIGN_DECL(^=);

#undef __SYCL_GEN_FRIEND_ASSIGN_DECL

#define __SYCL_GEN_FRIEND_UNARY_DECL(op)                                       \
  template <int Dims>                                                          \
  friend range<Dims> operator op(const range<Dims> &rhs)

  __SYCL_GEN_FRIEND_UNARY_DECL(+);
  __SYCL_GEN_FRIEND_UNARY_DECL(-);

#undef __SYCL_GEN_FRIEND_UNARY_DECL

#define __SYCL_GEN_FRIEND_INCDEC_DECL(op)                                      \
  template <int Dims>                                                          \
  friend range<Dims> &operator op(range<Dims> &rhs);                          \
  template <int Dims>                                                          \
  friend range<Dims> operator op(range<Dims> &lhs, int)

  __SYCL_GEN_FRIEND_INCDEC_DECL(++);
  __SYCL_GEN_FRIEND_INCDEC_DECL(--);

#undef __SYCL_GEN_FRIEND_INCDEC_DECL

private:
  friend class handler;
  friend class detail::Builder;

  // Adjust the first dim of the range
  void set_range_dim0(const size_t dim0) { this->common_array[0] = dim0; }
};

#ifdef __cpp_deduction_guides
range(size_t)->range<1>;
range(size_t, size_t)->range<2>;
range(size_t, size_t, size_t)->range<3>;
#endif

} // namespace _V1
} // namespace sycl