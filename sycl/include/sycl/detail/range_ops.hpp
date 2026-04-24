//==--------------- range_ops.hpp --- SYCL iteration range ops -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/range_core.hpp>

namespace sycl {
inline namespace _V1 {

// OP is: +, -, *, /, %, <<, >>, &, |, ^, &&, ||, <, >, <=, >=
#define __SYCL_GEN_OPT_BASE(op)                                                \
  template <int Dimensions>                                                    \
  range<Dimensions> operator op(const range<Dimensions> &lhs,                  \
                                const range<Dimensions> &rhs) {                \
    range<Dimensions> result(lhs);                                             \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result[i] = lhs[i] op rhs[i];                                            \
    }                                                                          \
    return result;                                                             \
  }

#ifndef __SYCL_DISABLE_ID_TO_INT_CONV__
// Enable operators with integral types only
#define __SYCL_GEN_OPT(op)                                                     \
  __SYCL_GEN_OPT_BASE(op)                                                      \
  template <int Dimensions, typename T>                                        \
  std::enable_if_t<std::is_integral_v<T>, range<Dimensions>> operator op(      \
      const range<Dimensions> &lhs, const T &rhs) {                            \
    range<Dimensions> result(lhs);                                             \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result[i] = lhs[i] op rhs;                                               \
    }                                                                          \
    return result;                                                             \
  }                                                                            \
  template <int Dimensions, typename T>                                        \
  std::enable_if_t<std::is_integral_v<T>, range<Dimensions>> operator op(      \
      const T &lhs, const range<Dimensions> &rhs) {                            \
    range<Dimensions> result(rhs);                                             \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result[i] = lhs op rhs[i];                                               \
    }                                                                          \
    return result;                                                             \
  }
#else
#define __SYCL_GEN_OPT(op)                                                     \
  __SYCL_GEN_OPT_BASE(op)                                                      \
  template <int Dimensions>                                                    \
  range<Dimensions> operator op(const range<Dimensions> &lhs,                  \
                                const size_t &rhs) {                           \
    range<Dimensions> result(lhs);                                             \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result[i] = lhs[i] op rhs;                                               \
    }                                                                          \
    return result;                                                             \
  }                                                                            \
  template <int Dimensions>                                                    \
  range<Dimensions> operator op(const size_t &lhs,                             \
                                const range<Dimensions> &rhs) {                \
    range<Dimensions> result(rhs);                                             \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result[i] = lhs op rhs[i];                                               \
    }                                                                          \
    return result;                                                             \
  }
#endif // __SYCL_DISABLE_ID_TO_INT_CONV__

__SYCL_GEN_OPT(+)
__SYCL_GEN_OPT(-)
__SYCL_GEN_OPT(*)
__SYCL_GEN_OPT(/)
__SYCL_GEN_OPT(%)
__SYCL_GEN_OPT(<<)
__SYCL_GEN_OPT(>>)
__SYCL_GEN_OPT(&)
__SYCL_GEN_OPT(|)
__SYCL_GEN_OPT(^)
__SYCL_GEN_OPT(&&)
__SYCL_GEN_OPT(||)
__SYCL_GEN_OPT(<)
__SYCL_GEN_OPT(>)
__SYCL_GEN_OPT(<=)
__SYCL_GEN_OPT(>=)

#undef __SYCL_GEN_OPT
#undef __SYCL_GEN_OPT_BASE

// OP is: +=, -=, *=, /=, %=, <<=, >>=, &=, |=, ^=
#define __SYCL_GEN_OPT(op)                                                     \
  template <int Dimensions>                                                    \
  range<Dimensions> &operator op(range<Dimensions> &lhs,                       \
                                 const range<Dimensions> &rhs) {               \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      lhs[i] op rhs[i];                                                        \
    }                                                                          \
    return lhs;                                                                \
  }                                                                            \
  template <int Dimensions>                                                    \
  range<Dimensions> &operator op(range<Dimensions> &lhs, const size_t &rhs) {  \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      lhs[i] op rhs;                                                           \
    }                                                                          \
    return lhs;                                                                \
  }

__SYCL_GEN_OPT(+=)
__SYCL_GEN_OPT(-=)
__SYCL_GEN_OPT(*=)
__SYCL_GEN_OPT(/=)
__SYCL_GEN_OPT(%=)
__SYCL_GEN_OPT(<<=)
__SYCL_GEN_OPT(>>=)
__SYCL_GEN_OPT(&=)
__SYCL_GEN_OPT(|=)
__SYCL_GEN_OPT(^=)

#undef __SYCL_GEN_OPT

// OP is unary +, -
#define __SYCL_GEN_OPT(op)                                                     \
  template <int Dimensions>                                                    \
  range<Dimensions> operator op(const range<Dimensions> &rhs) {                \
    range<Dimensions> result(rhs);                                             \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result[i] = (op rhs[i]);                                                 \
    }                                                                          \
    return result;                                                             \
  }

__SYCL_GEN_OPT(+)
__SYCL_GEN_OPT(-)

#undef __SYCL_GEN_OPT

// OP is prefix ++, --
#define __SYCL_GEN_OPT(op)                                                     \
  template <int Dimensions>                                                    \
  range<Dimensions> &operator op(range<Dimensions> &rhs) {                     \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      op rhs[i];                                                               \
    }                                                                          \
    return rhs;                                                                \
  }

__SYCL_GEN_OPT(++)
__SYCL_GEN_OPT(--)

#undef __SYCL_GEN_OPT

// OP is postfix ++, --
#define __SYCL_GEN_OPT(op)                                                     \
  template <int Dimensions>                                                    \
  range<Dimensions> operator op(range<Dimensions> &lhs, int) {                 \
    range<Dimensions> old_lhs(lhs);                                            \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      op lhs[i];                                                               \
    }                                                                          \
    return old_lhs;                                                            \
  }

__SYCL_GEN_OPT(++)
__SYCL_GEN_OPT(--)

#undef __SYCL_GEN_OPT

} // namespace _V1
} // namespace sycl