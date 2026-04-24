//==---------------- id_ops.hpp --- SYCL iteration id ops -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/id_core.hpp>

namespace sycl {
inline namespace _V1 {

#ifndef __SYCL_DISABLE_ID_TO_INT_CONV__
template <int Dimensions, typename T>
std::enable_if_t<std::is_integral_v<T>, bool>
operator==(const T &lhs, const id<Dimensions> &rhs) {
  if (lhs != rhs[0])
    return false;
  return true;
}

template <int Dimensions, typename T>
std::enable_if_t<std::is_integral_v<T>, bool>
operator!=(const T &lhs, const id<Dimensions> &rhs) {
  if (lhs != rhs[0])
    return true;
  return false;
}
#endif // __SYCL_DISABLE_ID_TO_INT_CONV__

// OP is: +, -, *, /, %, <<, >>, &, |, ^, &&, ||, <, >, <=, >=
#define __SYCL_GEN_OPT_BASE(op)                                                \
  template <int Dimensions>                                                    \
  id<Dimensions> operator op(const id<Dimensions> &lhs,                        \
                             const id<Dimensions> &rhs) {                      \
    id<Dimensions> result;                                                     \
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
  std::enable_if_t<std::is_integral_v<T>, id<Dimensions>> operator op(         \
      const id<Dimensions> &lhs, const T &rhs) {                               \
    id<Dimensions> result;                                                     \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result[i] = lhs[i] op rhs;                                               \
    }                                                                          \
    return result;                                                             \
  }                                                                            \
  template <int Dimensions, typename T>                                        \
  std::enable_if_t<std::is_integral_v<T>, id<Dimensions>> operator op(         \
      const T &lhs, const id<Dimensions> &rhs) {                               \
    id<Dimensions> result;                                                     \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result[i] = lhs op rhs[i];                                               \
    }                                                                          \
    return result;                                                             \
  }
#else
#define __SYCL_GEN_OPT(op)                                                     \
  __SYCL_GEN_OPT_BASE(op)                                                      \
  template <int Dimensions>                                                    \
  id<Dimensions> operator op(const id<Dimensions> &lhs, const size_t &rhs) {   \
    id<Dimensions> result;                                                     \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      result[i] = lhs[i] op rhs;                                               \
    }                                                                          \
    return result;                                                             \
  }                                                                            \
  template <int Dimensions>                                                    \
  id<Dimensions> operator op(const size_t &lhs, const id<Dimensions> &rhs) {   \
    id<Dimensions> result;                                                     \
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
  id<Dimensions> &operator op(id<Dimensions> &lhs, const id<Dimensions> &rhs) {\
    for (int i = 0; i < Dimensions; ++i) {                                     \
      lhs[i] op rhs[i];                                                        \
    }                                                                          \
    return lhs;                                                                \
  }                                                                            \
  template <int Dimensions>                                                    \
  id<Dimensions> &operator op(id<Dimensions> &lhs, const size_t &rhs) {        \
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
  id<Dimensions> operator op(const id<Dimensions> &rhs) {                      \
    id<Dimensions> result;                                                     \
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
  id<Dimensions> &operator op(id<Dimensions> &rhs) {                           \
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
  id<Dimensions> operator op(id<Dimensions> &lhs, int) {                       \
    id<Dimensions> old_lhs;                                                    \
    for (int i = 0; i < Dimensions; ++i) {                                     \
      old_lhs[i] = lhs[i];                                                     \
      op lhs[i];                                                               \
    }                                                                          \
    return old_lhs;                                                            \
  }

__SYCL_GEN_OPT(++)
__SYCL_GEN_OPT(--)

#undef __SYCL_GEN_OPT

} // namespace _V1
} // namespace sycl