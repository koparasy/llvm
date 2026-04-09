// RUN: %clangxx -fsycl -fsyntax-only -Wno-deprecated-declarations %s -DTEST_MATH_HEADER
// RUN: %clangxx -fsycl -fsyntax-only -Wno-deprecated-declarations %s -DTEST_RELATIONAL_HEADER
// RUN: %clangxx -fsycl -fsyntax-only -Wno-deprecated-declarations %s -DTEST_KHR_MATH_HEADER
// RUN: %clangxx -fsycl -fsyntax-only -Wno-deprecated-declarations %s -DTEST_MATH_WITH_BUILTINS
// RUN: %clangxx -fsycl -fsyntax-only -Wno-deprecated-declarations %s -DTEST_BUILTINS_WITH_MATH
// RUN: %clangxx -fsycl -fsyntax-only -Wno-deprecated-declarations %s -DTEST_RELATIONAL_WITH_BUILTINS

// Regression coverage for subset builtin headers.
// We want to preserve these behaviors:
// 1. <sycl/math.hpp> exposes generic math, native math, geometric, and
//    pointer-based math builtins without requiring <sycl/builtins.hpp>.
// 2. <sycl/relational.hpp> exposes relational builtins without requiring
//    <sycl/builtins.hpp>.
// 3. <sycl/khr/includes/math.hpp> follows the smaller math header path.

#if defined(TEST_MATH_HEADER)
#include <sycl/math.hpp>

int main() {
  float Storage = 0.0f;
  sycl::vec<float, 2> Value{1.0f, 2.0f};
  sycl::vec<float, 2> FractStorage{};

  auto Minimum = sycl::fmin(1.0f, 2.0f);
  auto Native = sycl::native::sin(1.0f);
  auto Dot = sycl::dot(Value, Value);
  auto Fract = sycl::fract(Value, &FractStorage);
  auto Modf = sycl::modf(1.0f, &Storage);

  (void)Minimum;
  (void)Native;
  (void)Dot;
  (void)Fract;
  (void)Modf;
  return 0;
}

#elif defined(TEST_RELATIONAL_HEADER)
#include <sycl/relational.hpp>

int main() {
  sycl::vec<float, 2> Value{1.0f, 0.0f / 0.0f};
  sycl::vec<int32_t, 2> Mask{-1, 0};

  auto Scalar = sycl::isfinite(1.0f);
  auto Vector = sycl::isnan(Value);
  auto Any = sycl::any(Mask);
  auto All = sycl::all(Mask);

  (void)Scalar;
  (void)Vector;
  (void)Any;
  (void)All;
  return 0;
}

#elif defined(TEST_KHR_MATH_HEADER)
#include <sycl/khr/includes/math.hpp>

int main() {
  auto Value = sycl::native::sqrt(4.0f);
  auto Minimum = sycl::fmin(1.0f, 2.0f);
  (void)Value;
  (void)Minimum;
  return 0;
}

#elif defined(TEST_MATH_WITH_BUILTINS)
#include <sycl/math.hpp>
#include <sycl/builtins.hpp>

int main() {
  auto Value = sycl::fmin(1.0f, 2.0f);
  (void)Value;
  return 0;
}

#elif defined(TEST_BUILTINS_WITH_MATH)
#include <sycl/builtins.hpp>
#include <sycl/math.hpp>

int main() {
  auto Value = sycl::native::sin(1.0f);
  (void)Value;
  return 0;
}

#elif defined(TEST_RELATIONAL_WITH_BUILTINS)
#include <sycl/relational.hpp>
#include <sycl/builtins.hpp>

int main() {
  auto Value = sycl::isfinite(1.0f);
  (void)Value;
  return 0;
}
#endif