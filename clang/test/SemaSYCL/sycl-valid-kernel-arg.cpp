// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -std=c++17 -verify %s
// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -std=c++17 -verify %s
// expected-no-diagnostics

// Tests the __is_valid_sycl_kernel_arg unary type-trait builtin that backs
// sycl::khr::is_valid_kernel_arg_v<T>. Step-2 semantics: true iff the type is
// trivially copyable (a ceiling, not a floor). This test deliberately uses the
// builtin DIRECTLY with no SYCL runtime headers so it stands fully on its own.

// Scalars and pointers are valid.
static_assert(__is_valid_sycl_kernel_arg(int));
static_assert(__is_valid_sycl_kernel_arg(float *));
static_assert(__is_valid_sycl_kernel_arg(double));

// A trivially-copyable POD aggregate is valid.
struct Pod {
  int a;
  float b;
};
static_assert(__is_valid_sycl_kernel_arg(Pod));

// Nested PODs are still trivially copyable -> valid.
struct NestedPod {
  Pod p;
  int c;
};
static_assert(__is_valid_sycl_kernel_arg(NestedPod));

// A user-provided copy constructor makes the type non-trivially-copyable.
struct NonTC {
  NonTC(const NonTC &) {}
};
static_assert(!__is_valid_sycl_kernel_arg(NonTC));

// A user-provided destructor makes the type non-trivially-copyable.
struct WithDtor {
  ~WithDtor();
};
static_assert(!__is_valid_sycl_kernel_arg(WithDtor));

// References are not trivially copyable types.
static_assert(!__is_valid_sycl_kernel_arg(int &));

// The builtin is usable in a dependent context and folds after substitution.
template <typename T> constexpr bool valid() {
  return __is_valid_sycl_kernel_arg(T);
}
static_assert(valid<Pod>());
static_assert(!valid<NonTC>());
