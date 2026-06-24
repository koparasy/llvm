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

// ---------------------------------------------------------------------------
// Step-3: the ceiling narrows. Trivially-copyable types that ARE, or
// recursively CONTAIN, a layout-unstable scalar (long double / __int128) are
// rejected. The recursion is the point: nested-in-struct/base/array all count.
// ---------------------------------------------------------------------------

// Direct unstable scalars -> rejected.
static_assert(!__is_valid_sycl_kernel_arg(long double));
static_assert(!__is_valid_sycl_kernel_arg(__int128));
static_assert(!__is_valid_sycl_kernel_arg(unsigned __int128));

// Nested in a struct -> rejected.
struct HasLD {
  int a;
  long double b;
};
static_assert(!__is_valid_sycl_kernel_arg(HasLD));

// Nested deeper / via base / via array -> rejected.
struct WrapsHasLD {
  HasLD inner;
};
static_assert(!__is_valid_sycl_kernel_arg(WrapsHasLD));

struct DerivedLD : HasLD {
  int c;
};
static_assert(!__is_valid_sycl_kernel_arg(DerivedLD));

static_assert(!__is_valid_sycl_kernel_arg(long double[4]));

struct HasLDArray {
  double x;
  long double arr[2];
};
static_assert(!__is_valid_sycl_kernel_arg(HasLDArray));

// An __int128 nested in a struct -> rejected (the other unstable scalar kind).
struct HasI128 {
  int a;
  __int128 b;
};
static_assert(!__is_valid_sycl_kernel_arg(HasI128));

// A trivially-copyable struct of only stable scalars -> still valid.
struct StableNest {
  struct Inner {
    int i;
    double d;
  } n;
  float f;
  int arr[3];
};
static_assert(__is_valid_sycl_kernel_arg(StableNest));

// ---------------------------------------------------------------------------
// Expanded layout-unstable raw-builtin scalar set. Matched on BuiltinType
// kind (NOT byte size, NOT library wrappers). Enums resolve to underlying.
// ---------------------------------------------------------------------------

// Direct unstable raw builtins -> rejected.
static_assert(!__is_valid_sycl_kernel_arg(_Float16));
static_assert(!__is_valid_sycl_kernel_arg(__bf16));
static_assert(!__is_valid_sycl_kernel_arg(wchar_t));
static_assert(!__is_valid_sycl_kernel_arg(long));
static_assert(!__is_valid_sycl_kernel_arg(unsigned long));

// Nested unstable builtins -> rejected (recursion is the point).
struct HasLong {
  int a;
  long b;
};
static_assert(!__is_valid_sycl_kernel_arg(HasLong));

struct HasHalf {
  float f;
  _Float16 h;
};
static_assert(!__is_valid_sycl_kernel_arg(HasHalf));

struct WrapsLong {
  HasLong inner;
};
static_assert(!__is_valid_sycl_kernel_arg(WrapsLong));

struct DerivedLong : HasLong {};
static_assert(!__is_valid_sycl_kernel_arg(DerivedLong));

// Enum whose underlying type is unstable -> rejected (resolve underlying).
enum ELong : long { e };
static_assert(!__is_valid_sycl_kernel_arg(ELong));

// CONTROL -- stable types must stay VALID (guard against over-rejection).
static_assert(__is_valid_sycl_kernel_arg(int));
static_assert(__is_valid_sycl_kernel_arg(float));
static_assert(__is_valid_sycl_kernel_arg(double));
static_assert(__is_valid_sycl_kernel_arg(long long)); // fixed 64-bit -> allowed
static_assert(__is_valid_sycl_kernel_arg(char));
static_assert(__is_valid_sycl_kernel_arg(int *));
struct StableNest2 {
  struct In {
    int i;
    double d;
  } n;
  float arr[3];
};
static_assert(__is_valid_sycl_kernel_arg(StableNest2));
