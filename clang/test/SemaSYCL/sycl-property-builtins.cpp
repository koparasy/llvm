// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -std=c++17 -verify %s
// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -std=c++17 -verify %s
// expected-no-diagnostics

// Tests the __builtin_sycl_has_property / __builtin_sycl_get_property builtins
// that back sycl::khr::is_kernel_v / is_nd_kernel_v / is_single_task_kernel_v.
//
// Step-4 semantics: a free function kernel's properties are DECLARATION-LOCAL.
// The decoration attribute is attached to the host-visible FunctionDecl in both
// the host and the device pass (it only generates LLVM IR in the device pass,
// so it is inert on the host), so these builtins fold to compile-time constants
// at host parse time WITHOUT any integration header. This test runs the host
// pass with NO SYCL runtime headers and NO integration header at all -- it
// stands entirely on its own -- which is exactly the property being proven.

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 2)]]
void k_nd(int *) {}

[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void k_st(int *) {}

// add_ir_attributes_function takes all the names first, then all the values
// (N-names-then-N-values), so this carries two properties:
//   "sycl-nd-range-kernel" = 3, "sycl-work-group-size" = "8,8,8".
[[__sycl_detail__::add_ir_attributes_function(
    "sycl-nd-range-kernel", "sycl-work-group-size", 3, "8,8,8")]]
void k_multi(int *) {}

void plain(int *) {}

template <typename T>
[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 1)]]
void k_tmpl(T *) {}

// has_property: name present / absent.
static_assert(__builtin_sycl_has_property(k_nd, "sycl-nd-range-kernel"));
static_assert(!__builtin_sycl_has_property(k_nd, "sycl-single-task-kernel"));
static_assert(__builtin_sycl_has_property(k_st, "sycl-single-task-kernel"));
static_assert(!__builtin_sycl_has_property(k_st, "sycl-nd-range-kernel"));
static_assert(!__builtin_sycl_has_property(plain, "sycl-nd-range-kernel"));
static_assert(!__builtin_sycl_has_property(plain, "sycl-single-task-kernel"));

// has_property accepts the address-of and dereference operand forms too.
static_assert(__builtin_sycl_has_property(&k_nd, "sycl-nd-range-kernel"));
static_assert(__builtin_sycl_has_property(*k_nd, "sycl-nd-range-kernel"));

// get_property: the kind/dim value as an int.
static_assert(__builtin_sycl_get_property(k_nd, "sycl-nd-range-kernel") == 2);
static_assert(__builtin_sycl_get_property(k_st, "sycl-single-task-kernel") == 0);
// Absent property -> 0.
static_assert(__builtin_sycl_get_property(k_nd, "sycl-single-task-kernel") == 0);
static_assert(__builtin_sycl_get_property(plain, "sycl-nd-range-kernel") == 0);

// A multi-property decoration: each name is independently queryable; for a
// multi-int value like "8,8,8" get_property returns the FIRST int (documented
// prototype limitation -- comma-lists are future work).
static_assert(__builtin_sycl_has_property(k_multi, "sycl-nd-range-kernel"));
static_assert(__builtin_sycl_has_property(k_multi, "sycl-work-group-size"));
static_assert(__builtin_sycl_get_property(k_multi, "sycl-nd-range-kernel") == 3);
static_assert(__builtin_sycl_get_property(k_multi, "sycl-work-group-size") == 8);

// Template instantiation: the attribute is copied to the specialization with
// the property values substituted, so &k_tmpl<int> reads them off the
// instantiated decl. The cast disambiguates the function-template-id.
static_assert(__builtin_sycl_has_property((void (*)(int *))k_tmpl<int>,
                                          "sycl-nd-range-kernel"));
static_assert(__builtin_sycl_get_property((void (*)(int *))k_tmpl<int>,
                                          "sycl-nd-range-kernel") == 1);

// The builtins are usable in a constexpr context.
constexpr bool HasNd = __builtin_sycl_has_property(k_nd, "sycl-nd-range-kernel");
static_assert(HasNd);
constexpr int Dim = __builtin_sycl_get_property(k_nd, "sycl-nd-range-kernel");
static_assert(Dim == 2);
