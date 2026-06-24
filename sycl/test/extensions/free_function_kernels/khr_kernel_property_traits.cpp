// RUN: %clangxx -fsycl -fsyntax-only %s
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only %s
//
// This test exercises the host-evaluable, compile-time kernel-property query
// traits surfaced in <sycl/khr/free_kernel.hpp> (Step 4):
//   sycl::khr::is_kernel_v<Func>
//   sycl::khr::is_nd_kernel_v<Func, Dims>
//   sycl::khr::is_single_task_kernel_v<Func>
//
// These traits read a free function kernel's properties straight off the
// decorated FunctionDecl via the __builtin_sycl_has_property /
// __builtin_sycl_get_property builtins, so they evaluate at host parse time
// WITHOUT the integration header. The FIRST RUN line is the load-bearing one:
// it is a plain -fsycl HOST compile and every static_assert below holds, which
// is the proof that kernel properties are declaration-local and host-queryable.
// The second RUN line confirms the same queries also hold in the device pass.

#include <sycl/khr/free_kernel.hpp>
#include <sycl/sycl.hpp>

namespace khr = sycl::khr;

SYCL_KHR_KERNEL(khr::nd_kernel<2>)
void k_nd(int *) {}

SYCL_KHR_KERNEL(khr::single_task_kernel)
void k_st(int *) {}

template <typename T>
SYCL_KHR_KERNEL(khr::nd_kernel<1>)
void k_tmpl(T *) {}

void plain(int *) {}

// is_kernel_v: a decorated free function kernel of either kind.
static_assert(khr::is_kernel_v<k_nd>);
static_assert(khr::is_kernel_v<k_st>);
static_assert(!khr::is_kernel_v<plain>);

// is_nd_kernel_v<Func, Dims>: nd-range kernel with matching dimensionality.
static_assert(khr::is_nd_kernel_v<k_nd, 2>);
static_assert(!khr::is_nd_kernel_v<k_nd, 3>); // wrong dim
static_assert(!khr::is_nd_kernel_v<k_st, 1>); // single_task is not nd
static_assert(!khr::is_nd_kernel_v<plain, 2>);

// is_single_task_kernel_v.
static_assert(khr::is_single_task_kernel_v<k_st>);
static_assert(!khr::is_single_task_kernel_v<k_nd>);
static_assert(!khr::is_single_task_kernel_v<plain>);

// Template instantiation: the property values are read off the specialization,
// which carries the substituted decoration.
static_assert(khr::is_kernel_v<k_tmpl<float>>);
static_assert(khr::is_nd_kernel_v<k_tmpl<float>, 1>);
static_assert(!khr::is_nd_kernel_v<k_tmpl<float>, 2>);
