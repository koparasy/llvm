// RUN: %clangxx -fsycl -fsycl-device-only -emit-llvm -S %s -o %t.ll
// RUN: FileCheck %s < %t.ll
//
// This test checks that the KHR SYCL_KHR_KERNEL(...) decoration macro lowers
// each free function kernel's properties to the expected
// add_ir_attributes_function IR attributes. It is the IR-level proof that the
// variadic macro preserves every property -- including properties whose
// template-argument list contains commas (e.g. work_group_size<8, 8, 8>) -- and
// does not silently drop or truncate them. A broken macro that split nested
// commas would still compile, so checking the emitted IR is mandatory.

#include <sycl/khr/free_kernel.hpp>
#include <sycl/sycl.hpp>

namespace khr = sycl::khr;
namespace syclexp = sycl::ext::oneapi::experimental;

// Kind only, no modifiers: the mandatory kind is the sole argument. Verifies the
// zero-modifier (leading-fixed-arg, trailing-comma) case compiles in C++17.
SYCL_KHR_KERNEL(khr::nd_kernel<1>)
void k1(int *) {}

// 1 property, single_task kind.
SYCL_KHR_KERNEL(khr::single_task_kernel)
void k1b(int *) {}

// 2 properties.
SYCL_KHR_KERNEL(khr::nd_kernel<1>, syclexp::work_group_size<32>)
void k2(int *) {}

// 2 properties where the 2nd has commas inside <...> (the critical case).
SYCL_KHR_KERNEL(khr::nd_kernel<2>, syclexp::work_group_size<8, 8>)
void k3(int *) {}

// 3 properties, the multi-arg one in the middle (commas not at the tail).
SYCL_KHR_KERNEL(khr::nd_kernel<3>, syclexp::work_group_size<8, 8, 8>,
                syclexp::sub_group_size<16>)
void k4(int *) {}

// Many properties, multiple multi-comma template args.
SYCL_KHR_KERNEL(khr::nd_kernel<3>, syclexp::work_group_size<4, 4, 4>,
                syclexp::work_group_size_hint<2, 2, 2>,
                syclexp::sub_group_size<8>)
void k5(int *) {}

// Each kernel is emitted as a device function carrying its own attribute group.
// We bind each definition to its attribute-group number, then assert the
// contents of that group below.

// Each free function kernel (every SYCL_KHR_KERNEL carries a mandatory kind) is
// emitted as a device kernel wrapper named __sycl_kernel_<fn>.
// CHECK: define {{.*}} @{{.*}}__sycl_kernel_k1{{.*}} #[[#K1:]]
// CHECK: define {{.*}} @{{.*}}__sycl_kernel_k1b{{.*}} #[[#K1B:]]
// CHECK: define {{.*}} @{{.*}}__sycl_kernel_k2{{.*}} #[[#K2:]]
// CHECK: define {{.*}} @{{.*}}__sycl_kernel_k3{{.*}} #[[#K3:]]
// CHECK: define {{.*}} @{{.*}}__sycl_kernel_k4{{.*}} #[[#K4:]]
// CHECK: define {{.*}} @{{.*}}__sycl_kernel_k5{{.*}} #[[#K5:]]

// k1: single nd-range property, dimension 1.
// CHECK-DAG: attributes #[[#K1]] = {{{.*}}"sycl-nd-range-kernel"="1"{{.*}}}

// k1b: single_task kind lowers to "sycl-single-task-kernel"="0".
// CHECK-DAG: attributes #[[#K1B]] = {{{.*}}"sycl-single-task-kernel"="0"{{.*}}}

// k2: nd-range + scalar work-group-size.
// CHECK-DAG: attributes #[[#K2]] = {{{.*}}"sycl-nd-range-kernel"="1"{{.*}}"sycl-work-group-size"="32"{{.*}}}

// k3 (critical comma-in-<> case): the full "8,8" value must survive, not "8".
// CHECK-DAG: attributes #[[#K3]] = {{{.*}}"sycl-nd-range-kernel"="2"{{.*}}"sycl-work-group-size"="8,8"{{.*}}}

// k4: multi-comma value in the middle of the list. Assert all three properties
// with their full values.
// CHECK-DAG: attributes #[[#K4]] = {{{.*}}"sycl-nd-range-kernel"="3"{{.*}}"sycl-sub-group-size"="16"{{.*}}"sycl-work-group-size"="8,8,8"{{.*}}}

// k5: multiple multi-comma values. All four properties must land with their
// full comma-separated values.
// CHECK-DAG: attributes #[[#K5]] = {{{.*}}"sycl-nd-range-kernel"="3"{{.*}}"sycl-sub-group-size"="8"{{.*}}"sycl-work-group-size"="4,4,4"{{.*}}"sycl-work-group-size-hint"="2,2,2"{{.*}}}
