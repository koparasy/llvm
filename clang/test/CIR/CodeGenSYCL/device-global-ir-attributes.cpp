// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fclangir -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -emit-llvm %s -o - | FileCheck %s

// A SYCL device_global carries the string IR attributes the runtime uses to
// find and initialize it: the type's add_ir_attributes_global_variable pairs
// and "sycl-unique-id". Without them the runtime never fills in the global's
// USM pointer and the kernel reads address 0.

template <typename T>
struct [[__sycl_detail__::global_variable_allowed,
         __sycl_detail__::device_global,
         __sycl_detail__::add_ir_attributes_global_variable(
             "sycl-device-global-size", sizeof(T))]] DeviceGlobal {
  T *usmptr;
  T val;
};

DeviceGlobal<double> DG;

__attribute__((sycl_device)) double read() { return DG.val; }

// CHECK: @DG = {{.*}}addrspace(1) global {{.*}} #[[ATTRS:[0-9]+]]
// CHECK: attributes #[[ATTRS]] = { "sycl-device-global-size"="8" "sycl-unique-id"="_Z2DG" }
