// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

// An address-space conversion on the base of an array subscript (as in
// sycl::multi_ptr::operator[]) must cast the pointer, not just reinterpret
// the element type; a local offset is not a valid generic address.

struct P {
  __attribute__((opencl_local)) float *m;
  float &at(long i) { return reinterpret_cast<float *>(m)[i]; }
};

SYCL_EXTERNAL float f(P p) { return p.at(3); }

// CIR-LABEL: cir.func{{.*}} @_ZN1P2atEl(
// CIR:         %[[M:.*]] = cir.load{{.*}} : !cir.ptr<!cir.ptr<!cir.float, target_address_space(3)>, target_address_space(4)>, !cir.ptr<!cir.float, target_address_space(3)>
// CIR:         %[[G:.*]] = cir.cast address_space %[[M]] : !cir.ptr<!cir.float, target_address_space(3)> -> !cir.ptr<!cir.float, target_address_space(4)>
// CIR:         cir.ptr_stride %[[G]], %{{.*}} : (!cir.ptr<!cir.float, target_address_space(4)>, !s64i) -> !cir.ptr<!cir.float, target_address_space(4)>
// CIR-NOT:     cir.cast bitcast
// CIR:         cir.return

// LLVM-LABEL: define {{.*}}@_ZN1P2atEl(
// LLVM:         %[[M:.*]] = load ptr addrspace(3), ptr addrspace(4) %{{.*}}
// LLVM:         %[[G:.*]] = addrspacecast ptr addrspace(3) %[[M]] to ptr addrspace(4)
// LLVM:         getelementptr {{.*}}float, ptr addrspace(4) %[[G]]
