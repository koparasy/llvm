// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

// The per-element pointer of an array constructor/destructor loop lives in
// the address space of the array. A member array reached through a generic
// `this` must not be iterated through a default (private) pointer.

struct I {
  double a, b;
  I() : a(0), b(0) {}
  ~I();
};

struct S {
  I buf[4];
  int tos;
  S() : tos(-1) {}
};

SYCL_EXTERNAL int f() {
  S s;
  return s.tos;
}

// CIR-LABEL: cir.func{{.*}} @_ZN1SC2Ev(
// CIR:         cir.cast array_to_ptrdecay %{{.*}} : !cir.ptr<!cir.array<!rec_I x 4>, target_address_space(4)> -> !cir.ptr<!rec_I, target_address_space(4)>
// CIR:         cir.call @_ZN1IC1Ev(%{{.*}}) {{.*}}: (!cir.ptr<!rec_I, target_address_space(4)>

// CIR-LABEL: cir.func{{.*}} @_ZN1SD2Ev(
// CIR:         cir.cast array_to_ptrdecay %{{.*}} : !cir.ptr<!cir.array<!rec_I x 4>, target_address_space(4)> -> !cir.ptr<!rec_I, target_address_space(4)>
// CIR:         cir.call @_ZN1ID1Ev(%{{.*}}) {{.*}}: (!cir.ptr<!rec_I, target_address_space(4)>

// LLVM-LABEL: define {{.*}}@_ZN1SC2Ev(
// LLVM:         %[[BEGIN:.*]] = getelementptr %struct.I, ptr addrspace(4) %{{.*}}, i32 0
// LLVM:         getelementptr %struct.I, ptr addrspace(4) %[[BEGIN]], i64 4
// LLVM:         call spir_func void @_ZN1IC1Ev(ptr addrspace(4)
