// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fclangir -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -emit-llvm %s -o - | FileCheck %s

// In SYCL device code only vtables of classes with 'indirectly_callable'
// virtual functions are emitted. A host-only polymorphic class whose key
// function is defined here must not drag its vtable, RTTI and virtual
// functions into the device module.

struct HostOnly {
  virtual ~HostOnly();
  int x;
};
HostOnly::~HostOnly() {}

// CHECK-NOT: @_ZTV8HostOnly
// CHECK-NOT: @_ZTI8HostOnly
// CHECK-NOT: define {{.*}}@_ZN8HostOnlyD
