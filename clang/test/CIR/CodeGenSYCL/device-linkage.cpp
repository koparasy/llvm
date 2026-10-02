// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fgpu-rdc -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=RDC
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fgpu-rdc -emit-llvm %s -o - | FileCheck %s --check-prefix=RDC
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fno-gpu-rdc -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=NORDC
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fno-gpu-rdc -emit-llvm %s -o - | FileCheck %s --check-prefix=NORDC

// SYCL device code that is not SYCL_EXTERNAL (or a kernel / sycl_device /
// device_global) is linkonce_odr under RDC and internal without it, as in
// classic CodeGen (SYCL defaults to RDC). A non-inline explicit specialization in a header (like the
// SYCL __spirv::getBuiltIn* helpers) must not become a strong symbol that
// collides across translation units.

template <int ID> unsigned long getId();
template <> unsigned long getId<0>() { return 7; }

// A SYCL_EXTERNAL function keeps external linkage.
// RDC: define dso_local spir_func {{.*}}@_Z3extv(
// NORDC: define dso_local spir_func {{.*}}@_Z3extv(
[[clang::sycl_external]] unsigned long ext() { return getId<0>(); }

// The specialization, emitted after its user.
// RDC: define linkonce_odr {{.*}}@_Z5getIdILi0EEmv(
// NORDC: define internal {{.*}}@_Z5getIdILi0EEmv(
