// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fno-gpu-rdc -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fno-gpu-rdc -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fno-gpu-rdc -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// A free-function kernel registered via __registered_kernels__ must emit the
// sycl_registered_kernels named metadata mapping the user-facing name to the
// mangled __sycl_kernel_ entry point, so the runtime can resolve it by name.

[[__sycl_detail__::add_ir_attributes_function("sycl-nd-range-kernel", 0)]]
void ff(int *p) { *p = 42; }

namespace N {
[[__sycl_detail__::__registered_kernels__({"my_kernel", ff})]];
}

// CIR: cir.sycl_registered_kernels = {{\[}}["my_kernel", "_Z16__sycl_kernel_ffPi"]]

// LLVM: !sycl_registered_kernels = !{![[LIST:[0-9]+]]}
// LLVM: ![[LIST]] = !{![[ENT:[0-9]+]]}
// LLVM: ![[ENT]] = !{!"my_kernel", !"_Z16__sycl_kernel_ffPi"}

// OGCG: !sycl_registered_kernels = !{![[LIST:[0-9]+]]}
// OGCG: ![[LIST]] = !{![[ENT:[0-9]+]]}
// OGCG: ![[ENT]] = !{!"my_kernel", !"_Z16__sycl_kernel_ffPi"}
