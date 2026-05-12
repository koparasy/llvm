// Regression test: when SYCL headers come from a Clang module (PCM), the
// sycl::aspect enum is lazily deserialized and CodeGenTypes does not observe
// it unless user code references it. Previously that left
// CodeGenModule::AspectsEnumDecl null, so sycl_aspects named metadata was
// never emitted and SYCLPropagateAspectsUsagePass early-exited without
// attaching !sycl_fixed_targets to kernels, breaking runtime kernel binding.
//
// CodeGenModule::Release now eagerly walks the sycl namespace under
// -fsycl-is-device to register the aspect enum regardless of whether user
// code references it.

// RUN: rm -rf %t
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:     -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:     -I %S/Inputs/module-aspect \
// RUN:     -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

#include "sycl_aspect.hpp"

// The TU never mentions sycl::aspect. Without the eager lookup fix the
// metadata would be missing.

// CHECK: !sycl_aspects = !{![[HOST:[0-9]+]], ![[CPU:[0-9]+]], ![[GPU:[0-9]+]], ![[ACC:[0-9]+]], ![[CUSTOM:[0-9]+]], ![[FP16:[0-9]+]], ![[FP64:[0-9]+]]}
// CHECK-DAG: ![[HOST]] = !{!"host", i32 0}
// CHECK-DAG: ![[CPU]] = !{!"cpu", i32 1}
// CHECK-DAG: ![[GPU]] = !{!"gpu", i32 2}
// CHECK-DAG: ![[ACC]] = !{!"accelerator", i32 3}
// CHECK-DAG: ![[CUSTOM]] = !{!"custom", i32 4}
// CHECK-DAG: ![[FP16]] = !{!"fp16", i32 5}
// CHECK-DAG: ![[FP64]] = !{!"fp64", i32 6}

void unused_stub() {}
