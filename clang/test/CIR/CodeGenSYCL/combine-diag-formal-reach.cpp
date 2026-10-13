// The formal-reach guard of the host diagnostics (SYCLHostFacts.cpp: rule
// (d) of hostOnlyMemoryOfClosureMember, rule (b) of deviceUSMOfPointer). In
// BU mode a function's own graph unifies its formals only with the actuals
// of the callers bound to it context-insensitively (here: the calls of the
// recursive helpers); the origins of what a formal reaches there are not
// every object a caller may pass. A helper that takes a pointer parameter
// and one caller passes host memory, another USM, must give no warning:
//  - sycl-host-ptr-in-kernel: launchRec launches a kernel with its pointer
//    parameter; main passes a stack array, viaShared a USM allocation;
//  - sycl-host-deref-device-usm: touchRec stores through its parameter;
//    main passes a stack array, viaDevice a device allocation.
// Both are rejected by the guard (the -debug-only verdicts below), in open
// and closed-tu mode.
//
// REQUIRES: asserts
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fclangir -I%S/Inputs -emit-cir %s -o %t/dev.cir
// RUN: %clang_cc1 -std=c++17 -fsycl-is-host -triple x86_64-unknown-linux-gnu \
// RUN:   -aux-triple spir64-unknown-unknown -fclangir -I%S/Inputs -emit-cir %s -o %t/host.cir
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir --cir-offload-host-diagnostics=all,sycl-host-heap-ptr-in-kernel \
// RUN:   -debug-only=cir-offload-host-diagnostics -output=%t/open.cir 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=warning:
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir --cir-offload-host-diagnostics=all,sycl-host-heap-ptr-in-kernel \
// RUN:   -cir-offload-assume-closed-tu -debug-only=cir-offload-host-diagnostics -output=%t/closed.cir 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=warning:

// CHECK-DAG: host-ptr-in-kernel: @_ZTS4KRec{{.*}} arg 0 'p' at {{.*}}: n{{[0-9]+}} is reachable from a formal of a function's own graph
// CHECK-DAG: host-deref-device-usm: @_ZL8touchRecPfi cir.store at {{.*}}: n{{[0-9]+}} is reachable from a formal of a function's own graph

#include "combine-canary-mock.h"
using namespace sycl;
class KRec;
static void launchRec(queue &q, float *p, int n) {
  if (n > 0)
    launchRec(q, p, n - 1);
  q.submit([&](handler &cgh) {
    cgh.parallel_for<KRec>(range<1>(64), [=](id<1> i) { p[i[0]] = 1.f; });
  });
}
static void touchRec(float *p, int n) {
  if (n > 0)
    touchRec(p, n - 1);
  p[0] = 1.f;
}
static void viaDevice(queue &q) {
  float *d = malloc_device<float>(64, q);
  touchRec(d, 2);
}
static void viaShared(queue &q) {
  float *s = malloc_shared<float>(64, q);
  launchRec(q, s, 2);
}
int main() {
  queue q;
  float host[64];
  launchRec(q, host, 2);
  viaShared(q);
  touchRec(host, 2);
  viaDevice(q);
  return 0;
}
