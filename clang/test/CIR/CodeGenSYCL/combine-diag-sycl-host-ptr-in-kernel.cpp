// End to end: the offload combine warns when a kernel dereferences a captured
// pointer that the host facts prove points to host memory (a stack variable,
// a global), which the device cannot access (SYCL 2020 sec:usm):
// cir-offload-host-diagnostics, check sycl-host-ptr-in-kernel; a host heap
// allocation is the opt-in check sycl-host-heap-ptr-in-kernel. The warning is at the kernel's access, with a note
// at the allocation. The SYCL and C++ library parts are the reduced headers
// of the combine-canary tests.
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fclangir -I%S/Inputs -emit-cir %s -o %t/dev.cir
// RUN: %clang_cc1 -std=c++17 -fsycl-is-host -triple x86_64-unknown-linux-gnu \
// RUN:   -aux-triple spir64-unknown-unknown -fclangir -I%S/Inputs -emit-cir %s -o %t/host.cir
//
// The default (open, internalized launch paths) and closed-tu give the same
// warnings.
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir -output=%t/open.cir 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not="kernel argument"
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir -cir-offload-assume-closed-tu -output=%t/closed.cir 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not="kernel argument"
//
// The heap check is off by default (heap memory is valid on a device with
// aspect::usm_system_allocations): only when named, alone or with all.
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir --cir-offload-host-diagnostics=sycl-host-heap-ptr-in-kernel \
// RUN:   -output=%t/heap.cir 2>&1 | FileCheck %s --check-prefix=HEAP --implicit-check-not="kernel argument"
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir --cir-offload-host-diagnostics=all,sycl-host-heap-ptr-in-kernel \
// RUN:   -output=%t/both.cir 2>&1 | FileCheck %s --check-prefixes=CHECK,HEAP --implicit-check-not="kernel argument"
//
// Off: no warning, and the same output (the pass changes nothing).
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir --cir-offload-host-diagnostics=none -output=%t/none.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NONE --allow-empty
// RUN: diff %t/open.cir %t/none.cir
// RUN: not cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir --cir-offload-host-diagnostics=bogus -output=%t/bad.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD

// Each warning has a note at the kernel (the lambda or functor) that captures
// the pointer: 5 warnings by default.
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir -output=%t/open2.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOTE
// NOTE-COUNT-5: note: the kernel that captures the pointer is defined here
// NOTE-NOT: note: the kernel that captures
// NONE-NOT: kernel argument
// BAD: unknown host diagnostic 'bogus'

#include "combine-canary-mock.h"

using namespace sycl;

extern "C" void *malloc(unsigned long);
float *opaque();

class KStack;
class KGlobal;
class KVector;
class KMalloc;
class KStruct;
class KFunctor;
class KHelper;
class KUsm;
class KAccessor;
class KUnknown;
class KParam;
class KNull;
class KNoDeref;
class KMixed;

// CHECK-DAG: "{{.*}}combine-diag-sycl-host-ptr-in-kernel.cpp":[[#@LINE+1]]:{{.*}}note: the global 'gtable' the pointer points to is defined here
float gtable[64];

struct Params {
  float *data;
  int n;
};

struct Functor {
  float *p;
  // CHECK-DAG: "{{.*}}combine-diag-sycl-host-ptr-in-kernel.cpp":[[#@LINE+1]]:{{.*}}warning: kernel argument 'p' points to host memory (stack variable) that the device cannot access [sycl-host-ptr-in-kernel]
  void operator()(id<1> i) const { p[i[0]] = 3.0f; }
};

// Launches with whatever the caller passes: the facts are per calling
// context, and must hold in every one.
template <class K> void launch(queue &q, float *p) {
  q.submit([&](handler &h) {
    // CHECK-DAG: "{{.*}}combine-diag-sycl-host-ptr-in-kernel.cpp":[[#@LINE+1]]:{{.*}}warning: kernel argument 'p' points to host memory (stack variable) that the device cannot access [sycl-host-ptr-in-kernel]
    h.parallel_for<K>(range<1>(64), [=](id<1> i) { p[i[0]] = 0.0f; });
  });
}

// Externally visible and not called here: its parameter is whatever another
// TU passes (open), or the USM pointer main passes (closed-tu).
void ext_entry(queue &q, float *p) {
  q.submit([&](handler &h) {
    h.parallel_for<KParam>(range<1>(64), [=](id<1> i) { p[i[0]] = 1.0f; });
  });
}

int main(int argc, char **) {
  queue q;
  // CHECK-DAG: "{{.*}}combine-diag-sycl-host-ptr-in-kernel.cpp":[[#@LINE+1]]:{{.*}}note: the stack variable 'local' the pointer points to is allocated here
  float local[64];
  float *sp = local;
  float *gp = gtable;
  std::vector<float> v;
  v.resize(64);
  float *vp = v.data();
  // HEAP-DAG: "{{.*}}combine-diag-sycl-host-ptr-in-kernel.cpp":[[#@LINE+1]]:{{.*}}note: the heap memory the pointer points to is allocated here
  float *mp = (float *)malloc(256);
  float *usm = malloc_shared<float>(64, q);
  float *up = opaque();
  float *np = argc > 1 ? local : nullptr;
  Params s{local, 64};
  buffer<float, 1> b(usm, range<1>(64));

  // Positive: a stack array, a global, a std::vector's heap storage and
  // malloc (the heap check), a pointer member of a captured struct, a named
  // functor's member, and a helper every caller of which passes the stack
  // array.
  q.submit([&](handler &h) {
    // CHECK-DAG: "{{.*}}combine-diag-sycl-host-ptr-in-kernel.cpp":[[#@LINE+1]]:{{.*}}warning: kernel argument 'sp' points to host memory (stack variable) that the device cannot access [sycl-host-ptr-in-kernel]
    h.parallel_for<KStack>(range<1>(64), [=](id<1> i) { sp[i[0]] = 1.0f; });
  });
  q.submit([&](handler &h) {
    // CHECK-DAG: "{{.*}}combine-diag-sycl-host-ptr-in-kernel.cpp":[[#@LINE+1]]:{{.*}}warning: kernel argument 'gp' points to host memory (global) that the device cannot access [sycl-host-ptr-in-kernel]
    h.parallel_for<KGlobal>(range<1>(64), [=](id<1> i) { gp[i[0]] += 1.0f; });
  });
  q.submit([&](handler &h) {
    // HEAP-DAG: "{{.*}}combine-diag-sycl-host-ptr-in-kernel.cpp":[[#@LINE+1]]:{{.*}}warning: kernel argument 'vp' points to host heap memory (heap allocation) that the device can access only if it has aspect::usm_system_allocations (SYCL 2020 sec:usm) [sycl-host-heap-ptr-in-kernel]
    h.parallel_for<KVector>(range<1>(64), [=](id<1> i) { vp[i[0]] = 1.0f; });
  });
  q.submit([&](handler &h) {
    // HEAP-DAG: "{{.*}}combine-diag-sycl-host-ptr-in-kernel.cpp":[[#@LINE+1]]:{{.*}}warning: kernel argument 'mp' points to host heap memory (heap allocation) that the device can access only if it has aspect::usm_system_allocations (SYCL 2020 sec:usm) [sycl-host-heap-ptr-in-kernel]
    h.parallel_for<KMalloc>(range<1>(64), [=](id<1> i) { mp[i[0]] = 1.0f; });
  });
  q.submit([&](handler &h) {
    // CHECK-DAG: "{{.*}}combine-diag-sycl-host-ptr-in-kernel.cpp":[[#@LINE+1]]:{{.*}}warning: kernel argument 's.data' points to host memory (stack variable) that the device cannot access [sycl-host-ptr-in-kernel]
    h.parallel_for<KStruct>(range<1>(64), [=](id<1> i) { s.data[i[0]] = 1.0f; });
  });
  q.submit([&](handler &h) {
    h.parallel_for<KFunctor>(range<1>(64), Functor{local});
  });
  launch<KHelper>(q, local);

  // Negative: a USM pointer, a buffer accessor, a pointer an unknown call
  // returns, a parameter of an externally visible function, a pointer that
  // may be null, a kernel that does not dereference the pointer, and a
  // helper one caller of which passes USM memory.
  q.submit([&](handler &h) {
    h.parallel_for<KUsm>(range<1>(64), [=](id<1> i) { usm[i[0]] = 1.0f; });
  });
  q.submit([&](handler &h) {
    auto a = b.get_access<access::mode::read_write>(h);
    h.parallel_for<KAccessor>(range<1>(64), [=](id<1> i) { a[i] = 1.0f; });
  });
  q.submit([&](handler &h) {
    h.parallel_for<KUnknown>(range<1>(64), [=](id<1> i) { up[i[0]] = 1.0f; });
  });
  ext_entry(q, usm);
  q.submit([&](handler &h) {
    h.parallel_for<KNull>(range<1>(64), [=](id<1> i) {
      if (np)
        np[i[0]] = 1.0f;
    });
  });
  q.submit([&](handler &h) {
    h.parallel_for<KNoDeref>(range<1>(64), [=](id<1> i) {
      if (sp == nullptr)
        usm[i[0]] = 1.0f;
    });
  });
  launch<KMixed>(q, local);
  launch<KMixed>(q, usm);
  q.wait();
  return 0;
}
