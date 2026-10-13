// End to end: the offload combine warns when a sycl::buffer is constructed
// over host memory that must overlap the host memory of another buffer
// constructed before it in the same straight-line code and still alive,
// which is undefined behavior (SYCL 2020 sec:buffer PI 5358-5359,
// sec:buf-sync-rules PI 6010-6014): cir-offload-host-diagnostics, check
// sycl-buffer-same-host-ptr. The warning is at the second construction, with
// a note at the first and, when the points-to graph knows the one object, at
// the allocation. The SYCL and C++ library parts are the reduced headers of
// the combine-canary tests.
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fclangir -I%S/Inputs -emit-cir %s -o %t/dev.cir
// RUN: %clang_cc1 -std=c++17 -fsycl-is-host -triple x86_64-unknown-linux-gnu \
// RUN:   -aux-triple spir64-unknown-unknown -fclangir -I%S/Inputs -emit-cir %s -o %t/host.cir
//
// Open mode (the default) and closed-tu give the same warnings. (The note at
// a heap allocation needs the points-to object, which needs the allocating
// function to run once: closed-tu only, see the .cir test.)
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir -output=%t/open.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK --implicit-check-not=warning:
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir -cir-offload-assume-closed-tu -output=%t/closed.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK --implicit-check-not=warning:
//
// Off, or only the other checks: no warning, and the same output (the pass
// changes nothing).
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir --cir-offload-host-diagnostics=none -output=%t/none.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NONE --allow-empty
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir \
// RUN:   --cir-offload-host-diagnostics=sycl-host-ptr-in-kernel,sycl-host-deref-device-usm \
// RUN:   -output=%t/other.cir 2>&1 | FileCheck %s --check-prefix=NONE --allow-empty
// RUN: diff %t/open.cir %t/none.cir
//
// Every warning has a note at the first construction.
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir -output=%t/open2.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOTE
// NOTE-COUNT-5: note: the buffer that owns the host memory is constructed here and is still alive; constructing another buffer over its host memory during its lifetime is undefined behavior (SYCL 2020 sec:buf-sync-rules)
// NOTE-NOT: note: the buffer that owns
// NONE-NOT: warning:

#include "combine-canary-mock.h"

using namespace sycl;

extern "C" void *malloc(unsigned long);
float *opaque();
bool cond();
void consume(buffer<float, 1> &b);

// CHECK-DAG: combine-diag-sycl-buffer-same-host-ptr.cpp":[[#@LINE+1]]:{{.*}}note: the host memory is allocated here
static float gArr[64];

class KUse;

// One pointer, two buffers: the second construction warns.
void sameVar(queue &q) {
  float *h = (float *)malloc(64 * sizeof(float));
  buffer<float, 1> a(h, range<1>(64));
  // CHECK-DAG: combine-diag-sycl-buffer-same-host-ptr.cpp":[[#@LINE+1]]:{{.*}}warning: buffer constructed over host memory already owned by the buffer at combine-diag-sycl-buffer-same-host-ptr.cpp:[[#@LINE-1]] [sycl-buffer-same-host-ptr]
  buffer<float, 1> b(h, range<1>(64));
  q.submit([&](handler &cgh) {
    auto x = a.get_access<access::mode::read_write>(cgh);
    auto y = b.get_access<access::mode::read_write>(cgh);
    cgh.parallel_for<KUse>(range<1>(64), [=](id<1> i) { x[i] = y[i]; });
  });
}

// A stack array, and an overlapping part of it.
void stackOverlap() {
  float arr[64];
  buffer<float, 1> a(arr, range<1>(64));
  // CHECK-DAG: combine-diag-sycl-buffer-same-host-ptr.cpp":[[#@LINE+1]]:{{.*}}warning: buffer constructed over host memory already owned by the buffer at combine-diag-sycl-buffer-same-host-ptr.cpp:[[#@LINE-1]] [sycl-buffer-same-host-ptr]
  buffer<float, 1> b(arr + 32, range<1>(16));
}

// A stack array, disjoint halves: silent.
void stackDisjoint() {
  float arr[64];
  buffer<float, 1> a(arr, range<1>(32));
  buffer<float, 1> b(arr + 32, range<1>(32));
}

// A global array.
void global() {
  buffer<float, 1> a(gArr, range<1>(64));
  // CHECK-DAG: combine-diag-sycl-buffer-same-host-ptr.cpp":[[#@LINE+1]]:{{.*}}warning: buffer constructed over host memory already owned by the buffer at combine-diag-sycl-buffer-same-host-ptr.cpp:[[#@LINE-1]] [sycl-buffer-same-host-ptr]
  buffer<float, 1> b(gArr, range<1>(8));
}

// A copy of the pointer in another variable: each load is followed to the
// one store of its variable.
void copiedPtr() {
  float *h = (float *)malloc(64 * sizeof(float));
  float *h2 = h;
  buffer<float, 1> a(h, range<1>(64));
  // CHECK-DAG: combine-diag-sycl-buffer-same-host-ptr.cpp":[[#@LINE+1]]:{{.*}}warning: buffer constructed over host memory already owned by the buffer at combine-diag-sycl-buffer-same-host-ptr.cpp:[[#@LINE-1]] [sycl-buffer-same-host-ptr]
  buffer<float, 1> b(h2, range<1>(64));
}

// vector::data() twice: two calls, and the vector's storage is an array
// node of the points-to graph (its offsets are not exact): missed.
void vectorData() {
  std::vector<float> v;
  v.resize(64);
  buffer<float, 1> a(v.data(), range<1>(64));
  buffer<float, 1> b(v.data(), range<1>(64));
}

// The parameter of an externally visible function, used twice.
void paramTwice(float *p) {
  buffer<float, 1> a(p, range<1>(4));
  // CHECK-DAG: combine-diag-sycl-buffer-same-host-ptr.cpp":[[#@LINE+1]]:{{.*}}warning: buffer constructed over host memory already owned by the buffer at combine-diag-sycl-buffer-same-host-ptr.cpp:[[#@LINE-1]] [sycl-buffer-same-host-ptr]
  buffer<float, 1> b(p, range<1>(4));
}

// Two parameters of an externally visible function: silent.
void twoParams(float *p, float *r) {
  buffer<float, 1> a(p, range<1>(4));
  buffer<float, 1> b(r, range<1>(4));
}

// Sequential scopes: the first buffer is gone: silent.
void sequential() {
  float *h = (float *)malloc(64 * sizeof(float));
  { buffer<float, 1> a(h, range<1>(64)); }
  { buffer<float, 1> b(h, range<1>(64)); }
}

// The pointer is reassigned in between: silent.
void reassigned() {
  float *h = (float *)malloc(64 * sizeof(float));
  buffer<float, 1> a(h, range<1>(64));
  h = (float *)malloc(64 * sizeof(float));
  buffer<float, 1> b(h, range<1>(64));
}

// Two unknown pointers (an unknown call): silent.
void unknown() {
  buffer<float, 1> a(opaque(), range<1>(64));
  buffer<float, 1> b(opaque(), range<1>(64));
}

// Two allocations: silent.
void twoAllocs() {
  float *h = (float *)malloc(64 * sizeof(float));
  float *g = (float *)malloc(64 * sizeof(float));
  buffer<float, 1> a(h, range<1>(64));
  buffer<float, 1> b(g, range<1>(64));
}

// Two USM allocations: silent.
void twoUSM(queue &q) {
  float *s = malloc_shared<float>(64, q);
  float *t = malloc_shared<float>(64, q);
  buffer<float, 1> a(s, range<1>(64));
  buffer<float, 1> b(t, range<1>(64));
}

// A sub-buffer and a copy of the first buffer: silent.
void subAndCopy() {
  float *h = (float *)malloc(64 * sizeof(float));
  buffer<float, 1> a(h, range<1>(64));
  buffer<float, 1> s(a, id<1>(8), range<1>(8));
  buffer<float, 1> c(a);
}

// The first buffer is given to a user function before the second: silent.
void consumed() {
  float *h = (float *)malloc(64 * sizeof(float));
  buffer<float, 1> a(h, range<1>(64));
  consume(a);
  buffer<float, 1> b(h, range<1>(64));
}

// The second construction is conditional (not straight-line): silent.
void conditional() {
  float *h = (float *)malloc(64 * sizeof(float));
  buffer<float, 1> a(h, range<1>(64));
  if (cond()) {
    buffer<float, 1> b(h, range<1>(64));
  }
}

// Empty buffers: silent.
void empty() {
  float arr[4];
  buffer<float, 1> a(arr, range<1>(0));
  buffer<float, 1> b(arr, range<1>(0));
}

// A pointer selected at run time: silent.
void selected() {
  float *h = (float *)malloc(64 * sizeof(float));
  float *g = (float *)malloc(64 * sizeof(float));
  float *p = cond() ? h : g;
  buffer<float, 1> a(h, range<1>(64));
  buffer<float, 1> b(p, range<1>(64));
}

int main() {
  queue q;
  sameVar(q);
  stackOverlap();
  stackDisjoint();
  global();
  copiedPtr();
  vectorData();
  sequential();
  reassigned();
  unknown();
  twoAllocs();
  twoUSM(q);
  subAndCopy();
  consumed();
  conditional();
  empty();
  selected();
}
