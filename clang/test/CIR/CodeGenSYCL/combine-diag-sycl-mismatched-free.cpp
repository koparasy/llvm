// End to end: the offload combine warns when a deallocation releases memory
// of another allocation family: free, operator delete or operator delete[]
// of a USM allocation, sycl::free of host memory (malloc, operator new, a
// stack object), and the generic C/C++ mismatches (SYCL 2020 sec:usm,
// sycl::free; C17 7.22.3.3; C++20 [expr.delete]):
// cir-offload-host-diagnostics, check sycl-mismatched-free. The warning is at
// the deallocation, with a note at the allocation. The SYCL and C++ library
// parts are the reduced headers of the combine-canary tests.
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fclangir -I%S/Inputs -emit-cir %s -o %t/dev.cir
// RUN: %clang_cc1 -std=c++17 -fsycl-is-host -triple x86_64-unknown-linux-gnu \
// RUN:   -aux-triple spir64-unknown-unknown -fclangir -I%S/Inputs -Wno-mismatched-new-delete \
// RUN:   -emit-cir %s -o %t/host.cir
//
// Open mode (the default) and closed-tu. A helper with an externally visible
// definition is judged only under closed-tu (open: other TUs may call it).
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir -output=%t/open.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK --implicit-check-not=warning:
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir -cir-offload-assume-closed-tu -output=%t/closed.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK,CLOSED --implicit-check-not=warning:
//
// Off, or only the other checks: no warning, and the same output (the pass
// changes nothing).
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir --cir-offload-host-diagnostics=none -output=%t/none.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NONE --allow-empty
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir \
// RUN:   --cir-offload-host-diagnostics=sycl-host-ptr-in-kernel,sycl-host-deref-device-usm,sycl-buffer-same-host-ptr,sycl-nd-range-geometry \
// RUN:   -output=%t/other.cir 2>&1 | FileCheck %s --check-prefix=NONE --allow-empty
// RUN: diff %t/open.cir %t/none.cir
//
// Every warning has a note at the allocation, with the rule.
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir -output=%t/open2.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOTE
// NOTE-COUNT-4: note: the memory is allocated here; USM memory must be released with sycl::free (SYCL 2020 sec:usm, sycl::free)
// NOTE-NOT: USM memory must be released
// NONE-NOT: warning:

#include "combine-canary-mock.h"

using namespace sycl;

extern "C" void *malloc(unsigned long);
extern "C" void free(void *);
float *opaque();
bool cond();

class KUse;

// free of a device allocation.
void freeDevice(queue &q) {
  float *d = malloc_device<float>(64, q);
  // CHECK-DAG: combine-diag-sycl-mismatched-free.cpp":[[#@LINE+1]]:{{.*}}warning: memory allocated by sycl::malloc_device at combine-diag-sycl-mismatched-free.cpp:[[#@LINE-1]] is released with free [sycl-mismatched-free]
  free(d);
}

// delete[] of a shared allocation, after a kernel used it.
void deleteShared(queue &q) {
  float *s = malloc_shared<float>(64, q);
  q.submit([&](handler &cgh) {
    cgh.parallel_for<KUse>(range<1>(64), [=](id<1> i) { s[i[0]] = 0; });
  });
  // CHECK-DAG: combine-diag-sycl-mismatched-free.cpp":[[#@LINE+1]]:{{.*}}warning: memory allocated by sycl::malloc_shared at combine-diag-sycl-mismatched-free.cpp:[[#@LINE-4]] is released with operator delete[] [sycl-mismatched-free]
  delete[] s;
}

// The pointer copied into another variable first.
void copied(queue &q) {
  float *h = malloc_host<float>(8, q);
  float *p = h;
  // CHECK-DAG: combine-diag-sycl-mismatched-free.cpp":[[#@LINE+1]]:{{.*}}warning: memory allocated by sycl::malloc_host at combine-diag-sycl-mismatched-free.cpp:[[#@LINE-2]] is released with free [sycl-mismatched-free]
  free(p);
}

// sycl::free of malloc, operator new[] and stack memory.
void syclFreeHost(queue &q) {
  float *h = (float *)malloc(64 * sizeof(float));
  // CHECK-DAG: combine-diag-sycl-mismatched-free.cpp":[[#@LINE+1]]:{{.*}}warning: memory allocated by malloc at combine-diag-sycl-mismatched-free.cpp:[[#@LINE-1]] is released with sycl::free [sycl-mismatched-free]
  sycl::free(h, q);
  float *n = new float[64];
  // CHECK-DAG: combine-diag-sycl-mismatched-free.cpp":[[#@LINE+1]]:{{.*}}warning: memory allocated by operator new[] at combine-diag-sycl-mismatched-free.cpp:[[#@LINE-1]] is released with sycl::free [sycl-mismatched-free]
  sycl::free(n, q);
  float a[4];
  // CHECK-DAG: combine-diag-sycl-mismatched-free.cpp":[[#@LINE+1]]:{{.*}}warning: memory of the stack variable 'a' at combine-diag-sycl-mismatched-free.cpp:[[#@LINE-1]] is released with sycl::free [sycl-mismatched-free]
  sycl::free(a, q);
}

// The generic C/C++ mismatches.
void generic() {
  int *m = (int *)malloc(4);
  // CHECK-DAG: combine-diag-sycl-mismatched-free.cpp":[[#@LINE+1]]:{{.*}}warning: memory allocated by malloc at combine-diag-sycl-mismatched-free.cpp:[[#@LINE-1]] is released with operator delete [sycl-mismatched-free]
  delete m;
  int *n = new int;
  // CHECK-DAG: combine-diag-sycl-mismatched-free.cpp":[[#@LINE+1]]:{{.*}}warning: memory allocated by operator new at combine-diag-sycl-mismatched-free.cpp:[[#@LINE-1]] is released with free [sycl-mismatched-free]
  free(n);
  int *v = new int[8];
  // CHECK-DAG: combine-diag-sycl-mismatched-free.cpp":[[#@LINE+1]]:{{.*}}warning: memory allocated by operator new[] at combine-diag-sycl-mismatched-free.cpp:[[#@LINE-1]] is released with operator delete [sycl-mismatched-free]
  delete v;
}

// A pointer that is null or a new[] array: a null pointer releases
// nothing, so when it releases memory it is the array's.
void nullOrArray() {
  int *h = nullptr;
  if (cond())
    h = new int[8];
  // CHECK-DAG: combine-diag-sycl-mismatched-free.cpp":[[#@LINE+1]]:{{.*}}warning: memory allocated by operator new[] at combine-diag-sycl-mismatched-free.cpp:[[#@LINE-1]] is released with operator delete [sycl-mismatched-free]
  delete h;
}

// A class with a constructor: the new-expression's own cleanup (operator
// delete of the allocation if the constructor throws) is no run-time choice
// of deallocator.
struct Obj {
  Obj();
  int v;
};
void classNew() {
  Obj *o = new Obj;
  // CHECK-DAG: combine-diag-sycl-mismatched-free.cpp":[[#@LINE+1]]:{{.*}}warning: memory allocated by operator new at combine-diag-sycl-mismatched-free.cpp:[[#@LINE-1]] is released with free [sycl-mismatched-free]
  free(o);
}

// Two allocations swapped in a loop (a cycle of copies), then freed.
void swapped(queue &q, int n) {
  float *a = malloc_shared<float>(8, q);
  float *b = malloc_shared<float>(8, q);
  for (int i = 0; i < n; ++i) {
    float *t = a;
    a = b;
    b = t;
  }
  // CHECK-DAG: combine-diag-sycl-mismatched-free.cpp":[[#@LINE+1]]:{{.*}}warning: memory allocated by sycl::malloc_shared at combine-diag-sycl-mismatched-free.cpp:[[#@LINE-7]] (and 1 other site) is released with free [sycl-mismatched-free]
  free(a);
}

// A helper that frees its parameter, given a host USM allocation: judged
// under closed-tu only, at the helper's free.
// CLOSED-DAG: combine-diag-sycl-mismatched-free.cpp":[[#@LINE+1]]:{{.*}}warning: memory allocated by sycl::malloc_host at combine-diag-sycl-mismatched-free.cpp:[[#@LINE+2]] is released with free [sycl-mismatched-free]
void release(float *p) { free(p); }
void viaHelper(queue &q) { release(malloc_host<float>(8, q)); }

// Negatives (silent).
// The matching deallocators.
void matching(queue &q) {
  float *d = malloc_device<float>(64, q);
  sycl::free(d, q);
  float *h = (float *)malloc(64);
  free(h);
  float *n = new float[4];
  delete[] n;
}
// A buffer over malloc memory, then free: the buffer does not own it.
void bufferOverMalloc() {
  float *h = (float *)malloc(64 * sizeof(float));
  { buffer<float, 1> b(h, range<1>(64)); }
  free(h);
}
// An interior pointer of a USM allocation: another bug.
void interior(queue &q) {
  float *d = malloc_device<float>(64, q);
  free(d + 1);
}
// A pointer from an unknown call, and the parameter of an external function.
void unknownPtr() { free(opaque()); }
void extParam(float *p) { free(p); }
// A pointer selected at run time between USM and malloc memory.
void selected(queue &q) {
  float *d = malloc_device<float>(64, q);
  float *h = (float *)malloc(4);
  free(cond() ? d : h);
}
// The function chooses the deallocator at run time.
void choose(queue &q, bool usm) {
  float *d = malloc_shared<float>(64, q);
  if (usm)
    sycl::free(d, q);
  else
    free(d);
}
// A helper given USM and malloc memory.
void mixedHelper(float *p, queue &q) { sycl::free(p, q); }

int main() {
  queue q;
  freeDevice(q);
  deleteShared(q);
  copied(q);
  syclFreeHost(q);
  generic();
  nullOrArray();
  classNew();
  swapped(q, 3);
  viaHelper(q);
  matching(q);
  bufferOverMalloc();
  interior(q);
  unknownPtr();
  selected(q);
  choose(q, true);
  mixedHelper(malloc_shared<float>(4, q), q);
  mixedHelper((float *)malloc(4), q);
}
