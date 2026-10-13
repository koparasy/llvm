// End to end: the offload combine warns when host code reads or writes
// memory that the host facts prove is a device USM allocation, which the
// host cannot access (SYCL 2020 sec:usm, table.usm.allocs):
// cir-offload-host-diagnostics, check sycl-host-deref-device-usm. The warning
// is at the host access, with a note at the allocation. The SYCL and C++
// library parts are the reduced headers of the combine-canary tests, plus
// sycl::malloc with a usm::alloc kind and aligned_alloc_device below.
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fclangir -I%S/Inputs -emit-cir %s -o %t/dev.cir
// RUN: %clang_cc1 -std=c++17 -fsycl-is-host -triple x86_64-unknown-linux-gnu \
// RUN:   -aux-triple spir64-unknown-unknown -fclangir -I%S/Inputs -emit-cir %s -o %t/host.cir
//
// CIRGen records the allocation kind on the usm.alloc declarations.
// RUN: FileCheck %s --check-prefix=KIND --input-file=%t/host.cir
// KIND-DAG: @_ZN4sycl3_V113malloc_deviceIfEEPT_mRKNS0_5queueE({{.*}}cir.sycl_host_api = "usm.alloc"{{.*}}cir.sycl_usm_kind = "device"
// KIND-DAG: @_ZN4sycl3_V113malloc_sharedIfEEPT_mRKNS0_5queueE({{.*}}cir.sycl_host_api = "usm.alloc"{{.*}}cir.sycl_usm_kind = "shared"
// KIND-DAG: @_ZN4sycl3_V111malloc_hostIfEEPT_mRKNS0_5queueE({{.*}}cir.sycl_host_api = "usm.alloc"{{.*}}cir.sycl_usm_kind = "host"
// KIND-DAG: @_ZN4sycl3_V120aligned_alloc_deviceEmmRKNS0_5queueE({{.*}}cir.sycl_host_api = "usm.alloc"{{.*}}cir.sycl_usm_kind = "device"
// KIND-DAG: @_ZN4sycl3_V16mallocEmRKNS0_5queueENS0_3usm5allocE({{.*}}cir.sycl_host_api = "usm.alloc"{{.*}}cir.sycl_usm_kind_arg = 2 : i32
//
// The default (open, internalized launch paths) and closed-tu give the same
// warnings, but for the access in an externally visible helper: closed-tu
// warns at the access (every caller is seen); in open mode another TU may
// call it with host memory, so the warning is at the call that passes it
// device memory, with a note at the access in the helper.
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir -output=%t/open.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK,OPEN --implicit-check-not=warning:
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
// RUN:   -input=%t/host.cir -input=%t/dev.cir --cir-offload-host-diagnostics=sycl-host-ptr-in-kernel \
// RUN:   -output=%t/other.cir 2>&1 | FileCheck %s --check-prefix=NONE --allow-empty
// RUN: diff %t/open.cir %t/none.cir
//
// Each warning has a note at the allocation: 8 warnings by default; the one
// at the call has a note at the access in the callee too.
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir -output=%t/open2.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOTE --implicit-check-not=note:
// NOTE-COUNT-8: note: the device USM memory is allocated here; device allocations are not accessible on the host (SYCL 2020 sec:usm)
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir -output=%t/open3.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CALLNOTE
// CALLNOTE-COUNT-1: note: the called function accesses the memory here
// CALLNOTE-NOT: note: the called function
// NONE-NOT: warning:

#include "combine-canary-mock.h"

namespace sycl {
inline namespace _V1 {
namespace usm {
enum class alloc { host = 0, device = 1, shared = 2, unknown = 3 };
}
void *malloc(unsigned long bytes, const queue &q, usm::alloc kind);
void *aligned_alloc_device(unsigned long alignment, unsigned long bytes,
                           const queue &q);
} // namespace _V1
} // namespace sycl

using namespace sycl;

extern "C" void *memcpy(void *, const void *, unsigned long);
float *opaque();
void escape(float **);
bool cond();
usm::alloc pickKind();

class KScale;

struct Pair {
  float a, b;
};
struct Holder {
  float *p;
  int n;
};

// An internal helper that only gets device memory.
static void initDevice(float *p, int n) {
  for (int i = 0; i < n; ++i)
    // CHECK-DAG: combine-diag-sycl-host-deref-device-usm.cpp":[[#@LINE+1]]:{{.*}}warning: host access to device USM memory allocated at combine-diag-sycl-host-deref-device-usm.cpp:{{[0-9]+}} [sycl-host-deref-device-usm]
    p[i] = 0.f;
}

// An internal helper called with host and device memory: silent.
static void initMixed(float *p) { p[0] = 1.f; }

// Externally visible: another TU may call it with host memory.
void initExtern(float *p) {
  // CLOSED-DAG: combine-diag-sycl-host-deref-device-usm.cpp":[[#@LINE+2]]:{{.*}}warning: host access to device USM memory allocated at
  // OPEN-DAG: combine-diag-sycl-host-deref-device-usm.cpp":[[#@LINE+1]]:{{.*}}note: the called function accesses the memory here
  p[0] = 2.f;
}

// Another TU's caller passes whatever it likes: silent.
void fromParam(float *p) { p[1] = 3.f; }

int main() {
  queue q;
  float host[64];

  // A store through malloc_device memory.
  // CHECK-DAG: combine-diag-sycl-host-deref-device-usm.cpp":[[#@LINE+1]]:{{.*}}note: the device USM memory is allocated here
  float *d = malloc_device<float>(64, q);
  // CHECK-DAG: combine-diag-sycl-host-deref-device-usm.cpp":[[#@LINE+1]]:{{.*}}warning: host access to device USM memory allocated at combine-diag-sycl-host-deref-device-usm.cpp:[[#@LINE-1]] [sycl-host-deref-device-usm]
  d[0] = 1.f;

  // A read of sycl::malloc(..., usm::alloc::device) memory.
  float *g = (float *)sycl::malloc(64 * sizeof(float), q, usm::alloc::device);
  // CHECK-DAG: combine-diag-sycl-host-deref-device-usm.cpp":[[#@LINE+1]]:{{.*}}warning: host access to device USM memory allocated at combine-diag-sycl-host-deref-device-usm.cpp:[[#@LINE-1]]
  float x = g[3];

  // A struct copied out of aligned_alloc_device memory, and the C memcpy
  // from it.
  Pair *pd = (Pair *)aligned_alloc_device(64, sizeof(Pair), q);
  // CHECK-DAG: combine-diag-sycl-host-deref-device-usm.cpp":[[#@LINE+1]]:{{.*}}warning: host access to device USM memory allocated at combine-diag-sycl-host-deref-device-usm.cpp:[[#@LINE-1]]
  Pair v = *pd;
  float *d2 = malloc_device<float>(64, q);
  initDevice(d2, 64);
  // CHECK-DAG: combine-diag-sycl-host-deref-device-usm.cpp":[[#@LINE+1]]:{{.*}}warning: host access to device USM memory allocated at combine-diag-sycl-host-deref-device-usm.cpp:[[#@LINE-2]]
  memcpy(host, d2, 64 * sizeof(float));

  // Through a struct member.
  Holder h{malloc_device<float>(4, q), 4};
  // CHECK-DAG: combine-diag-sycl-host-deref-device-usm.cpp":[[#@LINE+1]]:{{.*}}warning: host access to device USM memory allocated at combine-diag-sycl-host-deref-device-usm.cpp:[[#@LINE-1]]
  h.p[2] = 3.f;

  // In the command group function, which runs on the host.
  float *d3 = malloc_device<float>(64, q);
  q.submit([&](handler &cgh) {
    // CHECK-DAG: combine-diag-sycl-host-deref-device-usm.cpp":[[#@LINE+1]]:{{.*}}warning: host access to device USM memory allocated at combine-diag-sycl-host-deref-device-usm.cpp:[[#@LINE-2]]
    d3[0] = 0.f;
    // The kernel's accesses run on the device: silent.
    cgh.parallel_for<KScale>(range<1>(64), [=](id<1> i) { d3[i[0]] *= 2.f; });
  });

  // Silent: shared and host USM, an unknown kind, a select, a helper called
  // with both kinds of memory, unknown memory, an escaped holder, a
  // parameter, and the runtime's memory commands. (initExtern(dext) is
  // checked above.)
  float *s = malloc_shared<float>(64, q);
  s[0] = 1.f;
  float *hm = malloc_host<float>(64, q);
  hm[0] = 1.f;
  float *k = (float *)sycl::malloc(64, q, pickKind());
  k[0] = 1.f;
  float *dsel = malloc_device<float>(64, q);
  float *sel = cond() ? host : dsel;
  sel[1] = 1.f;
  float *dmix = malloc_device<float>(64, q);
  initMixed(host);
  initMixed(dmix);
  float *u = opaque();
  u[0] = 1.f;
  float *e = malloc_device<float>(4, q);
  escape(&e);
  e[0] = 1.f;
  fromParam(host);
  float *dext = malloc_device<float>(4, q);
  // OPEN-DAG: combine-diag-sycl-host-deref-device-usm.cpp":[[#@LINE+1]]:{{.*}}warning: host access to device USM memory allocated at combine-diag-sycl-host-deref-device-usm.cpp:[[#@LINE-1]]
  initExtern(dext);
  float *dcpy = malloc_device<float>(64, q);
  q.memcpy(host, dcpy, 64 * sizeof(float));
  free(dcpy, q);
  q.wait();
  return (int)(x + v.a + host[0]);
}
