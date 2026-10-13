// End to end: the offload combine warns when a kernel's captured closure has
// a different layout on the host and on the device, as `#ifdef
// __SYCL_DEVICE_ONLY__` around a captured variable or member gives
// (cir-offload-detect-host-device-closure-layout-mismatch, tag
// sycl-closure-layout-mismatch of --cir-offload-host-diagnostics). The
// sizeof(KernelType) == KernelSize check of the runtime only compares the
// total size, so an equal-size divergence goes unnoticed. The warning is at
// the capture, with a note at the kernel. The SYCL and C++ library parts are
// the reduced headers of the combine-canary tests.
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fclangir -I%S/Inputs -emit-cir %s -o %t/dev.cir
// RUN: %clang_cc1 -std=c++17 -fsycl-is-host -triple x86_64-unknown-linux-gnu \
// RUN:   -aux-triple spir64-unknown-unknown -fclangir -I%S/Inputs -emit-cir %s -o %t/host.cir
//
// Open mode (the default) and closed-tu give the same warnings.
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir -output=%t/open.cir 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=warning:
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir -cir-offload-assume-closed-tu -output=%t/closed.cir 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=warning:
//
// Off, or only the other checks: no warning, and the same output (the pass
// changes nothing).
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir --cir-offload-host-diagnostics=none -output=%t/none.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NONE --allow-empty
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir \
// RUN:   --cir-offload-host-diagnostics=sycl-host-ptr-in-kernel,sycl-mismatched-free \
// RUN:   -output=%t/other.cir 2>&1 | FileCheck %s --check-prefix=NONE --allow-empty
// RUN: diff %t/open.cir %t/none.cir
// NONE-NOT: warning:
//
// Every warning has a note at the kernel.
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir -output=%t/open2.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOTE
// NOTE-COUNT-4: note: in the SYCL kernel defined here
// NOTE-NOT: note:

#include "combine-canary-mock.h"

using namespace sycl;

float *ext(float *);

class KScalar;
class KMember;
class KPointer;
class KAccessor;
class KSame;
class KStructPtrs;
class KUsm;
class KAccOk;
class KUnknownPtr;
template <typename T> class KTemplate;

#ifdef __SYCL_DEVICE_ONLY__
using Elem = float;
#else
using Elem = int;
#endif

struct Params {
  int n;
#ifdef __SYCL_DEVICE_ONLY__
  float scale;
#else
  int scale;
#endif
};

struct Ptrs {
  float *in;
  float *out;
  int n;
};

template <typename T> void scaled(queue &q, T *p, T s) {
  q.submit([&](handler &h) {
    h.parallel_for<KTemplate<T>>(range<1>(16),
                                 [=](id<1> i) { p[i[0]] = p[i[0]] * s; });
  });
}

int main() {
  queue q;
  float *out = malloc_shared<float>(16, q);

#ifdef __SYCL_DEVICE_ONLY__
  float s = 2;
#else
  int s = 2;
#endif
  q.submit([&](handler &h) {
    h.parallel_for<KScalar>(range<1>(16), [=](id<1> i) {
      // CHECK-DAG: combine-diag-sycl-closure-layout-mismatch.cpp":[[#@LINE+1]]:{{.*}}warning: SYCL kernel captured-closure layout differs between host and device (captured field 's': host: int32 at offset 8, device: float at offset 8); likely a #ifdef __SYCL_DEVICE_ONLY__ or divergent [=] capture -- undefined behavior [sycl-closure-layout-mismatch]
      out[i[0]] = s;
    });
  });

  Params prm{16, 3};
  q.submit([&](handler &h) {
    h.parallel_for<KMember>(range<1>(16), [=](id<1> i) {
      // CHECK-DAG: combine-diag-sycl-closure-layout-mismatch.cpp":[[#@LINE+1]]:{{.*}}warning: SYCL kernel captured-closure layout differs between host and device (captured field 'prm': host: int32 at offset 12, device: float at offset 12); likely a #ifdef __SYCL_DEVICE_ONLY__ or divergent [=] capture -- undefined behavior [sycl-closure-layout-mismatch]
      out[i[0]] = prm.scale * prm.n;
    });
  });

#ifdef __SYCL_DEVICE_ONLY__
  long where = 0;
#else
  float *where = out;
#endif
  q.submit([&](handler &h) {
    h.parallel_for<KPointer>(range<1>(16), [=](id<1> i) {
      // CHECK-DAG: combine-diag-sycl-closure-layout-mismatch.cpp":[[#@LINE+1]]:{{.*}}warning: SYCL kernel captured-closure layout differs between host and device (captured field 'where': host: pointer at offset 8, device: int64 at offset 8); likely a #ifdef __SYCL_DEVICE_ONLY__ or divergent [=] capture -- undefined behavior [sycl-closure-layout-mismatch]
      out[i[0]] = (float)(long)where;
    });
  });

  Elem data[16];
  buffer<Elem, 1> buf(data, range<1>(16));
  q.submit([&](handler &h) {
    auto acc = buf.get_access<access::mode::read_write>(h);
    h.parallel_for<KAccessor>(range<1>(16), [=](id<1> i) {
      // CHECK-DAG: combine-diag-sycl-closure-layout-mismatch.cpp":[[#@LINE+1]]:{{.*}}warning: SYCL kernel captured-closure layout differs between host and device (captured accessor 'acc': host: element type 'int', device: element type 'float'); likely a #ifdef __SYCL_DEVICE_ONLY__ or divergent [=] capture -- undefined behavior [sycl-closure-layout-mismatch]
      acc[i] = 1;
    });
  });

  // Negatives (silent): the same captures on both sides, a struct of
  // pointers, a USM pointer, an accessor of one element type, a pointer from
  // an unknown call, one lambda instantiated for two types.
  int n = 16;
  double d = 1;
  q.submit([&](handler &h) {
    h.parallel_for<KSame>(range<1>(16),
                          [=](id<1> i) { out[i[0]] = n * d; });
  });
  Ptrs ps{out, out, 16};
  q.submit([&](handler &h) {
    h.parallel_for<KStructPtrs>(range<1>(16), [=](id<1> i) {
      ps.out[i[0]] = ps.in[i[0]] + ps.n;
    });
  });
  float *usm = malloc_device<float>(16, q);
  q.submit([&](handler &h) {
    h.parallel_for<KUsm>(range<1>(16), [=](id<1> i) { usm[i[0]] = 0; });
  });
  float fdata[16];
  buffer<float, 1> fbuf(fdata, range<1>(16));
  q.submit([&](handler &h) {
    auto facc = fbuf.get_access<access::mode::read_write>(h);
    h.parallel_for<KAccOk>(range<1>(16), [=](id<1> i) { facc[i] = 1; });
  });
  float *unknown = ext(out);
  q.submit([&](handler &h) {
    h.parallel_for<KUnknownPtr>(range<1>(16),
                                [=](id<1> i) { unknown[i[0]] = 0; });
  });
  scaled(q, out, 2.0f);
  double *dout = malloc_shared<double>(16, q);
  scaled(q, dout, 2.0);
  q.wait();
}
