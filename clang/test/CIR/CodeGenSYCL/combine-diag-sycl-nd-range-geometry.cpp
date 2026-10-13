// End to end: the offload combine warns at a launch whose constant nd_range
// throws errc::nd_range: a local size that does not divide the global size,
// a local size of 0, or a local size other than the kernel's
// [[sycl::reqd_work_group_size]] (SYCL 2020 sec:handlerClass, parallel_for;
// sec:kernel.attributes); with --cir-offload-max-work-group-size=N also a
// work-group of more than N work-items: cir-offload-host-diagnostics, check
// sycl-nd-range-geometry. The warning is at the launch, with a note at the
// kernel. The SYCL and C++ library parts are the reduced headers of the
// combine-canary tests.
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fclangir -I%S/Inputs -emit-cir %s -o %t/dev.cir
// RUN: %clang_cc1 -std=c++17 -fsycl-is-host -triple x86_64-unknown-linux-gnu \
// RUN:   -aux-triple spir64-unknown-unknown -fclangir -I%S/Inputs -emit-cir %s -o %t/host.cir
//
// CIRGen keeps the source's required work-group size on the device kernel.
// RUN: FileCheck %s --check-prefix=CIRGEN --input-file=%t/dev.cir
// CIRGEN: cir.func {{.*}}@_ZTS5KReqd() cc(spir_kernel) attributes {cir.reqd_work_group_size = [64 : i32]
// CIRGEN: cir.func {{.*}}@_ZTS7KReqdOk() cc(spir_kernel) attributes {cir.reqd_work_group_size = [64 : i32]
// The lowering gives the metadata classic CodeGen emits.
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fclangir -I%S/Inputs -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=LLVM
// LLVM: define {{.*}}spir_kernel void @_ZTS5KReqd() {{.*}}!work_group_num_dim ![[N:[0-9]+]] !reqd_work_group_size ![[WG:[0-9]+]]
// LLVM-DAG: ![[N]] = !{i32 1}
// LLVM-DAG: ![[WG]] = !{i32 64, i32 1, i32 1}
//
// Open mode (the default) and closed-tu give the same warnings.
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir -output=%t/open.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK --implicit-check-not=warning:
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir -cir-offload-assume-closed-tu -output=%t/closed.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK --implicit-check-not=warning:
// The source's required size stays (the stamp never overrides it).
// RUN: FileCheck %s --check-prefix=STAMP --input-file=%t/closed.cir
// STAMP: @_ZTS5KReqd() {{.*}}cir.reqd_work_group_size = [64 : i32]
//
// The opt-in maximum work-group size.
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir --cir-offload-max-work-group-size=1024 -output=%t/max.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK,MAX --implicit-check-not=warning:
//
// Off, or only the other checks: no warning, and the same output (the pass
// changes nothing).
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir --cir-offload-host-diagnostics=none -output=%t/none.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NONE --allow-empty
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir \
// RUN:   --cir-offload-host-diagnostics=sycl-host-ptr-in-kernel,sycl-host-deref-device-usm,sycl-buffer-same-host-ptr,sycl-mismatched-free \
// RUN:   -output=%t/other.cir 2>&1 | FileCheck %s --check-prefix=NONE --allow-empty
// RUN: diff %t/open.cir %t/none.cir
// RUN: diff %t/open.cir %t/max.cir
// NONE-NOT: warning:
//
// Every warning has a note at the kernel.
// RUN: cir-offload-merge -combine -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown \
// RUN:   -input=%t/host.cir -input=%t/dev.cir -output=%t/open2.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOTE
// NOTE-COUNT-2: note: the kernel launched is defined here; the launch throws errc::nd_range: the global size must be a multiple of the local size in each dimension (SYCL 2020 sec:handlerClass, parallel_for)
// NOTE: note: the kernel requires this work-group size; a launch with another local size throws errc::nd_range (SYCL 2020 sec:kernel.attributes)
// NOTE-NOT: note:

#include "combine-canary-mock.h"

using namespace sycl;

unsigned long runtime();

class KBad;
class KZero;
class KOk;
class KReqd;
class KReqdOk;
class KBig;
class KUnknown;
class KRange;
class KHelper;
class KEmpty;

// A helper launched with a good and a bad local size: the bad one holds in
// one calling context only, silent.
void launch(queue &q, unsigned long g, unsigned long l) {
  q.submit([&](handler &h) {
    h.parallel_for<KHelper>(nd_range<1>(range<1>(g), range<1>(l)),
                            [=](nd_item<1> it) {});
  });
}

int main() {
  queue q;
  q.submit([&](handler &h) {
    // CHECK-DAG: combine-diag-sycl-nd-range-geometry.cpp":[[#@LINE+1]]:{{.*}}warning: nd_range global size 1000 is not a multiple of local size 256 in dimension 0 [sycl-nd-range-geometry]
    h.parallel_for<KBad>(nd_range<1>(range<1>(1000), range<1>(256)),
                         [=](nd_item<1> it) {});
  });
  q.submit([&](handler &h) {
    // CHECK-DAG: combine-diag-sycl-nd-range-geometry.cpp":[[#@LINE+1]]:{{.*}}warning: nd_range local size is 0 in dimension 0 (global size 1024) [sycl-nd-range-geometry]
    h.parallel_for<KZero>(nd_range<1>(range<1>(1024), range<1>(0)),
                          [=](nd_item<1> it) {});
  });
  q.submit([&](handler &h) {
    // CHECK-DAG: combine-diag-sycl-nd-range-geometry.cpp":[[#@LINE+1]]:{{.*}}warning: nd_range local size 128 does not match the kernel's reqd_work_group_size(64) [sycl-nd-range-geometry]
    h.parallel_for<KReqd>(nd_range<1>(range<1>(1024), range<1>(128)),
                          [=](nd_item<1> it)
                              [[sycl::reqd_work_group_size(64)]] {});
  });
  q.submit([&](handler &h) {
    // MAX-DAG: combine-diag-sycl-nd-range-geometry.cpp":[[#@LINE+1]]:{{.*}}warning: nd_range work-group size 2048 (2048 work-items) exceeds the maximum work-group size 1024 given by --cir-offload-max-work-group-size [sycl-nd-range-geometry]
    h.parallel_for<KBig>(nd_range<1>(range<1>(4096), range<1>(2048)),
                         [=](nd_item<1> it) {});
  });
  // Negatives (silent): a divisible nd_range, a matching required size, a
  // global size computed at run time, an empty index space, a plain range.
  q.submit([&](handler &h) {
    h.parallel_for<KOk>(nd_range<1>(range<1>(1024), range<1>(256)),
                        [=](nd_item<1> it) {});
  });
  q.submit([&](handler &h) {
    h.parallel_for<KReqdOk>(nd_range<1>(range<1>(1024), range<1>(64)),
                            [=](nd_item<1> it)
                                [[sycl::reqd_work_group_size(64)]] {});
  });
  unsigned long n = runtime();
  q.submit([&](handler &h) {
    h.parallel_for<KUnknown>(nd_range<1>(range<1>(n), range<1>(256)),
                             [=](nd_item<1> it) {});
  });
  q.submit([&](handler &h) {
    h.parallel_for<KEmpty>(nd_range<1>(range<1>(0), range<1>(0)),
                           [=](nd_item<1> it) {});
  });
  q.submit([&](handler &h) {
    h.parallel_for<KRange>(range<1>(1000), [=](id<1> i) {});
  });
  launch(q, 1000, 100);
  launch(q, 1000, 300);
}
