// A USM lambda kernel launched from a header inline function, across two TUs
// (T26b). `Copy` is a command-group function object defined in a header: its
// in-class operator() is inline (linkonce_odr), so the kernel lambda in it has
// one closure type in every TU ([basic.def.odr]p14-15) and the device kernel
// KLam is one weak_odr definition for the program. TU1's main submits it with
// two distinct malloc_device pointers, TU2's tu2_api with the same pointer
// twice (and TU2's tu2_distinct with two distinct ones). The facts TU1 derives (noalias on d and s) are an AND over TU1's
// launches only, so they must not be stamped on KLam: the device link keeps
// one KLam for both TUs' launches. Before T26b the default (legacy) host
// facts stamped KLam in place ("the closure is a lambda"). Now:
//  - the default internalizes TU1's launch path from main and stamps the
//    kernel copy KLam.int.<TU1 id>, which only TU1's internal copies launch;
//    KLam stays unstamped;
//  - TU2 has no main, but the legacy facts are read off the body of the
//    function holding the launch alone: tu2_distinct is a root of its own
//    path and launches KLam.int.<TU2 id>; tu2_api launches KLam;
//  - TU-unique kernels stay stamped in place: KLocal (a lambda in a functor
//    of an unnamed namespace) and KMain (a lambda in main);
//  - with --cir-offload-internalize-launch-paths=false, KLam gets nothing,
//    with a remark saying why;
//  - -cir-offload-assume-closed-tu keeps the closed-TU rule (a lambda is
//    stamped in place).
// The device modules of the two TUs are linked (llvm-link; nothing is run).
//
// RUN: rm -rf %t && mkdir -p %t
// DEFINE: %{dev} = %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// DEFINE:   -aux-triple x86_64-unknown-linux-gnu -fclangir -mconstructor-aliases \
// DEFINE:   -DCOMBINE_MOCK_HOST_KERNEL_RESET -I%S/Inputs
// DEFINE: %{host} = %clang_cc1 -std=c++17 -fsycl-is-host -triple x86_64-unknown-linux-gnu \
// DEFINE:   -aux-triple spir64-unknown-unknown -fclangir -mconstructor-aliases \
// DEFINE:   -DCOMBINE_MOCK_HOST_KERNEL_RESET -I%S/Inputs
// DEFINE: %{targets} = -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown
// DEFINE: %{v} = tu1
// DEFINE: %{opts} =
// DEFINE: %{m} = default
// DEFINE: %{merge} = cir-offload-merge -combine %{targets} -input=%t/%{v}.host.cir \
// DEFINE:     -input=%t/%{v}.dev.cir %{opts} -output=%t/%{v}.%{m}.cir > %t/%{v}.%{m}.remarks 2>&1 \
// DEFINE:   && cir-offload-merge -split %{targets} -input=%t/%{v}.%{m}.cir \
// DEFINE:     -output=%t/%{v}.%{m}.split.host.cir -output=%t/%{v}.%{m}.split.dev.cir \
// DEFINE:   && %{dev} -O1 -emit-llvm -x cir %t/%{v}.%{m}.split.dev.cir -o %t/%{v}.%{m}.dev.ll \
// DEFINE:   && %{host} -emit-llvm -x cir %t/%{v}.%{m}.split.host.cir -o %t/%{v}.%{m}.host.ll
//
// RUN: %{dev} -emit-cir %s -o %t/tu1.dev.cir
// RUN: %{host} -emit-cir %s -o %t/tu1.host.cir
// RUN: %{dev} -emit-cir -DTU2 %s -o %t/tu2.dev.cir
// RUN: %{host} -emit-cir -DTU2 %s -o %t/tu2.host.cir
//
// The default (legacy host facts, internalized launch paths).
// RUN: %{merge}
// RUN: FileCheck %s --check-prefix=TU1-REMARK --input-file=%t/tu1.default.remarks
// RUN: FileCheck %s --check-prefix=TU1-DEV --input-file=%t/tu1.default.dev.ll
// RUN: FileCheck %s --check-prefix=TU1-HOST --input-file=%t/tu1.default.host.ll
// REDEFINE: %{v} = tu2
// RUN: %{merge}
// RUN: FileCheck %s --check-prefix=TU2-REMARK --input-file=%t/tu2.default.remarks
// RUN: FileCheck %s --check-prefix=TU2-DEV --input-file=%t/tu2.default.dev.ll
// RUN: FileCheck %s --check-prefix=TU2-HOST --input-file=%t/tu2.default.host.ll
// RUN: llvm-link -S %t/tu1.default.dev.ll %t/tu2.default.dev.ll -o %t/linked.dev.ll
// RUN: FileCheck %s --check-prefix=LINKED --input-file=%t/linked.dev.ll
//
// Internalization off: no fact for KLam.
// REDEFINE: %{v} = tu1
// REDEFINE: %{m} = noint
// REDEFINE: %{opts} = --cir-offload-internalize-launch-paths=false
// RUN: %{merge}
// RUN: FileCheck %s --check-prefix=NOINT --input-file=%t/tu1.noint.remarks
// RUN: FileCheck %s --check-prefix=NOINT-DEV --input-file=%t/tu1.noint.dev.ll
//
// The same under ipo (open, internalized).
// REDEFINE: %{m} = ipo
// REDEFINE: %{opts} = --cir-offload-host-facts=ipo
// RUN: %{merge}
// RUN: FileCheck %s --check-prefix=TU1-DEV --input-file=%t/tu1.ipo.dev.ll
//
// Closed-TU: the closed-TU rule, KLam stamped in place.
// REDEFINE: %{m} = closed
// REDEFINE: %{opts} = -cir-offload-assume-closed-tu
// RUN: %{merge}
// RUN: FileCheck %s --check-prefix=CLOSED --input-file=%t/tu1.closed.dev.ll

#include "combine-canary-mock.h"
using namespace sycl;

class KLam;
class KLocal;
class KMain;

// "Header": shared by every TU.
struct Copy {
  float *dst, *src;
  void operator()(handler &h) const {
    float *d = dst, *s = src;
    h.parallel_for<KLam>(range<1>(64),
                         [=](id<1> i) { d[i[0]] = s[i[0]] + 1; });
  }
};

// TU-unique: the functor and its call operator have internal linkage.
namespace {
struct CopyLocal {
  float *dst, *src;
  void operator()(handler &h) const {
    float *d = dst, *s = src;
    h.parallel_for<KLocal>(range<1>(64),
                           [=](id<1> i) { d[i[0]] = s[i[0]] + 1; });
  }
};
} // namespace

#ifndef TU2
int main() {
  queue q;
  float *a = malloc_device<float>(64, q);
  float *b = malloc_device<float>(64, q);
  q.submit(Copy{a, b});
  q.submit(CopyLocal{a, b});
  // TU-unique: a lambda in a non-inline function.
  q.submit([&](handler &h) {
    h.parallel_for<KMain>(range<1>(64), [=](id<1> i) { a[i[0]] = b[i[0]]; });
  });
  return 0;
}
#else
// TU2 (no main): d and s are the same allocation here...
void tu2_api(queue &q) {
  float *x = malloc_device<float>(64, q);
  q.submit(Copy{x, x});
}
// ... and distinct here: the legacy facts of this launch are read off this
// function's body alone, so they hold for any caller of it: it is a root of
// its own launch path (kept in place, external), which launches the copy
// KLam.int.<TU2 id>.
void tu2_distinct(queue &q) {
  float *x = malloc_device<float>(64, q);
  float *y = malloc_device<float>(64, q);
  q.submit(Copy{x, y});
}
#endif

// TU1-REMARK-DAG: remark: launch-path-internalization: kernel @_ZTS4KLam -> @_ZTS4KLam.int.[[ID:[0-9a-f]+]] for 1 launch marker through the internalized launch path from root @main
// TU1-REMARK-DAG: remark: kernel-param-noalias: kernel @_ZTS4KLam arg 0 (pointer 'd'): not stamped
// TU1-REMARK-DAG: remark: kernel-param-noalias: kernel @_ZTS4KLam arg 1 (pointer 's'): not stamped
// TU1-REMARK-DAG: remark: kernel-param-noalias: kernel @_ZTS4KLam.int.[[ID]] arg 0 (pointer 'd'): noalias (a USM allocation) through the internalized launch path from root @main
// TU1-REMARK-DAG: remark: kernel-param-noalias: kernel @_ZTS4KLam.int.[[ID]] arg 1 (pointer 's'): noalias (a USM allocation) through the internalized launch path from root @main
// TU1-REMARK-DAG: remark: kernel-param-noalias: kernel @_ZTS6KLocal arg 0 (pointer 'd'): noalias (a USM allocation)
// TU1-REMARK-DAG: remark: kernel-param-noalias: kernel @_ZTS5KMain arg 0 (pointer 'a'): noalias (a USM allocation)
// TU1-REMARK-DAG: remark: launch-path-internalization: kept 1 of 1 kernel copies

// TU1's device module: the shared KLam is not stamped, its copy is; the
// TU-unique kernels are stamped in place.
// TU1-DEV: define weak_odr spir_kernel void @_ZTS4KLam(
// TU1-DEV-NOT: noalias
// TU1-DEV-SAME: {{$}}
// TU1-DEV: define weak_odr spir_kernel void @_ZTS4KLam.int.{{[0-9a-f]+}}(ptr addrspace(1) noalias {{[^,]*}}, ptr addrspace(1) noalias
// TU1-DEV: define weak_odr spir_kernel void @_ZTS6KLocal(ptr addrspace(1) noalias {{[^,]*}}, ptr addrspace(1) noalias
// TU1-DEV: define weak_odr spir_kernel void @_ZTS5KMain(ptr addrspace(1) noalias {{[^,]*}}, ptr addrspace(1) noalias

// TU1's host module: main submits through internal copies; the original
// (linkonce_odr) chain, which TU2's call may bind to, names KLam.
// TU1-HOST-LABEL: define dso_local noundef i32 @main()
// TU1-HOST: call void @_ZN4sycl3_V15queue6submitI4CopyEENS0_5eventET_.int(
// TU1-HOST-LABEL: define linkonce_odr void @_ZN4sycl3_V17handler11StoreLambdaI4KLamZNK4CopyclERS1_EUlNS0_2idILi1EEEE_EEvRKT0_(
// TU1-HOST: call {{.*}}@_ZN4sycl3_V16detail19getDeviceKernelInfoI4KLamEERNS1_16DeviceKernelInfoEv()
// TU1-HOST-LABEL: define internal void @_ZN4sycl3_V17handler11StoreLambdaI4KLamZNK4CopyclERS1_EUlNS0_2idILi1EEEE_EEvRKT0_.int(
// TU1-HOST: call {{.*}}@_ZN4sycl3_V16detail19getDeviceKernelInfoI4KLamEERNS1_16DeviceKernelInfoEv.int()

// TU2: tu2_api launches the unstamped KLam, tu2_distinct the copy.
// TU2-REMARK: remark: launch-path-internalization: kernel @_ZTS4KLam -> @_ZTS4KLam.int.[[ID2:[0-9a-f]+]] for 1 launch marker through the internalized launch path from root @_Z12tu2_distinctRN4sycl3_V15queueE
// TU2-DEV: define weak_odr spir_kernel void @_ZTS4KLam(
// TU2-DEV-NOT: noalias
// TU2-DEV-SAME: {{$}}
// TU2-DEV: define weak_odr spir_kernel void @_ZTS4KLam.int.{{[0-9a-f]+}}(ptr addrspace(1) noalias {{[^,]*}}, ptr addrspace(1) noalias
// TU2-HOST-LABEL: define dso_local void @_Z7tu2_apiRN4sycl3_V15queueE(
// TU2-HOST: call void @_ZN4sycl3_V15queue6submitI4CopyEENS0_5eventET_(
// TU2-HOST-LABEL: define dso_local void @_Z12tu2_distinctRN4sycl3_V15queueE(
// TU2-HOST: call void @_ZN4sycl3_V15queue6submitI4CopyEENS0_5eventET_.int(

// The one KLam of the program carries no noalias; each TU's copy is its own.
// LINKED: define weak_odr spir_kernel void @_ZTS4KLam(
// LINKED-NOT: noalias
// LINKED-SAME: {{$}}
// LINKED-COUNT-2: define weak_odr spir_kernel void @_ZTS4KLam.int.{{[0-9a-f]+}}(ptr addrspace(1) noalias

// NOINT-DAG: remark: kernel-param-noalias: kernel @_ZTS4KLam arg 0 (pointer 'd'): not stamped (the kernel is visible to other TUs; facts this TU derives hold only through an internalized launch path, and launch-path internalization is off)
// NOINT-DAG: remark: kernel-param-noalias: kernel @_ZTS6KLocal arg 0 (pointer 'd'): noalias (a USM allocation)
// NOINT-DEV: define weak_odr spir_kernel void @_ZTS4KLam(
// NOINT-DEV-NOT: noalias
// NOINT-DEV-SAME: {{$}}
// NOINT-DEV-NOT: KLam.int

// CLOSED: define weak_odr spir_kernel void @_ZTS4KLam(ptr addrspace(1) noalias {{[^,]*}}, ptr addrspace(1) noalias
// CLOSED-NOT: .int.
