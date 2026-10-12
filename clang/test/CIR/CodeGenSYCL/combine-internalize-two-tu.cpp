// Launch-path internalization across two TUs (T26). Both TUs include the
// inline launch function `launch`, whose kernel KShared is one weak_odr
// device definition for the whole program and whose host function is
// linkonce_odr: the linker binds every TU's call of `launch` to one copy.
// TU1's main passes three distinct allocations; TU2's tu2_api passes the
// same allocation twice. Without -cir-offload-assume-closed-tu:
//  - TU1 internalizes the path from main: main calls an internal copy
//    `launch.int`, which launches the kernel copy KShared.int.<TU1 id>, and
//    that copy (not KShared) gets noalias; the original `launch` still
//    launches the unstamped KShared;
//  - TU2 has no closed root (no main), but tu2_api, which nothing in TU2
//    calls, is an open root (T28): kept in place, it calls TU2's own
//    internal copy `launch.int`, which launches KShared.int.<TU2 id>; that
//    copy gets the facts that hold for any caller of tu2_api: c (y) is
//    noalias, a and b (both x) are not.
// So whichever copy of `launch` the linker keeps, no TU's call reaches
// another TU's kernel copy: each internal copy launches the copy of its own
// TU, under a name only that TU mints, and the original `launch` launches the
// unstamped KShared. The linked host module (llvm-link of the two host
// modules; nothing is run) shows it.
//
// RUN: rm -rf %t && mkdir -p %t
// DEFINE: %{dev} = %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// DEFINE:   -aux-triple x86_64-unknown-linux-gnu -fclangir -I%S/Inputs
// DEFINE: %{host} = %clang_cc1 -std=c++17 -fsycl-is-host -triple x86_64-unknown-linux-gnu \
// DEFINE:   -aux-triple spir64-unknown-unknown -fclangir -I%S/Inputs
// DEFINE: %{targets} = -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown
//
// RUN: %{dev} -emit-cir %s -o %t/tu1.dev.cir
// RUN: %{host} -emit-cir %s -o %t/tu1.host.cir
// RUN: cir-offload-merge -combine --remarks-filter="cir-offload.*" %{targets} -input=%t/tu1.host.cir -input=%t/tu1.dev.cir \
// RUN:   --cir-offload-host-facts=ipo -output=%t/tu1.cir 2>&1 | FileCheck %s --check-prefix=TU1-REMARK \
// RUN:   --implicit-check-not=kernel-name-redirect:
// RUN: cir-offload-merge -split %{targets} -input=%t/tu1.cir -output=%t/tu1.split.host.cir \
// RUN:   -output=%t/tu1.split.dev.cir
// RUN: %{dev} -O1 -emit-llvm -x cir %t/tu1.split.dev.cir -o %t/tu1.dev.ll
// RUN: %{host} -emit-llvm -x cir %t/tu1.split.host.cir -o %t/tu1.host.ll
// RUN: FileCheck %s --check-prefix=TU1-DEV --input-file=%t/tu1.dev.ll
// RUN: FileCheck %s --check-prefix=TU1-HOST --input-file=%t/tu1.host.ll
// RUN: FileCheck %s --check-prefix=TU1-NAME --input-file=%t/tu1.host.ll
//
// RUN: %{dev} -emit-cir -DTU2 %s -o %t/tu2.dev.cir
// RUN: %{host} -emit-cir -DTU2 %s -o %t/tu2.host.cir
// RUN: cir-offload-merge -combine --remarks-filter="cir-offload.*" %{targets} -input=%t/tu2.host.cir -input=%t/tu2.dev.cir \
// RUN:   --cir-offload-host-facts=ipo -output=%t/tu2.cir 2>&1 | FileCheck %s --check-prefix=TU2-REMARK
// RUN: cir-offload-merge -split %{targets} -input=%t/tu2.cir -output=%t/tu2.split.host.cir \
// RUN:   -output=%t/tu2.split.dev.cir
// RUN: %{dev} -O1 -emit-llvm -x cir %t/tu2.split.dev.cir -o %t/tu2.dev.ll
// RUN: %{host} -emit-llvm -x cir %t/tu2.split.host.cir -o %t/tu2.host.ll
// RUN: FileCheck %s --check-prefix=TU2-DEV --input-file=%t/tu2.dev.ll
//
// RUN: llvm-link -S %t/tu1.host.ll %t/tu2.host.ll -o %t/linked.host.ll
// RUN: FileCheck %s --check-prefix=LINKED --input-file=%t/linked.host.ll

#include "combine-canary-mock.h"
using namespace sycl;

class KShared;

inline void launch(queue &q, float *a, float *b, float *c) {
  q.submit([&](handler &h) {
    h.parallel_for<KShared>(range<1>(64),
                            [=](id<1> i) { c[i[0]] = a[i[0]] + b[i[0]]; });
  });
}

#ifndef TU2
int main() {
  queue q;
  float *a = malloc_shared<float>(64, q);
  float *b = malloc_shared<float>(64, q);
  float *c = malloc_shared<float>(64, q);
  launch(q, a, b, c);
  return 0;
}
#else
// TU2: a and b are the same allocation.
void tu2_api(queue &q) {
  float *x = malloc_shared<float>(64, q);
  float *y = malloc_shared<float>(64, q);
  launch(q, x, x, y);
}
#endif

// TU1-REMARK: launch-path-internalization: kernel @_ZTS7KShared -> @_ZTS7KShared.int.[[ID:[0-9a-f]+]] for 1 launch marker through the internalized launch path from root @main
// TU1-REMARK-DAG: kernel-param-noalias: kernel @_ZTS7KShared arg {{[0-9]}} (pointer '{{[abc]}}'): not stamped
// TU1-REMARK-DAG: kernel-param-noalias: kernel @_ZTS7KShared.int.[[ID]] arg 0 (pointer 'c'): noalias (a USM allocation ({{.*}})) through the internalized launch path from root @main
// TU1-REMARK-DAG: kernel-param-noalias: kernel @_ZTS7KShared.int.[[ID]] arg 1 (pointer 'a'): noalias (a USM allocation ({{.*}})) through the internalized launch path from root @main
// TU1-REMARK-DAG: kernel-param-noalias: kernel @_ZTS7KShared.int.[[ID]] arg 2 (pointer 'b'): noalias (a USM allocation ({{.*}})) through the internalized launch path from root @main
// TU1-REMARK: launch-path-internalization: kept 1 of 1 kernel copies

// TU1's device module: the original kernel is not stamped, its copy is.
// TU1-DEV: define weak_odr spir_kernel void @_ZTS7KShared(
// TU1-DEV-NOT: noalias
// TU1-DEV-SAME: {{$}}
// TU1-DEV: define weak_odr spir_kernel void @_ZTS7KShared.int.{{[0-9a-f]+}}(ptr addrspace(1) noalias {{[^,]*}}, ptr addrspace(1) noalias {{[^,]*}}, ptr addrspace(1) noalias {{[^,]*}})

// TU1's host module: the kernel copy's name is only in an internal string,
// the getter copy that names it is internal and only TU1's internal path
// calls it; the original `launch` (linkonce_odr) launches KShared.
// TU1-HOST-DAG: @.str{{[.0-9]*}}.int = internal constant [{{[0-9]+}} x i8] c"_ZTS7KShared.int.{{[0-9a-f]+}}\00"
// TU1-HOST-DAG: define linkonce_odr void @_Z6launchRN4sycl3_V15queueEPfS3_S3_(
// TU1-HOST-DAG: define internal void @_Z6launchRN4sycl3_V15queueEPfS3_S3_.int(
// TU1-HOST-DAG: define internal {{.*}}@_ZN4sycl3_V16detail19getDeviceKernelInfoI7KSharedEERNS1_16DeviceKernelInfoEv.int()
// The handler's kernel name on the internal path is the copy's too (T26c):
// an internal copy of setDeviceKernelInfo<KShared>'s `__const` Info.
// TU1-NAME-DAG: @[[INFO:__const._ZN4sycl3_V17handler19setDeviceKernelInfoI7KSharedEEvPv.Info]].int = internal constant {{.*}} { ptr @[[STR:.str[.0-9]*]].int, i64 33 } }
// TU1-NAME-DAG: @[[STR]].int = internal constant [34 x i8] c"_ZTS7KShared.int.{{[0-9a-f]+}}\00"
// TU1-NAME-DAG: @[[INFO]] = private constant {{.*}} { ptr @[[STR]], i64 12 } }
// TU1-NAME-LABEL: define internal void @_ZN4sycl3_V17handler19setDeviceKernelInfoI7KSharedEEvPv.int(
// TU1-NAME: call void @llvm.memcpy{{.*}}(ptr {{.*}}, ptr {{.*}}@[[INFO]].int, i64 16
// TU1-NAME: call {{.*}}@_ZN4sycl3_V16detail19getDeviceKernelInfoI7KSharedEERNS1_16DeviceKernelInfoEv.int()

// TU2: tu2_api is an open root; its kernel copy gets the facts of its own
// body: c is distinct from x, a and b are the same allocation.
// TU2-REMARK: launch-path-internalization: kernel @_ZTS7KShared -> @_ZTS7KShared.int.[[ID2:[0-9a-f]+]] for 1 launch marker through the internalized launch path from root @_Z7tu2_apiRN4sycl3_V15queueE
// TU2-REMARK-DAG: kernel-param-noalias: kernel @_ZTS7KShared.int.[[ID2]] arg 0 (pointer 'c'): noalias (a USM allocation ({{.*}})) through the internalized launch path from root @_Z7tu2_apiRN4sycl3_V15queueE
// TU2-REMARK-DAG: kernel-param-noalias: kernel @_ZTS7KShared.int.[[ID2]] arg 1 (pointer 'a'): not stamped ('a' and 'b' use the same memory{{.*}})
// TU2-REMARK-DAG: kernel-param-noalias: kernel @_ZTS7KShared.int.[[ID2]] arg 2 (pointer 'b'): not stamped ('b' and 'a' use the same memory{{.*}})
// TU2-REMARK: launch-path-internalization: kept 1 of 1 kernel copies
// TU2-DEV: define weak_odr spir_kernel void @_ZTS7KShared(
// TU2-DEV-NOT: noalias
// TU2-DEV-SAME: {{$}}
// TU2-DEV: define weak_odr spir_kernel void @_ZTS7KShared.int.{{[0-9a-f]+}}(ptr addrspace(1) noalias {{[^,]*}}, ptr addrspace(1) noundef {{[^,]*}}, ptr addrspace(1) noundef {{[^,]*}})

// Linked: main calls TU1's internal copy and tu2_api TU2's (llvm-link
// renames the second internal `launch.int`); each launches the kernel copy
// of its own TU, named by an internal string only that TU mints. Nothing in
// either TU calls the original linkonce_odr `launch` any more (another TU's
// call would launch the unstamped KShared through it).
// LINKED-DAG: @{{.*}} = internal constant [{{[0-9]+}} x i8] c"_ZTS7KShared.int.[[ID1:[0-9a-f]+]]\00"
// LINKED-DAG: @{{.*}} = internal constant [{{[0-9]+}} x i8] c"_ZTS7KShared.int.[[ID2:[0-9a-f]+]]\00"
// LINKED-LABEL: define dso_local noundef i32 @main()
// LINKED: call void @_Z6launchRN4sycl3_V15queueEPfS3_S3_.int(
// LINKED-LABEL: define dso_local void @_Z7tu2_apiRN4sycl3_V15queueE(
// LINKED: call void @_Z6launchRN4sycl3_V15queueEPfS3_S3_.int.[[N:[0-9]+]](ptr {{.*}}%0, ptr noundef %[[X:[0-9]+]], ptr noundef %[[X]], ptr
// LINKED: define internal void @_Z6launchRN4sycl3_V15queueEPfS3_S3_.int.[[N]](
