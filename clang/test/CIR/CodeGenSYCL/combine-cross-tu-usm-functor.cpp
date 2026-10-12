// A USM functor kernel defined in a header, across two TUs (T26b). The kernel
// AddF is a named functor: one weak_odr device kernel for the program, and its
// host launch shims (handler::parallel_for<AddF, AddF>, StoreLambda<AddF,
// AddF>, getDeviceKernelInfo<AddF>) are linkonce_odr in every TU. TU1's main
// launches it with two distinct malloc_device pointers, TU2's tu2_api with the
// same pointer twice. Before T26b the default (legacy) host facts specialized
// AddF by clone: TU1 minted `_ZTS4AddF.spec.noalias.0` (a name any TU mints)
// and repointed the marker inside the linkonce_odr StoreLambda<AddF, AddF> at
// a linkonce_odr getter copy naming the clone, so TU2's launch, bound by the
// linker to TU1's StoreLambda, launched the stamped clone with aliasing
// pointers. Now the clone path runs only under -cir-offload-assume-closed-tu;
// the default internalizes TU1's path from main and stamps AddF.int.<TU1 id>,
// launched only by TU1's internal copies. The host modules of the two TUs are
// linked (llvm-link; nothing is run).
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
// REDEFINE: %{v} = tu2
// RUN: %{merge}
// RUN: FileCheck %s --check-prefix=TU2-DEV --input-file=%t/tu2.default.dev.ll
// RUN: llvm-link -S %t/tu1.default.host.ll %t/tu2.default.host.ll -o %t/linked.host.ll
// RUN: FileCheck %s --check-prefix=LINKED --input-file=%t/linked.host.ll
// RUN: FileCheck %s --check-prefix=LINKED-INT --input-file=%t/linked.host.ll
//
// Internalization off: no fact, no clone.
// REDEFINE: %{v} = tu1
// REDEFINE: %{m} = noint
// REDEFINE: %{opts} = --cir-offload-internalize-launch-paths=false
// RUN: %{merge}
// RUN: FileCheck %s --check-prefix=NOINT --input-file=%t/tu1.noint.remarks
// RUN: FileCheck %s --check-prefix=NOINT-DEV --input-file=%t/tu1.noint.dev.ll
//
// Closed-TU: the clone path.
// REDEFINE: %{m} = closed
// REDEFINE: %{opts} = -cir-offload-assume-closed-tu
// RUN: %{merge}
// RUN: FileCheck %s --check-prefix=CLOSED --input-file=%t/tu1.closed.dev.ll
// RUN: FileCheck %s --check-prefix=CLOSED-HOST --input-file=%t/tu1.closed.host.ll
// RUN: not grep kernel-name-redirect: %t/tu1.default.remarks %t/tu2.default.remarks \
// RUN:   %t/tu1.noint.remarks %t/tu1.closed.remarks
//
// The real headers' handler::setDeviceKernelInfo (without
// COMBINE_MOCK_HOST_KERNEL_RESET): the handler's kernel name comes from a
// `__const` copy of the CompileTimeKernelInfo, which the clone (ipo +
// closed-TU) and the kernel copy (ipo, internalized) redirect too (T26c).
// DEFINE: %{rdev} = %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// DEFINE:   -aux-triple x86_64-unknown-linux-gnu -fclangir -I%S/Inputs
// DEFINE: %{rhost} = %clang_cc1 -std=c++17 -fsycl-is-host -triple x86_64-unknown-linux-gnu \
// DEFINE:   -aux-triple spir64-unknown-unknown -fclangir -I%S/Inputs
// DEFINE: %{rmerge} = cir-offload-merge -combine %{targets} -input=%t/real.host.cir \
// DEFINE:     -input=%t/real.dev.cir %{opts} -output=%t/real.%{m}.cir > %t/real.%{m}.remarks 2>&1 \
// DEFINE:   && cir-offload-merge -split %{targets} -input=%t/real.%{m}.cir \
// DEFINE:     -output=%t/real.%{m}.split.host.cir -output=%t/real.%{m}.split.dev.cir \
// DEFINE:   && %{rhost} -emit-llvm -x cir %t/real.%{m}.split.host.cir -o %t/real.%{m}.host.ll
// RUN: %{rdev} -emit-cir %s -o %t/real.dev.cir
// RUN: %{rhost} -emit-cir %s -o %t/real.host.cir
// REDEFINE: %{m} = ipoclosed
// REDEFINE: %{opts} = --cir-offload-host-facts=ipo -cir-offload-assume-closed-tu
// RUN: %{rmerge}
// RUN: FileCheck %s --check-prefix=REAL-CLOSED --input-file=%t/real.ipoclosed.host.ll
// REDEFINE: %{m} = ipoopen
// REDEFINE: %{opts} = --cir-offload-host-facts=ipo
// RUN: %{rmerge}
// RUN: FileCheck %s --check-prefix=REAL-OPEN --input-file=%t/real.ipoopen.host.ll
// RUN: not grep kernel-name-redirect: %t/real.ipoclosed.remarks %t/real.ipoopen.remarks

#include "combine-canary-mock.h"
using namespace sycl;

// "Header": shared by every TU.
struct AddF {
  float *a, *b;
  void operator()(id<1> i) const { a[i[0]] += b[i[0]]; }
};

#ifndef TU2
int main() {
  queue q;
  float *a = malloc_device<float>(64, q);
  float *b = malloc_device<float>(64, q);
  q.submit([&](handler &h) { h.parallel_for<AddF>(range<1>(64), AddF{a, b}); });
  return 0;
}
#else
// TU2: a and b are the same allocation.
void tu2_api(queue &q) {
  float *x = malloc_device<float>(64, q);
  q.submit([&](handler &h) { h.parallel_for<AddF>(range<1>(64), AddF{x, x}); });
}
#endif

// TU1-REMARK-DAG: remark: launch-path-internalization: kernel @_ZTS4AddF -> @_ZTS4AddF.int.[[ID:[0-9a-f]+]] for 1 launch marker through the internalized launch path from root @main
// TU1-REMARK-DAG: remark: kernel-param-noalias: kernel @_ZTS4AddF arg 0 (pointer 'a'): not stamped
// TU1-REMARK-DAG: remark: kernel-param-noalias: kernel @_ZTS4AddF.int.[[ID]] arg 0 (pointer 'a'): noalias (a USM allocation) through the internalized launch path from root @main
// TU1-REMARK-DAG: remark: kernel-param-noalias: kernel @_ZTS4AddF.int.[[ID]] arg 1 (pointer 'b'): noalias (a USM allocation) through the internalized launch path from root @main
// TU1-REMARK-NOT: .spec.

// TU1-DEV: define weak_odr spir_kernel void @_ZTS4AddF(
// TU1-DEV-NOT: noalias
// TU1-DEV-SAME: {{$}}
// TU1-DEV: define weak_odr spir_kernel void @_ZTS4AddF.int.{{[0-9a-f]+}}(ptr addrspace(1) noalias {{[^,]*}}, ptr addrspace(1) noalias
// TU1-DEV-NOT: .spec.

// TU2-DEV: define weak_odr spir_kernel void @_ZTS4AddF(
// TU2-DEV-NOT: noalias
// TU2-DEV-SAME: {{$}}
// TU2-DEV-NOT: .spec.

// Linked: the one linkonce_odr StoreLambda<AddF, AddF> (TU2's launch binds
// to it) names the original kernel; only TU1's internal copy names the copy.
// LINKED-LABEL: define linkonce_odr void @_ZN4sycl3_V17handler11StoreLambdaI4AddFS3_EEvRKT0_(
// LINKED-NOT: define
// LINKED: call {{.*}}@_ZN4sycl3_V16detail19getDeviceKernelInfoI4AddFEERNS1_16DeviceKernelInfoEv()
// LINKED-INT-LABEL: define internal void @_ZN4sycl3_V17handler11StoreLambdaI4AddFS3_EEvRKT0_.int(
// LINKED-INT-NOT: define
// The handler's kernel name too (T26c).
// LINKED-INT: call {{.*}}@_ZN4sycl3_V16detail11string_viewaSERKS2_({{.*}}@_ZN4sycl3_V16detail21CompileTimeKernelInfoI4AddFEE.int)
// LINKED-INT: call {{.*}}@_ZN4sycl3_V16detail19getDeviceKernelInfoI4AddFEERNS1_16DeviceKernelInfoEv.int()

// NOINT: remark: kernel-param-noalias: kernel @_ZTS4AddF arg 0 (pointer 'a'): not stamped (the kernel is visible to other TUs; facts this TU derives hold only through an internalized launch path, and launch-path internalization is off)
// NOINT-NOT: .spec.
// NOINT-DEV: define weak_odr spir_kernel void @_ZTS4AddF(
// NOINT-DEV-NOT: noalias
// NOINT-DEV-SAME: {{$}}
// NOINT-DEV-NOT: AddF.

// CLOSED: define weak_odr spir_kernel void @_ZTS4AddF(
// CLOSED-NOT: noalias
// CLOSED-SAME: {{$}}
// CLOSED: define weak_odr spir_kernel void @_ZTS4AddF.spec.noalias.0(ptr addrspace(1) noalias {{[^,]*}}, ptr addrspace(1) noalias
// The clone's name string and CompileTimeKernelInfo; StoreLambda<AddF, AddF>
// names the clone in the handler and through the getter (T26c).
// CLOSED-HOST: @[[STR:.str_[0-9]+]] = private constant [25 x i8] c"_ZTS4AddF.spec.noalias.0\00"
// CLOSED-HOST: @[[CTKI:_ZN4sycl3_V16detail21CompileTimeKernelInfoI4AddFEE_[0-9]+]] = {{.*}} { ptr @[[STR]], i64 24 } }
// CLOSED-HOST-LABEL: define linkonce_odr void @_ZN4sycl3_V17handler11StoreLambdaI4AddFS3_EEvRKT0_(
// CLOSED-HOST: call {{.*}}@_ZN4sycl3_V16detail11string_viewaSERKS2_({{.*}}@[[CTKI]])
// CLOSED-HOST: call {{.*}}@_ZN4sycl3_V16detail19getDeviceKernelInfoI4AddFEERNS1_16DeviceKernelInfoEv_{{[0-9]+}}()

// The real headers' shape, clone path: setDeviceKernelInfo<AddF> (whose
// marker the clone path repoints in place) copies the handler's kernel name
// from an internal copy of its `__const` Info naming the clone's string.
// REAL-CLOSED-DAG: @[[INFO:__const._ZN4sycl3_V17handler19setDeviceKernelInfoI4AddFEEvPv.Info_[0-9]+]] = internal constant {{.*}} { ptr @[[STR:.str_[0-9]+]], i64 24 } }
// REAL-CLOSED-DAG: @[[STR]] = private constant [25 x i8] c"_ZTS4AddF.spec.noalias.0\00"
// REAL-CLOSED-LABEL: define linkonce_odr void @_ZN4sycl3_V17handler19setDeviceKernelInfoI4AddFEEvPv(
// REAL-CLOSED: call void @llvm.memcpy{{.*}}(ptr {{.*}}, ptr {{.*}}@[[INFO]], i64 16
// REAL-CLOSED: call {{.*}}@_ZN4sycl3_V16detail19getDeviceKernelInfoI4AddFEERNS1_16DeviceKernelInfoEv_{{[0-9]+}}()

// The real headers' shape, internalized path: the internal copy of
// setDeviceKernelInfo<AddF> names the kernel copy; the original, the
// original kernel.
// REAL-OPEN-DAG: @__const._ZN4sycl3_V17handler19setDeviceKernelInfoI4AddFEEvPv.Info = private constant {{.*}} { ptr @.str, i64 9 } }
// REAL-OPEN-DAG: @__const._ZN4sycl3_V17handler19setDeviceKernelInfoI4AddFEEvPv.Info.int = internal constant {{.*}} { ptr @.str.int, i64 30 } }
// REAL-OPEN-DAG: @.str.int = internal constant [31 x i8] c"_ZTS4AddF.int.{{[0-9a-f]+}}\00"
// REAL-OPEN-LABEL: define linkonce_odr void @_ZN4sycl3_V17handler19setDeviceKernelInfoI4AddFEEvPv(
// REAL-OPEN: call void @llvm.memcpy{{.*}}(ptr {{.*}}, ptr {{.*}}@__const._ZN4sycl3_V17handler19setDeviceKernelInfoI4AddFEEvPv.Info, i64 16
// REAL-OPEN: call {{.*}}@_ZN4sycl3_V16detail19getDeviceKernelInfoI4AddFEERNS1_16DeviceKernelInfoEv()
// REAL-OPEN-LABEL: define internal void @_ZN4sycl3_V17handler19setDeviceKernelInfoI4AddFEEvPv.int(
// REAL-OPEN: call void @llvm.memcpy{{.*}}(ptr {{.*}}, ptr {{.*}}@__const._ZN4sycl3_V17handler19setDeviceKernelInfoI4AddFEEvPv.Info.int, i64 16
// REAL-OPEN: call {{.*}}@_ZN4sycl3_V16detail19getDeviceKernelInfoI4AddFEERNS1_16DeviceKernelInfoEv.int()
