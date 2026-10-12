// The kernel-name reads of an internalized launch path (T26c). The SYCL
// runtime names a launch twice, by the DeviceKernelInfo the kernel.info
// getter returns and by the name handler::setDeviceKernelInfo stores in the
// handler, and handler::finalize() asserts that the two agree. So with the
// markers of a kernel copy K.int.<id>, every other reference of the SYCL
// launch implementation on the paths to them to K's name (its string, its
// CompileTimeKernelInfo, the getter, or a constant that carries one) names
// the copy. Here sycl::detail::two<K1, K2> launches both kernels and then
// reads K2's name:
//  - K1's captures are three distinct USM allocations: its copy gets
//    noalias and is kept;
//  - K2's captures `a` and `d` are one allocation: its copy gets nothing and
//    the prune pass drops it, restoring the path to K2 in the copy of
//    `two`, which K1 keeps: its K2 launch and its read of K2's name name K2
//    again (not the dropped copy);
//  - -DNAME_VARIABLE: `two` also reads K1's name through a variable (not a
//    constant: it cannot be copied), which the post-condition reports.
//
// RUN: rm -rf %t && mkdir -p %t
// DEFINE: %{dev} = %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// DEFINE:   -aux-triple x86_64-unknown-linux-gnu -fclangir -I%S/Inputs
// DEFINE: %{host} = %clang_cc1 -std=c++17 -fsycl-is-host -triple x86_64-unknown-linux-gnu \
// DEFINE:   -aux-triple spir64-unknown-unknown -fclangir -I%S/Inputs
// DEFINE: %{targets} = -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown
//
// RUN: %{dev} -emit-cir %s -o %t/dev.cir
// RUN: %{host} -emit-cir %s -o %t/host.cir
// RUN: cir-offload-merge -combine --remarks-filter="cir-offload.*" %{targets} -input=%t/host.cir -input=%t/dev.cir \
// RUN:   --cir-offload-host-facts=ipo -output=%t/out.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REMARK --implicit-check-not=kernel-name-redirect:
// RUN: cir-offload-merge -split %{targets} -input=%t/out.cir -output=%t/split.host.cir \
// RUN:   -output=%t/split.dev.cir
// RUN: %{host} -emit-llvm -x cir %t/split.host.cir -o %t/host.ll
// RUN: FileCheck %s --check-prefix=HOST --input-file=%t/host.ll
//
// RUN: %{dev} -emit-cir -DNAME_VARIABLE %s -o %t/var.dev.cir
// RUN: %{host} -emit-cir -DNAME_VARIABLE %s -o %t/var.host.cir
// RUN: cir-offload-merge -combine --remarks-filter="cir-offload.*" %{targets} -input=%t/var.host.cir -input=%t/var.dev.cir \
// RUN:   --cir-offload-host-facts=ipo -output=%t/var.cir 2>&1 \
// RUN:   | FileCheck %s --check-prefix=VAR

#include "combine-canary-mock.h"
using namespace sycl;

class K1;
class K2;

namespace sycl {
inline namespace _V1 {
namespace detail {
inline string_view LastName;
#ifdef NAME_VARIABLE
template <class K>
inline const char *NameVar = CompileTimeKernelInfo<K>.Name.str;
inline const char *LastPtr;
#endif
template <class A, class B>
inline void two(queue &q, float *a, float *b, float *c) {
  q.submit([&](handler &h) {
    h.parallel_for<A>(range<1>(64),
                      [=](id<1> i) { c[i[0]] = a[i[0]] + b[i[0]]; });
  });
  float *d = a;
  q.submit([&](handler &h) {
    h.parallel_for<B>(range<1>(64), [=](id<1> i) { a[i[0]] = d[i[0]] + 1; });
  });
  LastName = CompileTimeKernelInfo<B>.Name;
#ifdef NAME_VARIABLE
  LastPtr = NameVar<A>;
#endif
}
} // namespace detail
} // namespace _V1
} // namespace sycl

int main() {
  queue q;
  float *a = malloc_shared<float>(64, q);
  float *b = malloc_shared<float>(64, q);
  float *c = malloc_shared<float>(64, q);
  detail::two<K1, K2>(q, a, b, c);
  return 0;
}

// REMARK-DAG: launch-path-internalization: kernel @_ZTS2K1 -> @_ZTS2K1.int.{{[0-9a-f]+}}
// REMARK-DAG: launch-path-internalization: kernel @_ZTS2K2 -> @_ZTS2K2.int.{{[0-9a-f]+}}
// REMARK-DAG: kernel-param-noalias: kernel @_ZTS2K1.int.{{[0-9a-f]+}} arg 0 (pointer 'c'): noalias
// REMARK-DAG: kernel-param-noalias: kernel @_ZTS2K2.int.{{[0-9a-f]+}} arg 0 (pointer 'a'): not stamped
// REMARK: launch-path-internalization: kept 1 of 2 kernel copies

// K1's names on its path are the copy's; the copy of `two` launches K2 and
// reads K2's name as the original does.
// HOST-NOT: K2EE.int
// HOST-LABEL: define internal void @_ZN4sycl3_V16detail3twoI2K12K2EEvRNS0_5queueEPfS7_S7_.int(
// HOST: call void @_ZN4sycl3_V15queue6submitIZNS0_6detail3twoI2K12K2EEvRS1_PfS8_S8_EUlRNS0_7handlerEE_EENS0_5eventET_.int(
// HOST: call void @_ZN4sycl3_V15queue6submitIZNS0_6detail3twoI2K12K2EEvRS1_PfS8_S8_EUlRNS0_7handlerEE0_EENS0_5eventET_(
// HOST: call {{.*}}@_ZN4sycl3_V16detail11string_viewaSERKS2_({{.*}}@_ZN4sycl3_V16detail21CompileTimeKernelInfoI2K2EE)
// HOST-LABEL: define internal void @_ZN4sycl3_V17handler19setDeviceKernelInfoI2K1EEvPv.int(
// HOST: call void @llvm.memcpy{{.*}}(ptr {{.*}}, ptr {{.*}}@__const._ZN4sycl3_V17handler19setDeviceKernelInfoI2K1EEvPv.Info.int, i64 16
// HOST: call {{.*}}@_ZN4sycl3_V16detail19getDeviceKernelInfoI2K1EERNS1_16DeviceKernelInfoEv.int()
// HOST-NOT: K2EE.int

// VAR: remark: kernel-name-redirect: @_ZN4sycl3_V16detail3twoI2K12K2EEvRNS0_5queueEPfS7_S7_.int on a launch path of @_ZTS2K1.int.{{[0-9a-f]+}} still names the original kernel @_ZTS2K1; refers to @_ZN4sycl3_V16detail7NameVarI2K1EE
