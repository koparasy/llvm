// Scalar-capture constants and nd_range local sizes of kernels other TUs can
// launch (T26b). Both TUs include the "header" below:
//  - KScale: a USM lambda kernel in the in-class (inline) operator() of the
//    command-group function object Scale, with a captured scalar: one closure
//    type and one weak_odr device kernel for the program;
//  - KGeom: the lambda an inline variable holds, launched with nd_range local
//    size 16 by TU1 and 32 by TU2;
//  - ScaleF: a functor kernel with a scalar member, 2.0 in TU1, a runtime
//    value in TU2.
// Before T26b: the default (legacy) host facts specialized ScaleF by clone
// (`_ZTS6ScaleF.spec.noalias.0.spec.constprop.0`, 2.0 folded; TU2 minted
// `_ZTS6ScaleF.spec.noalias.0` too, and both repointed the marker in the
// linkonce_odr StoreLambda<ScaleF, ScaleF>), and ipo without closed-tu and
// without internalization stamped reqd_work_group_size 16 on the shared KGeom
// in place. Now a kernel other TUs can launch gets a constant or a required
// work-group size only on the kernel copy of an internalized launch path
// (K.int.<TU1 id>); TU-unique kernels (KGeomMain, a lambda in main) keep
// theirs in place; the clone path stays under -cir-offload-assume-closed-tu.
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
// DEFINE:   && %{dev} -O1 -emit-llvm -x cir %t/%{v}.%{m}.split.dev.cir -o %t/%{v}.%{m}.dev.ll
//
// RUN: %{dev} -emit-cir %s -o %t/tu1.dev.cir
// RUN: %{host} -emit-cir %s -o %t/tu1.host.cir
// RUN: %{dev} -emit-cir -DTU2 %s -o %t/tu2.dev.cir
// RUN: %{host} -emit-cir -DTU2 %s -o %t/tu2.host.cir
//
// The default (legacy host facts, internalized launch paths).
// RUN: %{merge}
// RUN: FileCheck %s --check-prefix=DEFAULT --input-file=%t/tu1.default.remarks
// RUN: FileCheck %s --check-prefix=DEFAULT-DEV --input-file=%t/tu1.default.dev.ll
//
// ipo, open, internalized.
// REDEFINE: %{m} = ipo
// REDEFINE: %{opts} = --cir-offload-host-facts=ipo
// RUN: %{merge}
// RUN: FileCheck %s --check-prefix=IPO --input-file=%t/tu1.ipo.remarks
// RUN: FileCheck %s --check-prefix=IPO-DEV --input-file=%t/tu1.ipo.dev.ll
//
// ipo, open, internalization off: nothing for the shared kernels.
// REDEFINE: %{m} = iponoint
// REDEFINE: %{opts} = --cir-offload-host-facts=ipo --cir-offload-internalize-launch-paths=false
// RUN: %{merge}
// RUN: FileCheck %s --check-prefix=NOINT --input-file=%t/tu1.iponoint.remarks
// RUN: FileCheck %s --check-prefix=NOINT-DEV --input-file=%t/tu1.iponoint.dev.ll
//
// TU2 (no main: no closed root): nothing for the shared kernels.
// REDEFINE: %{v} = tu2
// REDEFINE: %{m} = ipo
// REDEFINE: %{opts} = --cir-offload-host-facts=ipo
// RUN: %{merge}
// RUN: FileCheck %s --check-prefix=TU2-DEV --input-file=%t/tu2.ipo.dev.ll
//
// Closed-TU (legacy): the clone path; ipo + closed-TU: KGeom in place.
// REDEFINE: %{v} = tu1
// REDEFINE: %{m} = closed
// REDEFINE: %{opts} = -cir-offload-assume-closed-tu
// RUN: %{merge}
// RUN: FileCheck %s --check-prefix=CLOSED-DEV --input-file=%t/tu1.closed.dev.ll
// REDEFINE: %{m} = ipoclosed
// REDEFINE: %{opts} = --cir-offload-host-facts=ipo -cir-offload-assume-closed-tu
// RUN: %{merge}
// RUN: FileCheck %s --check-prefix=IPOCLOSED-DEV --input-file=%t/tu1.ipoclosed.dev.ll

#include "combine-canary-mock.h"
using namespace sycl;

class KScale;
class KGeom;
class KGeomMain;

// "Header": shared by every TU. (Scale is bigger than 16 bytes: passed in
// memory, not coerced into registers.)
struct Scale {
  float *p;
  float f;
  unsigned long tag;
  void operator()(handler &h) const {
    float *d = p;
    float s = f;
    h.parallel_for<KScale>(range<1>(64), [=](id<1> i) { d[i[0]] *= s; });
  }
};
inline auto zeroKernel = [](nd_item<1> it) {};
struct ScaleF {
  float *p;
  float f;
  void operator()(id<1> i) const { p[i[0]] *= f; }
};

#ifndef TU2
int main() {
  queue q;
  float *a = malloc_device<float>(64, q);
  q.submit(Scale{a, 2.0f, 0});
  q.submit([&](handler &h) {
    h.parallel_for<KGeom>(nd_range<1>(range<1>(64), range<1>(16)), zeroKernel);
  });
  q.submit([&](handler &h) {
    h.parallel_for<ScaleF>(range<1>(64), ScaleF{a, 2.0f});
  });
  // TU-unique: a lambda in main.
  q.submit([&](handler &h) {
    h.parallel_for<KGeomMain>(nd_range<1>(range<1>(64), range<1>(16)),
                              [=](nd_item<1> it) { a[it.get_global_id(0)] = 0; });
  });
  return 0;
}
#else
void tu2_api(queue &q, float f) {
  float *x = malloc_device<float>(64, q);
  q.submit(Scale{x, f, 0});
  q.submit([&](handler &h) {
    h.parallel_for<KGeom>(nd_range<1>(range<1>(64), range<1>(32)), zeroKernel);
  });
  q.submit([&](handler &h) { h.parallel_for<ScaleF>(range<1>(64), ScaleF{x, f}); });
}
#endif

// DEFAULT: remark: kernel-arg-constant: kernel @_ZTS6ScaleF.int.{{[0-9a-f]+}} arg 1 ('f'): #cir.fp<2.000000e+00> : !cir.float through the internalized launch path from root @main
// DEFAULT-NOT: .spec.
// DEFAULT-DEV: define weak_odr spir_kernel void @_ZTS6ScaleF(
// DEFAULT-DEV-NOT: define
// DEFAULT-DEV: fmul float %1,
// DEFAULT-DEV: define weak_odr spir_kernel void @_ZTS6ScaleF.int.{{[0-9a-f]+}}(
// DEFAULT-DEV-NOT: define
// DEFAULT-DEV: fmul float %{{[0-9]+}}, 2.000000e+00
// DEFAULT-DEV-NOT: .spec.

// IPO-DAG: remark: reqd-work-group-size: kernel @_ZTS5KGeom.int.{{[0-9a-f]+}} [16] through the internalized launch path from root @main
// IPO-DAG: remark: kernel-arg-constant: kernel @_ZTS6KScale.int.{{[0-9a-f]+}} arg 1 ('s'): #cir.fp<2.000000e+00> : !cir.float through the internalized launch path from root @main
// IPO-DAG: remark: kernel-arg-constant: kernel @_ZTS6ScaleF.int.{{[0-9a-f]+}} arg 1 ('f'): #cir.fp<2.000000e+00> : !cir.float through the internalized launch path from root @main
// IPO-DEV-DAG: define weak_odr spir_kernel void @_ZTS5KGeom() local_unnamed_addr #{{[0-9]+}} comdat {
// IPO-DEV-DAG: define weak_odr spir_kernel void @_ZTS5KGeom.int.{{[0-9a-f]+}}() {{.*}}!reqd_work_group_size
// IPO-DEV-DAG: define weak_odr spir_kernel void @_ZTS9KGeomMain({{.*}}!reqd_work_group_size

// NOINT-DAG: remark: reqd-work-group-size: kernel @_ZTS5KGeom [16] not stamped (the kernel is visible to other TUs; facts this TU derives hold only through an internalized launch path, and launch-path internalization is off)
// NOINT-DAG: remark: kernel-arg-constant: kernel @_ZTS6ScaleF arg 1 ('f'): #cir.fp<2.000000e+00> : !cir.float not propagated (the kernel is visible to other TUs; facts this TU derives hold only through an internalized launch path, and launch-path internalization is off)
// NOINT-DEV-DAG: define weak_odr spir_kernel void @_ZTS5KGeom() local_unnamed_addr #{{[0-9]+}} comdat {
// NOINT-DEV-DAG: define weak_odr spir_kernel void @_ZTS9KGeomMain({{.*}}!reqd_work_group_size
// NOINT-DEV-NOT: .spec.
// NOINT-DEV-NOT: 2.000000e+00

// TU2-DEV-NOT: reqd_work_group_size
// TU2-DEV-NOT: 2.000000e+00
// TU2-DEV-NOT: .int.

// CLOSED-DEV: define weak_odr spir_kernel void @_ZTS6ScaleF.spec.noalias.0.spec.constprop.0(
// CLOSED-DEV-NOT: define
// CLOSED-DEV: fmul float %{{[0-9]+}}, 2.000000e+00

// IPOCLOSED-DEV: define weak_odr spir_kernel void @_ZTS5KGeom() {{.*}}!reqd_work_group_size
