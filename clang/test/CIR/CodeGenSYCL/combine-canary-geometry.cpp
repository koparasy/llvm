// Geometry canaries for host-derived launch geometry (V6, compile half): a
// kernel whose local size is not one compile-time constant at every launch
// must not get reqd_work_group_size. With the offload combine,
// --cir-offload-host-facts=ipo and -cir-offload-assume-closed-tu, the final
// device LLVM IR of such a kernel carries no !reqd_work_group_size. -DCONST
// (one nd_range(64, 16) launch) is the positive control.
//
// RUN: rm -rf %t && mkdir -p %t
// DEFINE: %{v} = unset
// DEFINE: %{flags} = -DUNSET
// DEFINE: %{dev} = %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// DEFINE:   -aux-triple x86_64-unknown-linux-gnu -fclangir -I%S/Inputs
// DEFINE: %{host} = %clang_cc1 -std=c++17 -fsycl-is-host -triple x86_64-unknown-linux-gnu \
// DEFINE:   -aux-triple spir64-unknown-unknown -fclangir -I%S/Inputs
// DEFINE: %{targets} = -targets=host-x86_64-unknown-linux-gnu,sycl-spir64-unknown-unknown
// DEFINE: %{pipe} = %{dev} -emit-cir %{flags} %s -o %t/%{v}.dev.cir \
// DEFINE:   && %{host} -emit-cir %{flags} %s -o %t/%{v}.host.cir \
// DEFINE:   && cir-offload-merge -combine --remarks-filter="cir-offload.*" %{targets} -input=%t/%{v}.host.cir \
// DEFINE:        -input=%t/%{v}.dev.cir --cir-offload-host-facts=ipo \
// DEFINE:        -cir-offload-assume-closed-tu -output=%t/%{v}.cir \
// DEFINE:   && cir-offload-merge -split %{targets} -input=%t/%{v}.cir \
// DEFINE:        -output=%t/%{v}.split.host.cir -output=%t/%{v}.split.dev.cir \
// DEFINE:   && %{dev} -O1 -emit-llvm -x cir %t/%{v}.split.dev.cir -o %t/%{v}.ll
//
// Positive control: one launch with local size 16.
// REDEFINE: %{v} = const
// REDEFINE: %{flags} = -DCONST
// RUN: %{pipe}
// RUN: FileCheck %s --check-prefix=CONST --input-file=%t/const.ll
//
// One kernel launched with local sizes 16 and 32.
// REDEFINE: %{v} = two
// REDEFINE: %{flags} = -DTWO
// RUN: %{pipe}
// RUN: FileCheck %s --check-prefix=NONE --input-file=%t/two.ll
//
// The local size from argc.
// REDEFINE: %{v} = argv
// REDEFINE: %{flags} = -DARGV
// RUN: %{pipe}
// RUN: FileCheck %s --check-prefix=NONE --input-file=%t/argv.ll

// CONST: define {{.*}}spir_kernel void @_ZTS3Geo({{.*}} !reqd_work_group_size ![[WG:[0-9]+]]
// CONST: ![[WG]] = !{i32 16, i32 1, i32 1}

// NONE: define {{.*}}spir_kernel void @_ZTS3Geo(
// NONE-NOT: reqd_work_group_size
// NONE-SAME: {{$}}
// NONE-NOT: reqd_work_group_size

#include "combine-canary-mock.h"

using namespace sycl;
class Geo;

template <class Kn> void geo(queue &q, int *out, unsigned long lsz) {
  q.submit([&](handler &h) {
    h.parallel_for<Kn>(nd_range<1>(range<1>(64), range<1>(lsz)),
                       [=](nd_item<1> it) { out[it.get_global_id(0)] = 1; });
  });
  q.wait();
}

int main(int argc, char **argv) {
  queue q;
  int *out = malloc_shared<int>(64, q);
#if defined(CONST)
  geo<Geo>(q, out, 16);
#elif defined(TWO)
  geo<Geo>(q, out, 16);
  geo<Geo>(q, out, 32);
#elif defined(ARGV)
  geo<Geo>(q, out, argc > 1 ? 32 : 16);
#endif
  free(out, q);
  return 0;
}
