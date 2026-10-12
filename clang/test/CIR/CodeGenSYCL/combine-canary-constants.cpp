// Constant canaries for host-derived kernel-argument constants (V6, compile
// half): a scalar capture whose value is not one compile-time constant at
// every launch must not be substituted into the device kernel. With the
// offload combine, --cir-offload-host-facts=ipo and
// -cir-offload-assume-closed-tu, the final device LLVM IR must store the
// kernel parameter, not a constant. -DCONST (one launch with 7) is the
// positive control: the kernel stores 7 + 1.
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
// Positive control: one launch with s = 7.
// REDEFINE: %{v} = const
// REDEFINE: %{flags} = -DCONST
// RUN: %{pipe}
// RUN: FileCheck %s --check-prefix=CONST --input-file=%t/const.ll
//
// Two launches of one kernel with s = 3 and s = 5.
// REDEFINE: %{v} = two
// REDEFINE: %{flags} = -DTWO
// RUN: %{pipe}
// RUN: FileCheck %s --check-prefix=NONE --input-file=%t/two.ll
//
// s = 3, then overwritten by a queue::memcpy into &s.
// REDEFINE: %{v} = model
// REDEFINE: %{flags} = -DMODEL
// RUN: %{pipe}
// RUN: FileCheck %s --check-prefix=NONE --input-file=%t/model.ll
//
// r = range(1), then r = buffer::get_range(); s = r[0].
// REDEFINE: %{v} = range
// REDEFINE: %{flags} = -DRANGE
// RUN: %{pipe}
// RUN: FileCheck %s --check-prefix=NONE --input-file=%t/range.ll
//
// s from argc.
// REDEFINE: %{v} = argv
// REDEFINE: %{flags} = -DARGV
// RUN: %{pipe}
// RUN: FileCheck %s --check-prefix=NONE --input-file=%t/argv.ll

// CONST: define {{.*}}spir_kernel void @_ZTS4Fill(
// CONST: store i32 8, ptr addrspace(1)

// NONE: define {{.*}}spir_kernel void @_ZTS4Fill(ptr addrspace(1) {{.*}}%0, i32 noundef %1)
// NONE-NOT: store i32 {{[0-9]+}}, ptr addrspace(1)
// NONE: store i32 %{{[0-9]+}}, ptr addrspace(1)
// NONE-NOT: store i32 {{[0-9]+}}, ptr addrspace(1)
// NONE: ret void

#include "combine-canary-mock.h"

using namespace sycl;
class Fill;

template <class Kn> void fill(queue &q, int *out, int s) {
  q.submit([&](handler &h) {
    h.parallel_for<Kn>(range<1>(64), [=](id<1> it) { out[it[0]] = s + 1; });
  });
  q.wait();
}

int main(int argc, char **argv) {
  queue q;
  int *out = malloc_shared<int>(64, q);
#if defined(CONST)
  fill<Fill>(q, out, 7);
#elif defined(TWO)
  fill<Fill>(q, out, 3);
  fill<Fill>(q, out, 5);
#elif defined(MODEL)
  int *src = malloc_shared<int>(1, q);
  int s = 3;
  q.memcpy(&s, src, sizeof(int)).wait();
  fill<Fill>(q, out, s);
#elif defined(RANGE)
  std::vector<float> v;
  v.resize(37);
  buffer<float, 1> b(v.data(), range<1>(37));
  range<1> r(1);
  r = b.get_range();
  int s = (int)r[0];
  fill<Fill>(q, out, s);
#elif defined(ARGV)
  fill<Fill>(q, out, argc + 10);
#endif
  free(out, q);
  return 0;
}
