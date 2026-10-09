// Aliasing canaries for host-derived noalias (V6, compile half): kernels whose
// C and A parameters really alias, in every way the host can make them alias.
// With the offload combine, --cir-offload-host-facts=ipo and
// -cir-offload-assume-closed-tu, the final device LLVM IR must not carry
// noalias on C or A (a wrong noalias lets the k-loop store of C sink out of
// the loop and changes the result); only the distinct B is noalias. The
// default build (three distinct buffers) is the positive control: A, B and C
// are noalias.
//
// The runnable canaries (real headers, self-checking, with the
// kernel_args_restrict calibration variants) are in the T21 report.
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
// DEFINE:   && cir-offload-merge -combine %{targets} -input=%t/%{v}.host.cir \
// DEFINE:        -input=%t/%{v}.dev.cir --cir-offload-host-facts=ipo \
// DEFINE:        -cir-offload-assume-closed-tu -output=%t/%{v}.cir > %t/%{v}.remarks 2>&1 \
// DEFINE:   && cir-offload-merge -split %{targets} -input=%t/%{v}.cir \
// DEFINE:        -output=%t/%{v}.split.host.cir -output=%t/%{v}.split.dev.cir \
// DEFINE:   && %{dev} -O1 -emit-llvm -x cir %t/%{v}.split.dev.cir -o %t/%{v}.ll
//
// Positive control: three distinct buffers.
// REDEFINE: %{v} = distinct
// REDEFINE: %{flags} = -DDISTINCT
// RUN: %{pipe}
// RUN: FileCheck %s --check-prefix=DISTINCT --input-file=%t/distinct.ll
//
// The same buffer accessed twice.
// REDEFINE: %{v} = same
// REDEFINE: %{flags} = -DSAME
// RUN: %{pipe}
// RUN: FileCheck %s --check-prefix=ALIAS --input-file=%t/same.ll
// RUN: FileCheck %s --check-prefix=ALIAS-REMARK --input-file=%t/same.remarks
//
// b2 = b1 handle copy.
// REDEFINE: %{v} = copy
// REDEFINE: %{flags} = -DCOPY
// RUN: %{pipe}
// RUN: FileCheck %s --check-prefix=ALIAS --input-file=%t/copy.ll
// RUN: FileCheck %s --check-prefix=ALIAS-REMARK --input-file=%t/copy.remarks
//
// A shared_ptr copy between two members (C_buffer = A_buffer).
// REDEFINE: %{v} = member
// REDEFINE: %{flags} = -DMEMBER
// RUN: %{pipe}
// RUN: FileCheck %s --check-prefix=ALIAS --input-file=%t/member.ll
// RUN: FileCheck %s --check-prefix=ALIAS-REMARK --input-file=%t/member.remarks
//
// A sub-buffer overlapping its parent.
// REDEFINE: %{v} = sub
// REDEFINE: %{flags} = -DSUB
// RUN: %{pipe}
// RUN: FileCheck %s --check-prefix=ALIAS --input-file=%t/sub.ll
// RUN: FileCheck %s --check-prefix=ALIAS-REMARK --input-file=%t/sub.remarks
//
// Two sub-buffers of one parent whose ranges overlap ([0, NK) and
// [NK/2, 3NK/2)): the ranges are compared, and they overlap.
// REDEFINE: %{v} = subovl
// REDEFINE: %{flags} = -DSUBOVL
// RUN: %{pipe}
// RUN: FileCheck %s --check-prefix=ALIAS --input-file=%t/subovl.ll
// RUN: FileCheck %s --check-prefix=ALIAS-REMARK --input-file=%t/subovl.remarks
//
// Positive control for the ranges: two disjoint sub-buffers ([0, NK) and
// [NK, 2NK)) of one parent are distinct memory.
// REDEFINE: %{v} = subdis
// REDEFINE: %{flags} = -DSUBDIS
// RUN: %{pipe}
// RUN: FileCheck %s --check-prefix=DISTINCT --input-file=%t/subdis.ll
//
// buffer(host_ptr) over one vector at two overlapping offsets.
// REDEFINE: %{v} = hostptr
// REDEFINE: %{flags} = -DHOSTPTR
// RUN: %{pipe}
// RUN: FileCheck %s --check-prefix=ALIAS --input-file=%t/hostptr.ll
// RUN: FileCheck %s --check-prefix=ALIAS-REMARK --input-file=%t/hostptr.remarks
//
// One USM pointer passed twice.
// REDEFINE: %{v} = usm2
// REDEFINE: %{flags} = -DUSM2
// RUN: %{pipe}
// RUN: FileCheck %s --check-prefix=USM --input-file=%t/usm2.ll
// RUN: FileCheck %s --check-prefix=USM-REMARK --input-file=%t/usm2.remarks
//
// A USM host pointer behind a buffer, plus the same pointer as an argument.
// REDEFINE: %{v} = usmbuf
// REDEFINE: %{flags} = -DUSMBUF
// RUN: %{pipe}
// RUN: FileCheck %s --check-prefix=USMBUF --input-file=%t/usmbuf.ll
// RUN: FileCheck %s --check-prefix=USM-REMARK --input-file=%t/usmbuf.remarks

#include "combine-canary-mock.h"

constexpr unsigned long N = 64, K = 16, NK = N * K;

// Item i owns [i*K, i*K + K): C[i*K + K/2] += A[i*K + k] * B[k]. When C is A,
// iteration K/2 must read what the earlier iterations stored.
#define SINK_BODY                                                              \
  [=](sycl::id<1> it) {                                                        \
    unsigned long i = it[0];                                                   \
    for (unsigned long k = 0; k < K; ++k)                                      \
      C[i * K + K / 2] += A[i * K + k] * B[k];                                 \
  }

class Sink;
using namespace sycl;
using Buf = buffer<float, 1>;

#ifdef MEMBER
class Bench {
public:
  explicit Bench(queue &q) : q(q) {}
  void setup() {
    a.resize(NK);
    b.resize(K);
    A_buffer = std::make_shared<Buf>(a.data(), range<1>(NK));
    B_buffer = std::make_shared<Buf>(b.data(), range<1>(K));
    C_buffer = A_buffer;
  }
  void run() {
    q.submit([&](handler &h) {
      auto A = A_buffer->get_access<access::mode::read>(h);
      auto B = B_buffer->get_access<access::mode::read>(h);
      auto C = C_buffer->get_access<access::mode::read_write>(h);
      h.parallel_for<Sink>(range<1>(N), SINK_BODY);
    });
  }

private:
  queue &q;
  std::vector<float> a, b;
  std::shared_ptr<Buf> A_buffer, B_buffer, C_buffer;
};
#endif

int main() {
  queue q;
#if defined(MEMBER)
  Bench bench(q);
  bench.setup();
  bench.run();
#elif defined(USM2) || defined(USMBUF)
  float *bb = malloc_shared<float>(K, q);
#ifdef USM2
  float *p = malloc_shared<float>(NK, q);
#else
  float *p = malloc_host<float>(NK, q);
  Buf bA(p, range<1>(NK));
#endif
  q.submit([&](handler &h) {
#ifdef USM2
    float *A = p;
#else
    auto A = bA.get_access<access::mode::read>(h);
#endif
    float *C = p, *B = bb;
    h.parallel_for<Sink>(range<1>(N), SINK_BODY);
  });
  q.wait();
  free(p, q);
  free(bb, q);
#else
  std::vector<float> va, vb, vc;
  va.resize(2 * NK);
  vb.resize(K);
  vc.resize(NK);
  Buf bB(vb.data(), range<1>(K));
#if defined(SAME)
  Buf bA(va.data(), range<1>(NK));
  Buf &bC = bA;
#elif defined(COPY)
  Buf bA(va.data(), range<1>(NK));
  Buf bC = bA;
#elif defined(SUB)
  Buf bA(va.data(), range<1>(2 * NK));
  Buf bC(bA, id<1>(0), range<1>(NK));
#elif defined(SUBOVL) || defined(SUBDIS)
  Buf bP(va.data(), range<1>(2 * NK));
#ifdef SUBOVL
  Buf bA(bP, id<1>(0), range<1>(NK));
  Buf bC(bP, id<1>(NK / 2), range<1>(NK));
#else
  Buf bA(bP, id<1>(0), range<1>(NK));
  Buf bC(bP, id<1>(NK), range<1>(NK));
#endif
#elif defined(HOSTPTR)
  Buf bA(va.data(), range<1>(NK));
  Buf bC(va.data() + NK / 2, range<1>(NK));
#else
  Buf bA(va.data(), range<1>(NK));
  Buf bC(vc.data(), range<1>(NK));
#endif
  q.submit([&](handler &h) {
    auto A = bA.get_access<access::mode::read>(h);
    auto B = bB.get_access<access::mode::read>(h);
    auto C = bC.get_access<access::mode::read_write>(h);
    h.parallel_for<Sink>(range<1>(N), SINK_BODY);
  });
#endif
  return 0;
}

// The kernel captures C, A, B in this order (first use): accessor C is
// argument 0, A argument 4, B argument 8.
// DISTINCT: define {{.*}}spir_kernel void @_ZTS4Sink(ptr addrspace(1) noalias noundef align 4 %0,
// DISTINCT-SAME: ptr addrspace(1) noalias noundef readonly align 4 %4,
// DISTINCT-SAME: ptr addrspace(1) noalias noundef readonly align 4 %8,

// ALIAS: define {{.*}}spir_kernel void @_ZTS4Sink(ptr addrspace(1) noundef align 4 %0,
// ALIAS-SAME: ptr addrspace(1) noundef readonly align 4 %4,
// ALIAS-SAME: ptr addrspace(1) noalias noundef readonly align 4 %8,

// ALIAS-REMARK-DAG: remark: kernel-param-noalias: kernel @_ZTS4Sink arg 0 (accessor 'C'): not stamped ('C' and 'A' use the same memory
// ALIAS-REMARK-DAG: remark: kernel-param-noalias: kernel @_ZTS4Sink arg 4 (accessor 'A'): not stamped ('A' and 'C' use the same memory
// ALIAS-REMARK-DAG: remark: kernel-param-noalias: kernel @_ZTS4Sink arg 8 (accessor 'B'): noalias

// -DUSM2: pointers C, A, B.
// USM: define {{.*}}spir_kernel void @_ZTS4Sink(ptr addrspace(1) noundef align 4 %0, ptr addrspace(1) noundef align 4 %1, ptr addrspace(1) noalias noundef align 4 %2)

// USM-REMARK-DAG: remark: kernel-param-noalias: kernel @_ZTS4Sink arg 0 (pointer 'C'): not stamped ('C' and 'A' use the same memory, a USM allocation
// USM-REMARK-DAG: remark: kernel-param-noalias: kernel @_ZTS4Sink arg {{[0-9]+}} ({{accessor|pointer}} 'A'): not stamped ('A' and 'C' use the same memory, {{.*}}a USM allocation
// USM-REMARK-DAG: remark: kernel-param-noalias: kernel @_ZTS4Sink arg {{[0-9]+}} (pointer 'B'): noalias (a USM allocation

// -DUSMBUF: pointer C, accessor A, pointer B.
// USMBUF: define {{.*}}spir_kernel void @_ZTS4Sink(ptr addrspace(1) noundef align 4 %0, ptr addrspace(1) noundef readonly align 4 %1,
// USMBUF-SAME: ptr addrspace(1) noalias noundef align 4 %5)
