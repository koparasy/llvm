// T14 (std::vector<float> x3, .data() pointers, real libstdc++) and its twin
// negatives (plan V3). Positive, under closed-tu: C.data() and A.data() are
// passed to use(); the accesses on the PAIR line should be NoAlias, but today
// they are MayAlias (array-may-overlap: one allocator site), see the T17
// comment. The positive still runs. The twins run under closed-tu (but OPEN),
// make the two pointers one buffer, so the pair must not be NoAlias, and every
// run (all NoAlias pairs checked) must stay clean:
//   TWIN_ASSIGN    h.a = h.c
//   TWIN_SWAP      std::swap(A, C), then h.a = A.data(): the old C buffer
//   TWIN_REASSIGN  h.a = C.data() between setup and use
//   TWIN_ESCAPE    ext_alias(&h), defined in the external part, sets a = c
//   TWIN_OPEN      use() externally callable, closed-tu off, and an external
//                  caller passes one vector's data() twice
//
// RUN: %pt_run %s %t.pos %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=POS --input-file=%t.pos.table
// RUN: env CXXFLAGS=-DTWIN_ASSIGN %pt_run %s %t.assign %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.assign.table
// RUN: env CXXFLAGS=-DTWIN_SWAP %pt_run %s %t.swap %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.swap.table
// RUN: env CXXFLAGS=-DTWIN_REASSIGN %pt_run %s %t.reassign %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.reassign.table
// RUN: env CXXFLAGS=-DTWIN_ESCAPE %pt_run %s %t.escape %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.escape.table
// RUN: env CXXFLAGS=-DTWIN_OPEN %pt_run %s %t.open %pt_opts 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.open.table
//
// EXEC-NOT: PT-VIOLATION
// EXEC: PT-CHECKS {{[1-9][0-9]*}}

#include <utility>
#include <vector>

// The owner of the two cells.
struct Ptrs {
  float *c, *a;
};
extern "C" void ext_alias(Ptrs *h);

#ifdef TWIN_OPEN
#define USE_LINKAGE extern "C"
#else
#define USE_LINKAGE static
#endif
USE_LINKAGE float use(float *c, float *a);
extern "C" int ext_call_use();

#ifdef PT_EXTERNAL
extern "C" void ext_alias(Ptrs *h) { h->a = h->c; }
#ifdef TWIN_OPEN
extern "C" int ext_call_use() {
  std::vector<float> v(64, 1.0f);
  return use(v.data(), v.data()) == 2.0f ? 0 : 1;
}
#endif
#else
USE_LINKAGE float use(float *c, float *a) {
  // T17: the POS line below must become NoAlias (distinct-objects). Today the
  // three vector buffers come from one allocation site in the allocator,
  // which the context-insensitive analysis merges into one array node.
  for (int i = 0; i < 64; ++i)
    c[i] = a[i] + 1; // PAIR
    // POS: cand {{[0-9]+}} @{{[^ ]+}} {{MayAlias|PartialAlias|MustAlias}} array-may-overlap {{[a-z.]+}}(cir.{{[a-z_]+}}@t14-vector.cpp:[[#@LINE-1]]:{{[0-9]+}}) {{[0-9]+}} | {{[a-z.]+}}(cir.{{[a-z_]+}}@t14-vector.cpp:[[#@LINE-1]]:{{[0-9]+}}) {{[0-9]+}}{{$}}
    // TWIN: cand {{[0-9]+}} @{{[^ ]+}} {{MayAlias|PartialAlias|MustAlias}} {{[a-z-]+}} {{[a-z.]+}}(cir.{{[a-z_]+}}@t14-vector.cpp:[[#@LINE-2]]:{{[0-9]+}}) {{[0-9]+}} | {{[a-z.]+}}(cir.{{[a-z_]+}}@t14-vector.cpp:[[#@LINE-2]]:{{[0-9]+}}) {{[0-9]+}}{{$}}
  return c[63];
}

int main() {
  std::vector<float> A(64, 1.0f), B(64, 2.0f), C(64);
  Ptrs h{C.data(), A.data()};
  float *b = B.data();
#ifdef TWIN_ASSIGN
  h.a = h.c;
#endif
#ifdef TWIN_SWAP
  // After the swap A owns the old C buffer, which h.c still points to.
  std::swap(A, C);
  h.a = A.data();
#endif
#ifdef TWIN_REASSIGN
  h.a = C.data();
#endif
#ifdef TWIN_ESCAPE
  ext_alias(&h);
#endif
  float r = use(h.c, h.a) + b[0];
#ifdef TWIN_OPEN
  r += ext_call_use();
#endif
  return r >= 3.0f ? 0 : 1;
}
#endif
