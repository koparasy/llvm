// T13 (three std::shared_ptr members from make_shared, real libstdc++) and its
// twin negatives (plan V3). Positive, under closed-tu: C and A are set by two
// make_shared calls in init() and read in run(); the accesses on the PAIR line
// should be NoAlias, but today they are MayAlias (collapsed, N11), see the
// T17 comment. The positive still runs. The twins run under closed-tu (but
// OPEN), make C and A share one object, so the pair must not be NoAlias, and
// every run (all NoAlias pairs checked) must stay clean:
//   TWIN_ASSIGN    C = A in init()
//   TWIN_COPY      C move-assigned from a copy-constructed shared_ptr(A)
//   TWIN_SWAP      std::swap(C, other) with other = A
//   TWIN_REASSIGN  b.C = b.A between init() and run()
//   TWIN_ESCAPE    ext_alias(&b), defined in the external part, sets C = A
//   TWIN_INDIRECT  ext_alias called through a function pointer from the
//                  external part
//   TWIN_OPEN      closed-tu off, and an external caller runs a Bench whose
//                  A, B and C are one object
//
// RUN: %pt_run %s %t.pos %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=POS --input-file=%t.pos.table
// RUN: env CXXFLAGS=-DTWIN_ASSIGN %pt_run %s %t.assign %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.assign.table
// RUN: env CXXFLAGS=-DTWIN_COPY %pt_run %s %t.copy %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.copy.table
// RUN: env CXXFLAGS=-DTWIN_SWAP %pt_run %s %t.swap %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.swap.table
// RUN: env CXXFLAGS=-DTWIN_REASSIGN %pt_run %s %t.reassign %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.reassign.table
// RUN: env CXXFLAGS=-DTWIN_ESCAPE %pt_run %s %t.escape %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.escape.table
// RUN: env CXXFLAGS=-DTWIN_INDIRECT %pt_run %s %t.indirect %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.indirect.table
// RUN: env CXXFLAGS=-DTWIN_OPEN %pt_run %s %t.open %pt_opts 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.open.table
//
// EXEC-NOT: PT-VIOLATION
// EXEC: PT-CHECKS {{[1-9][0-9]*}}

#include <memory>
#include <utility>

struct Arr {
  float v[32];
};
struct Bench {
  std::shared_ptr<Arr> A, B, C;
  void init();
  float run();
};
extern "C" void ext_alias(Bench *b);
using AliasFn = void (*)(Bench *);
extern "C" AliasFn ext_get_alias();
extern "C" int ext_call_run();

#ifdef PT_EXTERNAL
extern "C" void ext_alias(Bench *b) { b->C = b->A; }
extern "C" AliasFn ext_get_alias() { return ext_alias; }
#ifdef TWIN_OPEN
extern "C" int ext_call_run() {
  Bench b;
  b.A = b.B = b.C = std::make_shared<Arr>();
  return b.run() == 63.0f ? 0 : 1;
}
#endif
#else
void Bench::init() {
  A = std::make_shared<Arr>();
  B = std::make_shared<Arr>();
#if defined(TWIN_ASSIGN)
  C = A;
#elif defined(TWIN_COPY)
  std::shared_ptr<Arr> copy(A);
  C = std::move(copy);
#elif defined(TWIN_SWAP)
  std::shared_ptr<Arr> other = A;
  std::swap(C, other);
#else
  C = std::make_shared<Arr>();
#endif
}

float Bench::run() {
  for (int i = 0; i < 32; ++i)
    B->v[i] = 2 * i;
  // T17: the POS line below must become NoAlias (distinct-objects). Today the
  // three make_shared objects are one collapsed node (the N11 blob).
  for (int i = 0; i < 32; ++i)
    C->v[i] = A->v[i] + 1; // PAIR
    // POS: cand {{[0-9]+}} @{{[^ ]+}} {{MayAlias|PartialAlias|MustAlias}} collapsed {{[a-z.]+}}(cir.{{[a-z_]+}}@t13-shared-ptr.cpp:[[#@LINE-1]]:{{[0-9]+}}) {{[0-9]+}} | {{[a-z.]+}}(cir.{{[a-z_]+}}@t13-shared-ptr.cpp:[[#@LINE-1]]:{{[0-9]+}}) {{[0-9]+}}{{$}}
    // TWIN: cand {{[0-9]+}} @{{[^ ]+}} {{MayAlias|PartialAlias|MustAlias}} {{[a-z-]+}} {{[a-z.]+}}(cir.{{[a-z_]+}}@t13-shared-ptr.cpp:[[#@LINE-2]]:{{[0-9]+}}) {{[0-9]+}} | {{[a-z.]+}}(cir.{{[a-z_]+}}@t13-shared-ptr.cpp:[[#@LINE-2]]:{{[0-9]+}}) {{[0-9]+}}{{$}}
  return C->v[31];
}

int main() {
  Bench b;
  b.init();
#ifdef TWIN_REASSIGN
  b.C = b.A;
#endif
#ifdef TWIN_ESCAPE
  ext_alias(&b);
#endif
#ifdef TWIN_INDIRECT
  ext_get_alias()(&b);
#endif
  int r = b.run() >= 1.0f ? 0 : 1;
#ifdef TWIN_OPEN
  r += ext_call_run();
#endif
  return r;
}
#endif
