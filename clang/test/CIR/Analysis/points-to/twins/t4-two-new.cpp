// T4 (two new results) and its twin negatives (plan V3). Positive: two
// new[] results passed to use(); the accesses on the PAIR line are NoAlias
// (distinct-objects). Each twin changes the program so that the pair must
// not be NoAlias, and every run (all NoAlias pairs checked) must stay clean:
//   TWIN_ASSIGN    h.q = h.p
//   TWIN_ESCAPE    ext_alias(&h), defined in the external part, sets q = p
//   TWIN_LOOP      one new[] in a loop, two iterations' results kept (no
//                  real alias; one allocation site, so not NoAlias)
//   TWIN_REASSIGN  h.q written through, then re-pointed to h.p before use
//   TWIN_INDIRECT  ext_alias called through a function pointer from the
//                  external part
//   TWIN_OPEN      use() externally callable (closed-tu off), and an external
//                  caller passes the same array twice
//
// RUN: %pt_run %s %t.pos %pt_opts 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=POS --input-file=%t.pos.table
// RUN: env CXXFLAGS=-DTWIN_ASSIGN %pt_run %s %t.assign %pt_opts 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.assign.table
// RUN: env CXXFLAGS=-DTWIN_ESCAPE %pt_run %s %t.escape %pt_opts 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.escape.table
// RUN: env CXXFLAGS=-DTWIN_LOOP %pt_run %s %t.loop %pt_opts 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.loop.table
// RUN: env CXXFLAGS=-DTWIN_REASSIGN %pt_run %s %t.reassign %pt_opts 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.reassign.table
// RUN: env CXXFLAGS=-DTWIN_INDIRECT %pt_run %s %t.indirect %pt_opts 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.indirect.table
// RUN: env CXXFLAGS=-DTWIN_OPEN %pt_run %s %t.open %pt_opts 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.open.table
//
// EXEC-NOT: PT-VIOLATION
// EXEC: PT-CHECKS {{[1-9][0-9]*}}

// The owner of the two cells.
struct Ptrs {
  int *p, *q;
};
extern "C" void ext_alias(Ptrs *h);
using AliasFn = void (*)(Ptrs *);
extern "C" AliasFn ext_get_alias();

#ifdef TWIN_OPEN
#define USE_LINKAGE extern "C"
#else
#define USE_LINKAGE static
#endif
USE_LINKAGE int use(int *p, int *q);
extern "C" int ext_call_use();

#ifdef PT_EXTERNAL
extern "C" void ext_alias(Ptrs *h) { h->q = h->p; }
extern "C" AliasFn ext_get_alias() { return ext_alias; }
#ifdef TWIN_OPEN
extern "C" int ext_call_use() {
  int *z = new int[8]();
  int r = use(z, z);
  delete[] z;
  return r;
}
#endif
#else
USE_LINKAGE int use(int *p, int *q) {
  for (int i = 0; i < 8; ++i)
    p[i] = q[i] + i; // PAIR
    // POS: {{^[0-9]+}} @{{[^ ]+}} distinct-objects {{[a-z.]+}}(cir.{{[a-z_]+}}@t4-two-new.cpp:[[#@LINE-1]]:{{[0-9]+}}) {{[0-9]+}} | {{[a-z.]+}}(cir.{{[a-z_]+}}@t4-two-new.cpp:[[#@LINE-1]]:{{[0-9]+}}) {{[0-9]+}} at
    // TWIN: cand {{[0-9]+}} @{{[^ ]+}} {{MayAlias|PartialAlias|MustAlias}} {{[a-z-]+}} {{[a-z.]+}}(cir.{{[a-z_]+}}@t4-two-new.cpp:[[#@LINE-2]]:{{[0-9]+}}) {{[0-9]+}} | {{[a-z.]+}}(cir.{{[a-z_]+}}@t4-two-new.cpp:[[#@LINE-2]]:{{[0-9]+}}) {{[0-9]+}}{{$}}
  return p[7];
}

int main() {
#ifdef TWIN_LOOP
  int *v[2];
  for (int i = 0; i < 2; ++i)
    v[i] = new int[8]();
  Ptrs h{v[0], v[1]};
#else
  Ptrs h{new int[8](), new int[8]()};
#endif
  int *keep = h.q;
#ifdef TWIN_ASSIGN
  h.q = h.p;
#endif
#ifdef TWIN_ESCAPE
  ext_alias(&h);
#endif
#ifdef TWIN_REASSIGN
  h.q[0] = 1;
  h.q = h.p + 0;
#endif
#ifdef TWIN_INDIRECT
  ext_get_alias()(&h);
#endif
  int r = use(h.p, h.q);
#ifdef TWIN_OPEN
  r += ext_call_use();
#endif
  delete[] h.p;
  delete[] keep;
  return r >= 7 ? 0 : 1;
}
#endif
