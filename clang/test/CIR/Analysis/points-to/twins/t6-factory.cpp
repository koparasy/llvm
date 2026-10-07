// T6 (one factory per pointer, the second at depth 8 through wrappers) and its
// twin negatives (plan V3). Positive: make1() and w7() -> ... -> make2(); the
// accesses on the PAIR line in use() are NoAlias (distinct-objects), checked
// under closed-tu and also open. The twins run under closed-tu (but OPEN),
// change the program so that the pair must not be NoAlias, and every run (all
// NoAlias pairs checked) must stay clean:
//   TWIN_SAME      the wrappers end in make1() too: one factory called twice
//                  (no real alias; the analysis is context-insensitive, so one
//                  allocation site, and the pair is not NoAlias)
//   TWIN_ASSIGN    h.q = h.p
//   TWIN_ESCAPE    ext_alias(&h), defined in the external part, sets q = p
//   TWIN_OPEN      use() externally callable, closed-tu off, and an external
//                  caller passes the same object twice
//
// RUN: %pt_run %s %t.pos %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=POS --input-file=%t.pos.table
// RUN: %pt_run %s %t.posci %pt_opts 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=POS --input-file=%t.posci.table
// RUN: env CXXFLAGS=-DTWIN_SAME %pt_run %s %t.same %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.same.table
// RUN: env CXXFLAGS=-DTWIN_ASSIGN %pt_run %s %t.assign %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.assign.table
// RUN: env CXXFLAGS=-DTWIN_ESCAPE %pt_run %s %t.escape %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.escape.table
// RUN: env CXXFLAGS=-DTWIN_OPEN %pt_run %s %t.open %pt_opts 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.open.table
//
// EXEC-NOT: PT-VIOLATION
// EXEC: PT-CHECKS {{[1-9][0-9]*}}

struct T {
  int v[4];
};
// The owner of the two cells.
struct Ptrs {
  T *p, *q;
};
extern "C" void ext_alias(Ptrs *h);

#ifdef TWIN_OPEN
#define USE_LINKAGE extern "C"
#else
#define USE_LINKAGE static
#endif
USE_LINKAGE int use(T *a, T *b);
extern "C" int ext_call_use();

#ifdef PT_EXTERNAL
extern "C" void ext_alias(Ptrs *h) { h->q = h->p; }
#ifdef TWIN_OPEN
extern "C" int ext_call_use() {
  T t{{1, 2, 3, 4}};
  return use(&t, &t);
}
#endif
#else
static T *make1() { return new T{{1, 2, 3, 4}}; }
static T *make2() { return new T{{5, 6, 7, 8}}; }
#ifdef TWIN_SAME
// The same factory at both depths.
static T *w1() { return make1(); }
#else
static T *w1() { return make2(); }
#endif
static T *w2() { return w1(); }
static T *w3() { return w2(); }
static T *w4() { return w3(); }
static T *w5() { return w4(); }
static T *w6() { return w5(); }
static T *w7() { return w6(); }

USE_LINKAGE int use(T *a, T *b) {
  for (int i = 0; i < 4; ++i)
    a->v[i] = b->v[i] + i; // PAIR
    // POS: {{^[0-9]+}} @{{[^ ]+}} distinct-objects {{[a-z.]+}}(cir.{{[a-z_]+}}@t6-factory.cpp:[[#@LINE-1]]:{{[0-9]+}}) {{[0-9]+}} | {{[a-z.]+}}(cir.{{[a-z_]+}}@t6-factory.cpp:[[#@LINE-1]]:{{[0-9]+}}) {{[0-9]+}} at
    // TWIN: cand {{[0-9]+}} @{{[^ ]+}} {{MayAlias|PartialAlias|MustAlias}} {{[a-z-]+}} {{[a-z.]+}}(cir.{{[a-z_]+}}@t6-factory.cpp:[[#@LINE-2]]:{{[0-9]+}}) {{[0-9]+}} | {{[a-z.]+}}(cir.{{[a-z_]+}}@t6-factory.cpp:[[#@LINE-2]]:{{[0-9]+}}) {{[0-9]+}}{{$}}
  return a->v[0];
}

int main() {
  Ptrs h{make1(), w7()};
  T *keep = h.q;
#ifdef TWIN_ASSIGN
  h.q = h.p;
#endif
#ifdef TWIN_ESCAPE
  ext_alias(&h);
#endif
  int r = use(h.p, h.q);
#ifdef TWIN_OPEN
  r += ext_call_use();
#endif
  delete h.p;
  delete keep;
  return r > 0 ? 0 : 1;
}
#endif
