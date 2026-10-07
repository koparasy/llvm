// T3 (fields) and its twin negatives (plan V3). Positive: two fields of one
// struct, reached through an owner struct and passed to use(); the accesses on
// the PAIR line are NoAlias (disjoint-ranges). Each twin changes the program
// so that the pair must not be NoAlias, and every run (all NoAlias pairs
// checked) must stay clean:
//   TWIN_ASSIGN    h.q = h.p
//   TWIN_ESCAPE    ext_alias(&h), defined in the external part, sets q = p
//   TWIN_REASSIGN  h.q re-pointed to &s.x between setup and use
//   TWIN_INDIRECT  ext_alias called through a function pointer from the
//                  external part
//   TWIN_OPEN      use() externally callable (closed-tu off), and an external
//                  caller passes the same address twice
//
// RUN: %pt_run %s %t.pos %pt_opts 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=POS --input-file=%t.pos.table
// RUN: env CXXFLAGS=-DTWIN_ASSIGN %pt_run %s %t.assign %pt_opts 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.assign.table
// RUN: env CXXFLAGS=-DTWIN_ESCAPE %pt_run %s %t.escape %pt_opts 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.escape.table
// RUN: env CXXFLAGS=-DTWIN_REASSIGN %pt_run %s %t.reassign %pt_opts 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.reassign.table
// RUN: env CXXFLAGS=-DTWIN_INDIRECT %pt_run %s %t.indirect %pt_opts 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.indirect.table
// RUN: env CXXFLAGS=-DTWIN_OPEN %pt_run %s %t.open %pt_opts 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.open.table
//
// EXEC-NOT: PT-VIOLATION
// EXEC: PT-CHECKS {{[1-9][0-9]*}}

struct S {
  int x, y;
};
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
  int z = 1;
  return use(&z, &z) == 3 ? 0 : 1;
}
#endif
#else
USE_LINKAGE int use(int *p, int *q) {
  *p = *q + 1; // PAIR
  // POS: {{^[0-9]+}} @{{[^ ]+}} disjoint-ranges {{[a-z.]+}}(cir.{{[a-z_]+}}@t3-fields.cpp:[[#@LINE-1]]:{{[0-9]+}}) {{[0-9]+}} | {{[a-z.]+}}(cir.{{[a-z_]+}}@t3-fields.cpp:[[#@LINE-1]]:{{[0-9]+}}) {{[0-9]+}} at
  // TWIN: cand {{[0-9]+}} @{{[^ ]+}} {{MayAlias|PartialAlias|MustAlias}} {{[a-z-]+}} {{[a-z.]+}}(cir.{{[a-z_]+}}@t3-fields.cpp:[[#@LINE-2]]:{{[0-9]+}}) {{[0-9]+}} | {{[a-z.]+}}(cir.{{[a-z_]+}}@t3-fields.cpp:[[#@LINE-2]]:{{[0-9]+}}) {{[0-9]+}}{{$}}
  return *p + *q;
}

int main() {
  S s{1, 2};
  Ptrs h{&s.x, &s.y};
#ifdef TWIN_ASSIGN
  h.q = h.p;
#endif
#ifdef TWIN_ESCAPE
  ext_alias(&h);
#endif
#ifdef TWIN_REASSIGN
  h.q = &s.y;
  s.y = 5;
  h.q = &s.x;
#endif
#ifdef TWIN_INDIRECT
  ext_get_alias()(&h);
#endif
  int r = use(h.p, h.q);
#ifdef TWIN_OPEN
  r += ext_call_use();
#endif
  return r > 0 ? 0 : 1;
}
#endif
