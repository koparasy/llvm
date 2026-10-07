// T10 (the owner of the cells passed to an external readOnly function) and its
// twin negatives (plan V3). Positive: ext_read(&h) is __attribute__((pure)),
// so it cannot rewrite h.p or h.q, and the accesses on the PAIR line in use()
// are NoAlias (distinct-objects). Each twin changes the program so that the
// pair must not be NoAlias, and every run (all NoAlias pairs checked) must
// stay clean:
//   TWIN_ASSIGN    h.q = h.p after the readOnly call
//   TWIN_ESCAPE    also ext_alias(&h), which may write the cells and sets
//                  q = p
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
// Reads the cells and what they point to, writes nothing (readOnly).
extern "C" __attribute__((pure)) int ext_read(const Ptrs *h);
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
extern "C" int ext_read(const Ptrs *h) { return *h->p + *h->q; }
extern "C" void ext_alias(Ptrs *h) { h->q = h->p; }
extern "C" AliasFn ext_get_alias() { return ext_alias; }
#ifdef TWIN_OPEN
extern "C" int ext_call_use() {
  int z = 1;
  return use(&z, &z) == 2 ? 0 : 1;
}
#endif
#else
USE_LINKAGE int use(int *p, int *q) {
  *q = *p + 1; // PAIR
  // POS: {{^[0-9]+}} @{{[^ ]+}} distinct-objects {{[a-z.]+}}(cir.{{[a-z_]+}}@t10-readonly.cpp:[[#@LINE-1]]:{{[0-9]+}}) {{[0-9]+}} | {{[a-z.]+}}(cir.{{[a-z_]+}}@t10-readonly.cpp:[[#@LINE-1]]:{{[0-9]+}}) {{[0-9]+}} at
  // TWIN: cand {{[0-9]+}} @{{[^ ]+}} {{MayAlias|PartialAlias|MustAlias}} {{[a-z-]+}} {{[a-z.]+}}(cir.{{[a-z_]+}}@t10-readonly.cpp:[[#@LINE-2]]:{{[0-9]+}}) {{[0-9]+}} | {{[a-z.]+}}(cir.{{[a-z_]+}}@t10-readonly.cpp:[[#@LINE-2]]:{{[0-9]+}}) {{[0-9]+}}{{$}}
  return *q;
}

int main() {
  int a = 3, b = 4;
  Ptrs h{&a, &b};
  int r = ext_read(&h);
#ifdef TWIN_ASSIGN
  h.q = h.p;
#endif
#ifdef TWIN_ESCAPE
  // The same call, to a function that may write the cells.
  ext_alias(&h);
#endif
#ifdef TWIN_INDIRECT
  ext_get_alias()(&h);
#endif
  r += use(h.p, h.q);
#ifdef TWIN_OPEN
  r += ext_call_use();
#endif
  return r >= 7 ? 0 : 1;
}
#endif
