// T11 (a virtual call devirtualized from the vptr of a local object) and its
// twin negatives (plan V3). Positive, under closed-tu: b->f() resolves to
// Local::f with this = &l, and the accesses on the PAIR line are NoAlias
// (distinct-objects). Open, Local::f's this is incomplete (its address is in
// the vtable, N14) and the pair is not NoAlias. The twins change the program
// so that the pair must not be NoAlias, and every run (all NoAlias pairs
// checked) must stay clean:
//   TWIN_ASSIGN    l.in = l.out (closed-tu)
//   TWIN_REASSIGN  l.in re-pointed to &cell before the call (closed-tu)
//   TWIN_ESCAPE    ext_alias(&l), defined in the external part, sets
//                  in = out (open)
//   TWIN_INDIRECT  the virtual call is on a Local built by the external part
//                  with in == out: unknown target (open)
//   TWIN_OPEN      closed-tu off, and an external caller calls f() on a Local
//                  with in == out
// TWIN_ESCAPE and TWIN_INDIRECT are wrong under closed-tu (NoAlias, and the run
// aborts): the call turns unresolved, but Local::f keeps complete formals.
// That is the closed-tu gap N13, for T18; see the XFAIL tests
// t11-devirt-closed-escape.cpp and t11-devirt-closed-indirect.cpp.
//
// RUN: %pt_run %s %t.pos %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=POS --input-file=%t.pos.table
// RUN: env CXXFLAGS=-DTWIN_ASSIGN %pt_run %s %t.assign %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.assign.table
// RUN: env CXXFLAGS=-DTWIN_REASSIGN %pt_run %s %t.reassign %pt_opts closed-tu=1 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.reassign.table
// RUN: env CXXFLAGS=-DTWIN_ESCAPE %pt_run %s %t.escape %pt_opts 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.escape.table
// RUN: env CXXFLAGS=-DTWIN_INDIRECT %pt_run %s %t.indirect %pt_opts 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.indirect.table
// RUN: env CXXFLAGS=-DTWIN_OPEN %pt_run %s %t.open %pt_opts 2>&1 | FileCheck %s --check-prefix=EXEC
// RUN: FileCheck %s --check-prefix=TWIN --input-file=%t.open.table
//
// EXEC-NOT: PT-VIOLATION
// EXEC: PT-CHECKS {{[1-9][0-9]*}}

struct Base {
  virtual int f() { return 1; }
  virtual ~Base() {}
};
struct Local : Base {
  int *out, *in;
  Local(int *o, int *i) : out(o), in(i) {}
  int f() override;
};
extern "C" void ext_alias(Local *l);
// A Local built by the external part, whose out and in are one cell.
extern "C" Base *ext_make(int *cell);
extern "C" int ext_call_f();

#ifdef PT_EXTERNAL
extern "C" void ext_alias(Local *l) { l->in = l->out; }
extern "C" Base *ext_make(int *cell) { return new Local(cell, cell); }
#ifdef TWIN_OPEN
extern "C" int ext_call_f() {
  int z = 1;
  Local l(&z, &z);
  Base *b = &l;
  return b->f() == 4 ? 0 : 1;
}
#endif
#else
// The key function: the vtable and Local::f are only in this TU.
int Local::f() {
  *out = *in + 3; // PAIR
  // POS: {{^[0-9]+}} @{{[^ ]+}} distinct-objects {{[a-z.]+}}(cir.{{[a-z_]+}}@t11-devirt.cpp:[[#@LINE-1]]:{{[0-9]+}}) {{[0-9]+}} | {{[a-z.]+}}(cir.{{[a-z_]+}}@t11-devirt.cpp:[[#@LINE-1]]:{{[0-9]+}}) {{[0-9]+}} at
  // TWIN: cand {{[0-9]+}} @{{[^ ]+}} {{MayAlias|PartialAlias|MustAlias}} {{[a-z-]+}} {{[a-z.]+}}(cir.{{[a-z_]+}}@t11-devirt.cpp:[[#@LINE-2]]:{{[0-9]+}}) {{[0-9]+}} | {{[a-z.]+}}(cir.{{[a-z_]+}}@t11-devirt.cpp:[[#@LINE-2]]:{{[0-9]+}}) {{[0-9]+}}{{$}}
  return *out;
}

int main() {
  int cell = 1, other = 5;
  Local l(&cell, &other);
#ifdef TWIN_ASSIGN
  l.in = l.out;
#endif
#ifdef TWIN_REASSIGN
  l.in = &other;
  other = 6;
  l.in = &cell;
#endif
#ifdef TWIN_ESCAPE
  ext_alias(&l);
#endif
#ifdef TWIN_INDIRECT
  // A virtual call on an object from the external part: unknown target.
  Base *b = ext_make(&cell);
#else
  // Devirtualized from the vptr of l: the target is Local::f.
  Base *b = &l;
#endif
  int r = b->f();
#ifdef TWIN_INDIRECT
  delete b;
#endif
#ifdef TWIN_OPEN
  r += ext_call_f();
#endif
  return r >= 4 ? 0 : 1;
}
#endif
