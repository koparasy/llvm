// Run-time check of the points-to facts of internalized paths (T26): the
// instrument pass first internalizes every path from main
// (LaunchPathInternalization.h, every function a sink), then checks every
// NoAlias pair. main calls the copies of `work` (directly, through a
// function pointer, made a direct call, and through a virtual call, guarded)
// with two distinct objects; the external part (compiled without CIR or
// checks) calls the original `work` with the same object twice. The linker
// keeps the instrumented copy of the linkonce_odr `work` (first on the link
// line), so the external call runs the original's checks.
//  - open, internalized: the copies' formals are complete (only main calls
//    them), so `*a` and `*b` are NoAlias there and checked; the original's
//    stay unknown: no violation;
//  - closed-TU without internalization (the negative control): the
//    original is assumed to have only main's callers, its pair is NoAlias,
//    and the external call with a == b aborts.
//
// REQUIRES: native, system-linux, target=x86_64{{.*}}
// RUN: %clangxx -fclangir -emit-cir -O0 %s -o %t.cir
// RUN: %clang -c %S/../Inputs/pt_check_runtime.c -o %t.rt.o
// RUN: %clangxx -c -O0 -DPT_EXTERNAL %s -o %t.ext.o
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.int.table max-pairs-per-func=0 mode=bu local-closed=1 internalize=1' \
// RUN:   -o %t.int.cir 2>&1 | FileCheck %s --check-prefix=INT
// RUN: FileCheck %s --check-prefix=TABLE --input-file=%t.int.table
// RUN: FileCheck %s --check-prefix=ORIG --input-file=%t.int.table
// RUN: cir-translate --cir-to-llvmir %t.int.cir -o %t.int.ll
// RUN: grep -v '^target datalayout' %t.int.ll > %t.int.fixed.ll
// RUN: %clangxx %t.int.fixed.ll %t.rt.o %t.ext.o -o %t.int.exe
// RUN: %t.int.exe 2>&1 | FileCheck %s --check-prefix=EXEC
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.closed.table max-pairs-per-func=0 mode=bu closed-tu=1' \
// RUN:   -o %t.closed.cir
// RUN: cir-translate --cir-to-llvmir %t.closed.cir -o %t.closed.ll
// RUN: grep -v '^target datalayout' %t.closed.ll > %t.closed.fixed.ll
// RUN: %clangxx %t.closed.fixed.ll %t.rt.o %t.ext.o -o %t.closed.exe
// RUN: not --crash %t.closed.exe 2>&1 | FileCheck %s --check-prefix=VIOLATION

// INT: PT-INTERNALIZED copies={{[1-9][0-9]*}} in-place={{[0-9]+}} guards=2 vtables=0
// The copy's *a and *b (the stores of `work`) are a checked NoAlias pair;
// the original's are not claimed.
// TABLE: @_Z4workPiS_.int distinct-objects store(cir.load@rt-internalize.cpp:{{[0-9]+}}:4) 4 | store(cir.load@rt-internalize.cpp:{{[0-9]+}}:4) 4
// ORIG-NOT: @_Z4workPiS_ {{[a-z-]+}} store(cir.load@rt-internalize.cpp:{{[0-9]+}}:4) 4 | store(cir.load@rt-internalize.cpp:{{[0-9]+}}:4) 4
// EXEC-NOT: PT-VIOLATION
// EXEC: PT-CHECKS {{[1-9][0-9]*}}
// VIOLATION: PT-VIOLATION

inline int work(int *a, int *b) {
  *a = 1;
  *b = 2;
  return *a;
}

#ifdef PT_EXTERNAL
int ext_call() {
  int x;
  return work(&x, &x);
}
#else
struct Base {
  virtual int run(int *a, int *b) = 0;
};
struct Impl : Base {
  int run(int *a, int *b) override { return work(a, b); }
};
int ext_call();

int main() {
  int x, y;
  int r = work(&x, &y);
  int (*fp)(int *, int *) = &work;
  r += fp(&x, &y);
  Impl impl;
  Base *p = &impl;
  r += p->run(&x, &y);
  r += ext_call();
  return r == 5 ? 0 : 1;
}
#endif
