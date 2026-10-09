// Run-time check of the bottom-up status propagation (T27): in g, an unknown
// call is handed &s, and s holds &y only through the store init makes, a
// link g's graph gets when init's graph is copied, after g's local phase
// spread External. Before propagateBottomUpStatus, main's copy of g's graph
// got y without External, and the load of y after the call was claimed to
// be the 0 stored before it, while ext stored 7.
//
// REQUIRES: native, system-linux, target=x86_64{{.*}}
// RUN: %clangxx -fclangir -emit-cir -O0 %s -o %t.cir
// RUN: %clangxx -O0 -DPT_EXTERNAL -c %s -o %t.ext.o
// RUN: %clang -c %S/../Inputs/pt_check_runtime.c -o %t.rt.o
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.bu.table values=1 mode=bu' -o %t.bu.cir
// RUN: FileCheck %s --check-prefix=TABLE --input-file=%t.bu.table
// RUN: cir-translate --cir-to-llvmir %t.bu.cir -o %t.bu.ll
// RUN: grep -v '^target datalayout' %t.bu.ll > %t.bu.fixed.ll
// RUN: %clangxx %t.bu.fixed.ll %t.ext.o %t.rt.o -o %t.bu.exe
// RUN: %t.bu.exe 2>&1 | FileCheck %s --check-prefix=CLEAN
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.buc.table values=1 mode=bu closed-tu=1 subobjects=1' -o %t.buc.cir
// RUN: FileCheck %s --check-prefix=TABLE --input-file=%t.buc.table
// RUN: cir-translate --cir-to-llvmir %t.buc.cir -o %t.buc.ll
// RUN: grep -v '^target datalayout' %t.buc.ll > %t.buc.fixed.ll
// RUN: %clangxx %t.buc.fixed.ll %t.ext.o %t.rt.o -o %t.buc.exe
// RUN: %t.buc.exe 2>&1 | FileCheck %s --check-prefix=CLEAN
//
// TABLE-NOT: value {{[0-9]+}} @main {{.*}}load(cir.alloca 'y'
// TABLE: # functions
// CLEAN: PT-CHECKS {{[1-9][0-9]*}}
// CLEAN-NOT: VIOLATION

struct S {
  int *p;
};
#ifdef PT_EXTERNAL
void ext(S *s) { *s->p = 7; }
#else
void ext(S *s);
static void init(S *s, int *x) { s->p = x; }
static void g(int *x) {
  S s;
  init(&s, x);
  ext(&s);
}
int main() {
  int y = 0;
  g(&y);
  return y == 7 ? 0 : 1;
}
#endif
