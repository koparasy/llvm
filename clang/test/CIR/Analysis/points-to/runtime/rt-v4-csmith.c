// Run-time regression of the soundness bugs the V4 Csmith campaign found
// (reduced programs; V4fix). Every NoAlias and MustAlias pair is checked
// (max-pairs-per-func=0) and every constant load (values=1), in the four
// modes ci, closed-tu, bu and bu+closed-tu; none may be violated. Before the
// fixes:
//  - @f2 (F2): bu claimed `load g_7` NoAlias `store *p` while p == &g_7
//    (PT-VIOLATION): projectClones did not bring back that the formal's
//    copy is the shared node of g_7 in the caller (now NodeFlags::MayBeShared);
//  - @f1 (F1) and @hole: bu claimed the local and `*p` MustAlias (one origin
//    in the callee's own graph) while p points to the caller's object
//    (PT-MUST-VIOLATION);
//  - @byval (s1121): every mode claimed the byval formal and the sret
//    destination MustAlias, because the formal was bound to the caller's
//    object instead of the callee's copy of it (PT-MUST-VIOLATION);
//  - @mergedGlobal (seed 100570, values=1): the load of g_89 was claimed
//    constant 3 (PT-VALUE-VIOLATION): g_89's initializer was indexed at
//    offset 0 of the node it shares with g_110 instead of at g_110.f1;
//  - @gotoFn (seeds 1027, 1039): the instrumented module did not lower
//    ("operand #0 does not dominate this use"; see instrument-goto.cir).
//
// REQUIRES: native, system-linux, target=x86_64{{.*}}
// RUN: %clang -fclangir -emit-cir -O0 %s -o %t.cir
// RUN: %clang -c %S/../Inputs/pt_check_runtime.c -o %t.rt.o
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.ci.table max-pairs-per-func=0 values=1' -o %t.ci.cir
// RUN: FileCheck %s --check-prefixes=TABLE --input-file=%t.ci.table
// RUN: cir-translate --cir-to-llvmir %t.ci.cir -o %t.ci.ll
// RUN: grep -v '^target datalayout' %t.ci.ll > %t.ci.fixed.ll
// RUN: %clang %t.ci.fixed.ll %t.rt.o -o %t.ci.exe
// RUN: %t.ci.exe 2>&1 | FileCheck %s --check-prefix=CLEAN
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.closed.table max-pairs-per-func=0 values=1 closed-tu=1' -o %t.closed.cir
// RUN: FileCheck %s --check-prefixes=TABLE --input-file=%t.closed.table
// RUN: cir-translate --cir-to-llvmir %t.closed.cir -o %t.closed.ll
// RUN: grep -v '^target datalayout' %t.closed.ll > %t.closed.fixed.ll
// RUN: %clang %t.closed.fixed.ll %t.rt.o -o %t.closed.exe
// RUN: %t.closed.exe 2>&1 | FileCheck %s --check-prefix=CLEAN
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.bu.table max-pairs-per-func=0 values=1 mode=bu' -o %t.bu.cir
// RUN: FileCheck %s --check-prefixes=TABLE,BU --input-file=%t.bu.table
// RUN: cir-translate --cir-to-llvmir %t.bu.cir -o %t.bu.ll
// RUN: grep -v '^target datalayout' %t.bu.ll > %t.bu.fixed.ll
// RUN: %clang %t.bu.fixed.ll %t.rt.o -o %t.bu.exe
// RUN: %t.bu.exe 2>&1 | FileCheck %s --check-prefix=CLEAN
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.buclosed.table max-pairs-per-func=0 values=1 mode=bu closed-tu=1' -o %t.buclosed.cir
// RUN: FileCheck %s --check-prefixes=TABLE,BU --input-file=%t.buclosed.table
// RUN: cir-translate --cir-to-llvmir %t.buclosed.cir -o %t.buclosed.ll
// RUN: grep -v '^target datalayout' %t.buclosed.ll > %t.buclosed.fixed.ll
// RUN: %clang %t.buclosed.fixed.ll %t.rt.o -o %t.buclosed.exe
// RUN: %t.buclosed.exe 2>&1 | FileCheck %s --check-prefix=CLEAN
//
// The byval copy is another object than the sret destination, in every mode.
// TABLE-DAG: @byval distinct-objects copy.src(formal#3) 24 | copy.dst(formal#0) 24
// Bottom-up, the pair is answered in @f2's one calling context (top-down),
// where p is g_7: the same address, as in ci (its own graph says MayAlias:
// MayBeShared).
// BU-DAG: @f2 must calling-contexts load(cir.get_global @g_7@{{.*}}) 4 | store(cir.load@{{.*}}) 4
// BU-NOT: @f1 must
// BU-NOT: @hole must
// TABLE: # functions
//
// CLEAN-NOT: VIOLATION
// CLEAN: PT-CHECKS {{[1-9][0-9]*}}
// CLEAN: PT-VALUE-CHECKS {{[1-9][0-9]*}}

// F2: the callee reads the global directly and writes it through its formal.
static unsigned g_7 = 5;
static int f2(unsigned *p) {
  *p = g_7 + 1;
  return 0;
}

// F1: a local merged with the formal's pointee through a slot; the formal
// points to a global.
static int g_5;
static void f1(int *p_25) {
  int l_49 = 1;
  int *l_62 = &l_49;
  l_62 = p_25;
  *p_25 = l_49;
}

// The same, with the caller's local for the global.
static void hole(int *p) {
  int x = 1;
  int *q = &x;
  q = p;
  *p = x;
}

// s1121: the caller passes l_87 by value and as the sret destination.
struct S2 {
  unsigned long f1;
  long f2;
  unsigned short f3;
};
static struct S2 s2g = {4, 5, 6};
static struct S2 byval(long a, int *b, struct S2 c) {
  s2g.f2 += c.f3;
  return s2g;
}
static void byvalCaller(void) {
  struct S2 l_87 = {1, 2, 3};
  struct S2 *l_218 = &l_87;
  *l_218 = byval(0, 0, l_87);
}

// s1027/s1039: a goto into a label block after a global's address is made.
static int g_16;
static int gotoFn(int c) {
  if (c)
    goto lbl;
  g_16 = -7;
lbl:
  return g_16;
}

// Seed 100570 (values=1): g_89 is unified with g_110.f1 through g_154, so
// it sits at an offset in g_110's node.
struct S0 {
  long f0;
  short f1;
  long f2;
};
static short g_89 = 4;
static struct S0 g_110 = {7, 3, 9};
static short *g_154 = &g_110.f1;
static int mergedGlobal(int c) {
  if (c)
    g_154 = &g_89;
  short v = g_89;
  return v + *g_154;
}

int main(void) {
  f2(&g_7);
  f1(&g_5);
  int y = 0;
  hole(&y);
  byvalCaller();
  int m = mergedGlobal(g_5 == 1);
  int r = gotoFn(1) + gotoFn(0);
  return !(g_7 == 6 && g_5 == 1 && y == 1 && s2g.f2 == 8 && m == 8 && r == -7);
}
