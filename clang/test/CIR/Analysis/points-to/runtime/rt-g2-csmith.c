// Run-time regression of the soundness and termination bugs the G2 Csmith
// campaign found with subobject nodes (reduced programs; G2fix). Every
// NoAlias and MustAlias pair is checked (max-pairs-per-func=0) and every
// constant load (values=1), with subobjects=1, in the four modes ci,
// closed-tu, bu and bu+closed-tu; none may be violated. Before the fixes:
//  - @sretMember (seed 200566): bu claimed g's sret slot and p
//    distinct-objects (PT-VIOLATION). ClangIR passes &l itself as the sret
//    destination of `*pl = g(&l.f4.f0)` (classic codegen uses a temporary),
//    and projectClones only brought back merges of copies that are one node,
//    not of l's node and its grandchild f4.f0;
//  - @sretShared: the same with l2.f4.f0 also merged with g153's (shared)
//    node, a copy the pull-back used to skip;
//  - @wholeSlot (seeds 202245, 203995, 205699, 208275): the instrumentation
//    never finished (closeMemTransfers made a child below each new child,
//    without end; see subobj-copy-whole-slot.cir);
//  - @coerced (seed 309470, values=1): every mode claimed the u64 that
//    ClangIR reloads from the "coerce" slot of a by-value `struct P` argument
//    constant 0 (PT-VALUE-VIOLATION): the slot is written by an aggregate
//    store, which leaves the padding bytes 1-3 undefined (see
//    stored-value-padding.cir).
//
// REQUIRES: native, system-linux, target=x86_64{{.*}}
// RUN: %clang -fclangir -emit-cir -O0 %s -o %t.cir
// RUN: %clang -c %S/../Inputs/pt_check_runtime.c -o %t.rt.o
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.ci.table max-pairs-per-func=0 values=1 subobjects=1' -o %t.ci.cir
// RUN: cir-translate --cir-to-llvmir %t.ci.cir -o %t.ci.ll
// RUN: grep -v '^target datalayout' %t.ci.ll > %t.ci.fixed.ll
// RUN: %clang %t.ci.fixed.ll %t.rt.o -o %t.ci.exe
// RUN: %t.ci.exe 2>&1 | FileCheck %s --check-prefix=CLEAN
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.closed.table max-pairs-per-func=0 values=1 subobjects=1 closed-tu=1' -o %t.closed.cir
// RUN: cir-translate --cir-to-llvmir %t.closed.cir -o %t.closed.ll
// RUN: grep -v '^target datalayout' %t.closed.ll > %t.closed.fixed.ll
// RUN: %clang %t.closed.fixed.ll %t.rt.o -o %t.closed.exe
// RUN: %t.closed.exe 2>&1 | FileCheck %s --check-prefix=CLEAN
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.bu.table max-pairs-per-func=0 values=1 subobjects=1 mode=bu' -o %t.bu.cir
// RUN: cir-translate --cir-to-llvmir %t.bu.cir -o %t.bu.ll
// RUN: grep -v '^target datalayout' %t.bu.ll > %t.bu.fixed.ll
// RUN: %clang %t.bu.fixed.ll %t.rt.o -o %t.bu.exe
// RUN: %t.bu.exe 2>&1 | FileCheck %s --check-prefix=CLEAN
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.buclosed.table max-pairs-per-func=0 values=1 subobjects=1 mode=bu closed-tu=1' -o %t.buclosed.cir
// RUN: cir-translate --cir-to-llvmir %t.buclosed.cir -o %t.buclosed.ll
// RUN: grep -v '^target datalayout' %t.buclosed.ll > %t.buclosed.fixed.ll
// RUN: %clang %t.buclosed.fixed.ll %t.rt.o -o %t.buclosed.exe
// RUN: %t.buclosed.exe 2>&1 | FileCheck %s --check-prefix=CLEAN
//
// CLEAN-NOT: VIOLATION
// CLEAN: PT-CHECKS {{[1-9][0-9]*}}
// CLEAN: PT-VALUE-CHECKS {{[1-9][0-9]*}}

struct S0 {
  int f0;
  short f2;
};
struct S2 {
  short f2;
  struct S0 f4;
  unsigned long f5;
};

// The callee reads through p after its result object (the sret slot) is
// initialised.
static struct S2 g(int *p) {
  struct S2 r = {0};
  r.f2 = (short)*p;
  return r;
}

static int sretMember(void) {
  struct S2 l = {0};
  l.f4.f0 = 7;
  int *q = &l.f4.f0;
  struct S2 *pl = &l;
  *pl = g(q);
  return l.f2;
}

static int g153 = 3;
static int *pick(int *x, int c) { return c ? &g153 : x; }
static struct S2 g2(int *p) {
  struct S2 r = {0};
  r.f2 = (short)*p;
  return r;
}

static int sretShared(int c) {
  struct S2 l2 = {0};
  l2.f4.f0 = 7;
  int *q = &l2.f4.f0;
  struct S2 *pl = &l2;
  *pl = g2(q);
  q = pick(q, c);
  return l2.f2 + *q;
}

struct T0 {
  short f1;
};
struct T1 {
  struct T0 f2;
};
static struct T1 gt;
static struct T0 *gtp = &gt.f2;
static struct T0 wholeSlot(int c) {
  if (c)
    return gt.f2;
  return *gtp;
}

struct P {
  signed char f0;
  unsigned f1;
};
static int takeP(struct P s) { return s.f0 + (int)s.f1; }
static int coerced(void) {
  const struct P l[2] = {{0, 0}, {0, 0}};
  // Leave non-zero bytes on the stack below, where the coerce slot goes.
  volatile char junk[64];
  for (int i = 0; i < 64; i++)
    junk[i] = (char)0xAB;
  return takeP(l[1]);
}

int main(void) {
  int sum = sretMember() + sretShared(0) + wholeSlot(1).f1 + wholeSlot(0).f1 +
            coerced();
  return sum < 0;
}
