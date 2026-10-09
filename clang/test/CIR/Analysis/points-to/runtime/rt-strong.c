// Run-time check of the strong updates of the stored-value queries (T24):
// locals reassigned on straight-line code, in branches, in loops and in
// scopes; every load isConstantAt calls constant (values=1) is checked
// against the constant, in ci and bu+closed-tu, with the indeterminate-value
// rule on and off; none may differ. The negative control
// (inject-wrong-strong=1) makes the strong-update search skip the writes
// between the load and the store it finds: the load of x after the
// `if (argc > 0) x = 3` is then claimed to be 2, and the run must abort with
// PT-VALUE-VIOLATION.
//
// REQUIRES: native, system-linux, target=x86_64{{.*}}
// RUN: %clang -fclangir -emit-cir -O0 %s -o %t.cir
// RUN: %clang -c %S/../Inputs/pt_check_runtime.c -o %t.rt.o
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.ci.table values=1 subobjects=1' -o %t.ci.cir
// RUN: FileCheck %s --check-prefix=TABLE --input-file=%t.ci.table
// RUN: cir-translate --cir-to-llvmir %t.ci.cir -o %t.ci.ll
// RUN: grep -v '^target datalayout' %t.ci.ll > %t.ci.fixed.ll
// RUN: %clang %t.ci.fixed.ll %t.rt.o -o %t.ci.exe
// RUN: %t.ci.exe 2>&1 | FileCheck %s --check-prefix=CLEAN
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.bu.table values=1 subobjects=1 mode=bu closed-tu=1' -o %t.bu.cir
// RUN: FileCheck %s --check-prefixes=TABLE,BU --input-file=%t.bu.table
// RUN: cir-translate --cir-to-llvmir %t.bu.cir -o %t.bu.ll
// RUN: grep -v '^target datalayout' %t.bu.ll > %t.bu.fixed.ll
// RUN: %clang %t.bu.fixed.ll %t.rt.o -o %t.bu.exe
// RUN: %t.bu.exe 2>&1 | FileCheck %s --check-prefix=CLEAN
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.off.table values=1 subobjects=1 mode=bu closed-tu=1 indeterminate-ub=0' -o %t.off.cir
// RUN: FileCheck %s --check-prefixes=TABLE,BU --input-file=%t.off.table
// RUN: cir-translate --cir-to-llvmir %t.off.cir -o %t.off.ll
// RUN: grep -v '^target datalayout' %t.off.ll > %t.off.fixed.ll
// RUN: %clang %t.off.fixed.ll %t.rt.o -o %t.off.exe
// RUN: %t.off.exe 2>&1 | FileCheck %s --check-prefix=CLEAN
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.inject.table values=1 subobjects=1 inject-wrong-strong=1' -o %t.inject.cir
// RUN: FileCheck %s --check-prefix=INJECT --input-file=%t.inject.table
// RUN: cir-translate --cir-to-llvmir %t.inject.cir -o %t.inject.ll
// RUN: grep -v '^target datalayout' %t.inject.ll > %t.inject.fixed.ll
// RUN: %clang %t.inject.fixed.ll %t.rt.o -o %t.inject.exe
// RUN: not --crash %t.inject.exe 2>&1 | FileCheck %s --check-prefix=CANARY
//
// The reassigned locals read back with the value stored last.
// TABLE-DAG: value {{[0-9]+}} @main #cir.int<1> : !cir.int<s, 32> load(cir.alloca 'x'@rt-strong.c:{{.*}})
// TABLE-DAG: value {{[0-9]+}} @main #cir.int<2> : !cir.int<s, 32> load(cir.alloca 'x'@rt-strong.c:{{.*}})
// TABLE-DAG: value {{[0-9]+}} @main #cir.int<30> : !cir.int<s, 32> load(cir.alloca 'y'@rt-strong.c:{{.*}})
// TABLE-DAG: value {{[0-9]+}} @main #cir.int<4> : !cir.int<s, 32> load(cir.alloca 'z'@rt-strong.c:{{.*}})
// TABLE-DAG: value {{[0-9]+}} @main #cir.int<7> : !cir.int<s, 32> load(cir.get_member{{.*}})
// TABLE-NOT: #cir.int<3> : !cir.int<s, 32> load(cir.alloca 'x'
// TABLE-NOT: #cir.int<20> : !cir.int<s, 32> load(cir.alloca 'y'
// TABLE-NOT: #cir.int<8> : !cir.int<s, 32> load(cir.alloca 'w'
// A callee's read of the caller's local during the call (bu, in the calling
// context): the value stored before the call, not the one after it; and two
// levels down, a parameter spilled by mid and read by peek2.
// BU-DAG: value {{[0-9]+}} @peek #cir.int<13> : !cir.int<s, 32> load({{.*}})
// BU-DAG: value {{[0-9]+}} @peek2 #cir.int<15> : !cir.int<s, 32> load({{.*}})
// TABLE: # functions {{.*}} values {{[1-9][0-9]*}}
// The second claim that x is 2 is the wrong one, after the if.
// INJECT: value {{[0-9]+}} @main #cir.int<2> : !cir.int<s, 32> load(cir.alloca 'x'
// INJECT-NEXT: value {{[0-9]+}} @main #cir.int<2> : !cir.int<s, 32> load(cir.alloca 'x'
//
// CLEAN-NOT: PT-VALUE-VIOLATION
// CLEAN: PT-VALUE-CHECKS {{[1-9][0-9]*}}
// CANARY: PT-VALUE-VIOLATION

struct S {
  int a, b;
};

static void set9(int *p) { *p = 9; }
static int peek(int *p) { return *p; }
static int peek2(int *p) { return *p; }
static int mid(int n) { return peek2(&n); }

int main(int argc, char **argv) {
  (void)argv;
  int r = 0;
  int x = 1;
  r += x;
  x = 2;
  r += x;
  if (argc > 0)
    x = 3;
  r += x;
  int y = 10;
  for (int i = 0; i < argc + 2; ++i) {
    r += y;
    y = 20;
  }
  r += y;
  y = 30;
  r += y;
  int z = 4;
  {
    r += z;
  }
  struct S s;
  s.a = 5;
  s.b = 6;
  s.a = 7;
  r += s.a;
  int w = 8;
  set9(&w);
  r += w;
  int q = 13;
  r += peek(&q);
  q = 14;
  r += q;
  r += mid(15);
  return r == 0;
}
