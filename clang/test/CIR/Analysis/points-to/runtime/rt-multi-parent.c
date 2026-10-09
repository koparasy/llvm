// Run-time regression of Csmith seed 407679 (T24/G3): with subobject nodes,
// a pointer that may point to l.f1 or to l.f3.a merges the two struct members
// into one node with two parents. The read of l.f3.a.y is then at two places
// of l, and the copy of l's initializer contains it at both: the stored
// values must be the union (1 and 2), not the value at the first place (1,
// which is l.f1.y: PT-VALUE-VIOLATION before the fix). Checked in ci and
// bu+closed-tu, with the indeterminate-value rule on and off.
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
// RUN: %t.ci.exe 2>&1 | FileCheck %s --check-prefix=CLEAN --allow-empty
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.bu.table values=1 subobjects=1 mode=bu closed-tu=1' -o %t.bu.cir
// RUN: FileCheck %s --check-prefix=TABLE --input-file=%t.bu.table
// RUN: cir-translate --cir-to-llvmir %t.bu.cir -o %t.bu.ll
// RUN: grep -v '^target datalayout' %t.bu.ll > %t.bu.fixed.ll
// RUN: %clang %t.bu.fixed.ll %t.rt.o -o %t.bu.exe
// RUN: %t.bu.exe 2>&1 | FileCheck %s --check-prefix=CLEAN --allow-empty
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.off.table values=1 subobjects=1 mode=bu closed-tu=1 indeterminate-ub=0' -o %t.off.cir
// RUN: FileCheck %s --check-prefix=TABLE --input-file=%t.off.table
// RUN: cir-translate --cir-to-llvmir %t.off.cir -o %t.off.ll
// RUN: grep -v '^target datalayout' %t.off.ll > %t.off.fixed.ll
// RUN: %clang %t.off.fixed.ll %t.rt.o -o %t.off.exe
// RUN: %t.off.exe 2>&1 | FileCheck %s --check-prefix=CLEAN --allow-empty
//
// The read of l.f3.a.y (2 at run time) is not claimed to be 1.
// TABLE-NOT: value {{.*}} #cir.int<1> : !cir.int<u, 32>
// TABLE: # functions
// CLEAN-NOT: PT-VALUE-VIOLATION

struct T0 {
  long a;
  unsigned y;
};
struct T1 {
  struct T0 a;
  unsigned x;
};
struct T2 {
  struct T0 f1;
  struct T1 f3;
};

static unsigned multiParent(int c) {
  struct T2 l = {{7, 1}, {{8, 2}, 9}};
  struct T0 *p = c ? &l.f1 : &l.f3.a;
  return l.f3.a.y + p->a;
}

int main(int argc, char **argv) {
  (void)argv;
  return multiParent(argc) == 0;
}
