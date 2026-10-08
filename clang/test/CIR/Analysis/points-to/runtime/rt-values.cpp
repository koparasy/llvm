// Run-time check of the stored-value constants (T22): every load that
// PointsToAnalysis::isConstantAt calls constant is checked against the
// constant (values=1), in ci and in bu mode with closed-tu; none may differ.
// The negative control: inject-wrong-value checks value 0 against the
// constant plus one, and the run must abort with PT-VALUE-VIOLATION.
//
// REQUIRES: native, system-linux, target=x86_64{{.*}}
// RUN: %clangxx -fclangir -emit-cir -O0 %s -o %t.cir
// RUN: %clang -c %S/../Inputs/pt_check_runtime.c -o %t.rt.o
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.ci.table values=1' -o %t.ci.cir
// RUN: FileCheck %s --check-prefix=TABLE --input-file=%t.ci.table
// RUN: cir-translate --cir-to-llvmir %t.ci.cir -o %t.ci.ll
// RUN: grep -v '^target datalayout' %t.ci.ll > %t.ci.fixed.ll
// RUN: %clangxx %t.ci.fixed.ll %t.rt.o -o %t.ci.exe
// RUN: %t.ci.exe 2>&1 | FileCheck %s --check-prefix=CLEAN
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.bu.table values=1 mode=bu closed-tu=1 subobjects=1' -o %t.bu.cir
// RUN: FileCheck %s --check-prefixes=TABLE,BU --input-file=%t.bu.table
// RUN: cir-translate --cir-to-llvmir %t.bu.cir -o %t.bu.ll
// RUN: grep -v '^target datalayout' %t.bu.ll > %t.bu.fixed.ll
// RUN: %clangxx %t.bu.fixed.ll %t.rt.o -o %t.bu.exe
// RUN: %t.bu.exe 2>&1 | FileCheck %s --check-prefix=CLEAN
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='values=1 inject-wrong-value=0' -o %t.inject.cir
// RUN: cir-translate --cir-to-llvmir %t.inject.cir -o %t.inject.ll
// RUN: grep -v '^target datalayout' %t.inject.ll > %t.inject.fixed.ll
// RUN: %clangxx %t.inject.fixed.ll %t.rt.o -o %t.inject.exe
// RUN: not --crash %t.inject.exe 2>&1 | FileCheck %s --check-prefix=CANARY
//
// TABLE-DAG: value {{[0-9]+}} @main #cir.int<7> : !cir.int<s, 32> load(cir.get_global @_ZL1g@rt-values.cpp:{{.*}})
// TABLE-DAG: value {{[0-9]+}} @main #cir.int<5> : !cir.int<s, 32> load(cir.get_element@rt-values.cpp:{{.*}})
// TABLE-DAG: value {{[0-9]+}} @main #cir.bool<true> : !cir.bool load(cir.alloca 'flag'@rt-values.cpp:{{.*}})
// The argument of @_ZL4addkii is 5 in both of its calling contexts.
// BU-DAG: value {{[0-9]+}} @_ZL4addkii #cir.int<5> : !cir.int<s, 32> load(cir.alloca 'k'@rt-values.cpp:{{.*}})
// TABLE: # functions {{.*}} values {{[1-9][0-9]*}}
//
// CLEAN-NOT: PT-VALUE-VIOLATION
// CLEAN: PT-VALUE-CHECKS {{[1-9][0-9]*}}
// CANARY: PT-VALUE-VIOLATION 0
// CANARY-NOT: PT-CHECKS

static int g = 7;
static int addk(int x, int k) { return x + k; }

int main(int argc, char **) {
  int r = g;
  int arr[4];
  for (int i = 0; i < 4; ++i)
    arr[i] = 5;
  r += arr[2];
  bool flag = true;
  if (flag)
    r += addk(r, 5) + addk(argc, 5);
  int x = 10;
  int *px = &x;
  *px = argc;
  r += x;
  return r > 0 ? 0 : 1;
}
