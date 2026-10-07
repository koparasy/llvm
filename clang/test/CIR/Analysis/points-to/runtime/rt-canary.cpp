// Negative control of the run-time points-to checks: p and q both point to
// x. The analysis gets it right (MustAlias, so no check), and
// inject-false-noalias forces a check on that pair as if it were NoAlias.
// The run must abort with PT-VIOLATION, which shows the harness catches a
// wrong NoAlias. Without the injection the same program runs clean.
//
// REQUIRES: native, system-linux, target=x86_64{{.*}}
// RUN: %clangxx -fclangir -emit-cir -O0 %s -o %t.cir
// RUN: %clang -c %S/../Inputs/pt_check_runtime.c -o %t.rt.o
//
// Candidate 24 is the pair of stores *p = 1 and *q = 2.
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.cand print-candidates=1' -o %t.plain.cir
// RUN: FileCheck %s --check-prefix=CAND --input-file=%t.cand
// CAND: cand 24 @main MustAlias same-address store(cir.load@rt-canary.cpp:{{[0-9]+}}:4) 4 | store(cir.load@rt-canary.cpp:{{[0-9]+}}:4) 4
// CAND: # functions 1 candidates 49 noalias 44 checks 44
//
// RUN: cir-translate --cir-to-llvmir %t.plain.cir -o %t.plain.ll
// RUN: grep -v '^target datalayout' %t.plain.ll > %t.plain.fixed.ll
// RUN: %clangxx %t.plain.fixed.ll %t.rt.o -o %t.plain.exe
// RUN: %t.plain.exe 2>&1 | FileCheck %s --check-prefix=CLEAN
// CLEAN-NOT: PT-VIOLATION
// CLEAN: PT-CHECKS 44
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.table inject-false-noalias=24' -o %t.inject.cir
// RUN: FileCheck %s --check-prefix=TABLE --input-file=%t.table
// TABLE: 24 @main injected(really MustAlias same-address) store({{.*}}) 4 | store({{.*}}) 4 at cir.store@rt-canary.cpp:{{.*}} | a, b: the same address of one object
// TABLE: # rule injected 1
// RUN: cir-translate --cir-to-llvmir %t.inject.cir -o %t.inject.ll
// RUN: grep -v '^target datalayout' %t.inject.ll > %t.inject.fixed.ll
// RUN: %clangxx %t.inject.fixed.ll %t.rt.o -o %t.inject.exe
// RUN: not --crash %t.inject.exe 2>&1 | FileCheck %s --check-prefix=CANARY
// CANARY: PT-VIOLATION 24
// CANARY-NOT: PT-CHECKS

int main() {
  int x = 0;
  int *p = &x;
  int *q = &x;
  *p = 1;
  *q = 2;
  return x == 2 ? 0 : 1;
}
