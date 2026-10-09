// Run-time check of the std::thread model (T27): the thread's _M_run and the
// state's deleting destructor are virtual calls the model makes at
// _M_start_thread, and everything the state reaches is AsyncAccess, since
// the thread runs concurrently with its creator. The thread waits for the
// creator's `flag`, then reads x (stored by the creator after the spawn) and
// writes y (stored by the creator after the spawn too); the creator reads y
// after join. No load may be claimed constant with a value it does not hold,
// in ci and in bu mode with closed-tu. The negative control:
// inject-no-concurrent=1 drops the AsyncAccess rule, so the thread's read of
// x is claimed to be the 1 stored before the spawn (a strong update at the
// spawn), and the run must abort with PT-VALUE-VIOLATION.
//
// REQUIRES: native, system-linux, target=x86_64{{.*}}
// RUN: %clangxx -fclangir -emit-cir -O0 %s -o %t.cir
// RUN: %clang -c %S/../Inputs/pt_check_runtime.c -o %t.rt.o
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.ci.table values=1 closed-tu=1' -o %t.ci.cir
// RUN: FileCheck %s --check-prefix=TABLE --input-file=%t.ci.table
// RUN: cir-translate --cir-to-llvmir %t.ci.cir -o %t.ci.ll
// RUN: grep -v '^target datalayout' %t.ci.ll > %t.ci.fixed.ll
// RUN: %clangxx %t.ci.fixed.ll %t.rt.o -pthread -o %t.ci.exe
// RUN: %t.ci.exe 2>&1 | FileCheck %s --check-prefix=CLEAN
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.bu.table values=1 mode=bu closed-tu=1 subobjects=1' -o %t.bu.cir
// RUN: FileCheck %s --check-prefix=TABLE --input-file=%t.bu.table
// RUN: cir-translate --cir-to-llvmir %t.bu.cir -o %t.bu.ll
// RUN: grep -v '^target datalayout' %t.bu.ll > %t.bu.fixed.ll
// RUN: %clangxx %t.bu.fixed.ll %t.rt.o -pthread -o %t.bu.exe
// RUN: %t.bu.exe 2>&1 | FileCheck %s --check-prefix=CLEAN
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.inject.table values=1 mode=bu closed-tu=1 subobjects=1 inject-no-concurrent=1' -o %t.inject.cir
// RUN: FileCheck %s --check-prefix=INJECT --input-file=%t.inject.table
// RUN: cir-translate --cir-to-llvmir %t.inject.cir -o %t.inject.ll
// RUN: grep -v '^target datalayout' %t.inject.ll > %t.inject.fixed.ll
// RUN: %clangxx %t.inject.fixed.ll %t.rt.o -pthread -o %t.inject.exe
// RUN: not --crash %t.inject.exe 2>&1 | FileCheck %s --check-prefix=CANARY
//
// Nothing the thread or its creator shares is claimed constant.
// TABLE-NOT: value {{[0-9]+}} @main {{.*}}load(cir.alloca '{{x|y|seen|r}}'
// TABLE-NOT: value {{[0-9]+}} @_ZZ4mainENK3$_0clEv
// TABLE: # functions
// INJECT: value {{[0-9]+}} @_ZZ4mainENK3$_0clEv #cir.int<1> : !cir.int<s, 32> load(cir.load@rt-thread.cpp:{{[0-9]+}}:12)
// CLEAN: PT-CHECKS {{[1-9][0-9]*}}
// CLEAN-NOT: VIOLATION
// CANARY: PT-VALUE-VIOLATION

#include <atomic>
#include <cstdio>
#include <thread>

int main(int argc, char **argv) {
  std::atomic<int> flag{0};
  int y = 0;
  int x = 1;
  int seen = 0;
  std::thread t([&] {
    while (flag.load() == 0) {
    }
    seen = x;
    y = 5;
  });
  x = 2;
  y = 3;
  flag.store(1);
  t.join();
  int r = y;
  printf("%d %d\n", r, seen);
  return r == 5 && seen == 2 ? 0 : 1;
}
