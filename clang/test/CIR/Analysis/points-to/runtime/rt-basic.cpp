// Run-time check of the points-to NoAlias facts on a small program: locals,
// fields, heap objects, two factories, and pointers stored in and loaded
// from globals. Every pair the analysis calls NoAlias is checked
// (max-pairs-per-func=0), open and closed-tu; none may overlap.
//
// REQUIRES: native, system-linux, target=x86_64{{.*}}
// RUN: %clangxx -fclangir -emit-cir -O0 %s -o %t.cir
// RUN: %clang -c %S/../Inputs/pt_check_runtime.c -o %t.rt.o
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.open.table max-pairs-per-func=0' -o %t.open.cir
// RUN: FileCheck %s --check-prefix=TABLE --input-file=%t.open.table
// RUN: cir-translate --cir-to-llvmir %t.open.cir -o %t.open.ll
// RUN: grep -v '^target datalayout' %t.open.ll > %t.open.fixed.ll
// RUN: %clangxx %t.open.fixed.ll %t.rt.o -o %t.open.exe
// RUN: %t.open.exe 2>&1 | FileCheck %s --check-prefix=EXEC
//
// RUN: cir-opt %t.cir -test-cir-points-to-instrument='table=%t.closed.table max-pairs-per-func=0 closed-tu=1' -o %t.closed.cir
// RUN: FileCheck %s --check-prefix=TABLE --input-file=%t.closed.table
// RUN: cir-translate --cir-to-llvmir %t.closed.cir -o %t.closed.ll
// RUN: grep -v '^target datalayout' %t.closed.ll > %t.closed.fixed.ll
// RUN: %clangxx %t.closed.fixed.ll %t.rt.o -o %t.closed.exe
// RUN: %t.closed.exe 2>&1 | FileCheck %s --check-prefix=EXEC
//
// The fields, the factory results and the globals are among the checked
// pairs (the TABLE-DAG lines next to the code); the totals are at the end.
//
// EXEC-NOT: PT-VIOLATION
// EXEC: PT-CHECKS {{[1-9][0-9]*}}

struct Pair {
  int a;
  int b;
};

struct Node {
  int v;
  Node *next;
};

static int g1, g2;
static int *gp;
static int *gq;

static int *make_a() { return new int(1); }
static int *make_b() { return new int(2); }

static void fill(Pair *p) {
  p->a = 3;
  p->b = 4;
}

static int sum(Node *n) {
  int s = 0;
  for (; n; n = n->next)
    s += n->v;
  return s;
}

int main() {
  int x = 0, y = 0;
  int *px = &x, *py = &y;
  *px = 1;
  *py = 2;

  Pair pr;
  fill(&pr);
  int *pa = &pr.a, *pb = &pr.b;
  *pa += *pb;
  // TABLE-DAG: @main disjoint-ranges load(cir.load@rt-basic.cpp:[[#@LINE-1]]:11) 4 | load(cir.load@rt-basic.cpp:[[#@LINE-1]]:4) 4

  int *h1 = new int(5), *h2 = new int(6);
  *h1 += *h2;

  int *f1 = make_a(), *f2 = make_b();
  *f1 += *f2;
  // TABLE-DAG: @main distinct-objects load(cir.load@rt-basic.cpp:[[#@LINE-1]]:11) 4 | store(cir.load@rt-basic.cpp:[[#@LINE-1]]:4) 4

  gp = &g1;
  gq = &g2;
  *gp = 7;
  *gq = *gp + x;
  // TABLE-DAG: @main distinct-objects load(cir.load@rt-basic.cpp:[[#@LINE-1]]:10) 4 | store(cir.load@rt-basic.cpp:[[#@LINE-1]]:4) 4

  Node n2{2, nullptr}, n1{1, &n2};
  int s = sum(&n1);

  int r = x + y + pr.a + *h1 + *f1 + g2 + s;
  delete h1;
  delete h2;
  delete f1;
  delete f2;
  return r == 1 + 2 + 7 + 11 + 3 + 8 + 3 ? 0 : 1;
}

// TABLE: # functions 5 candidates {{[0-9]+}} noalias [[N:[0-9]+]] checks [[N]]
