// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll --check-prefix=LLVM %s

// The defaulted copy/move assignment operator of a union has an empty
// synthesized body. CIRGen mirrors classic codegen by producing the effect of
// the trivial assignment at the call site via an aggregate copy, so the
// operator is never called or emitted.

union U {
  void *p;
  int i;
};

// The union assignment operator must not be emitted.
// CIR-NOT: cir.func {{.*}}aSER
// CIR-NOT: cir.func {{.*}}aSEO

// CIR-LABEL: cir.func {{.*}}@_Z11copy_assignR1US0_
// CIR-NOT: cir.call {{.*}}aSER
// CIR: cir.copy {{.*}} : !cir.ptr<!rec_U>
void copy_assign(U &a, U &b) { a = b; }

// CIR-LABEL: cir.func {{.*}}@_Z11move_assignR1US0_
// CIR-NOT: cir.call {{.*}}aSEO
// CIR: cir.copy {{.*}} : !cir.ptr<!rec_U>
void move_assign(U &a, U &b) { a = static_cast<U &&>(b); }

// LLVM-LABEL: define {{.*}}@_Z11copy_assignR1US0_
// LLVM: call void @llvm.memcpy.{{.*}}(ptr align 8 %{{.*}}, ptr align 8 %{{.*}}, i64 8, i1 false)
// LLVM-LABEL: define {{.*}}@_Z11move_assignR1US0_
// LLVM: call void @llvm.memcpy.{{.*}}(ptr align 8 %{{.*}}, ptr align 8 %{{.*}}, i64 8, i1 false)
