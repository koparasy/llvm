// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t-ogcg.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t-ogcg.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -fwrapv
// RUN: FileCheck --check-prefix=CIR_NO_POISON --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -fwrapv
// RUN: FileCheck --check-prefix=LLVM_NO_POISON --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t-ogcg-wrapv.ll -fwrapv
// RUN: FileCheck --check-prefix=OGCG_NO_POISON --input-file=%t-ogcg-wrapv.ll %s

// Note: memcpy/memmove are lowered to dedicated CIR ops (matching classic
// CodeGen's llvm.memcpy/llvm.memmove intrinsics). memset is still emitted as a
// normal function call until its special handling is implemented.

void *memcpy(void *, const void *, unsigned long);
void testMemcpy(void *dst, const void *src, unsigned long size) {
  memcpy(dst, src, size);
  // CHECK: cir.libc.memcpy
}

void *mempcpy(void *, const void *, unsigned long);
void *testMempcpy(void *dst, const void *src, unsigned long size) {
  return mempcpy(dst, src, size);
  // CHECK: cir.libc.memcpy
  // CHECK: cir.ptr_stride
}

void *memmove(void *, const void *, unsigned long);
void testMemmove(void *src, const void *dst, unsigned long size) {
  memmove(dst, src, size);
  // CHECK: cir.libc.memmove
  // LLVM: call void @llvm.memmove.p0.p0.i64
  // OGCG: call void @llvm.memmove.p0.p0.i64
}

void *memset(void *, int, unsigned long);
void testMemset(void *dst, int val, unsigned long size) {
  memset(dst, val, size);
  // CHECK: cir.call @memset
}

double fabs(double);
double testFabs(double x) {
  return fabs(x);
  // CHECK: cir.fabs %{{.+}} : !cir.double
}

float fabsf(float);
float testFabsf(float x) {
  return fabsf(x);
  // CHECK: cir.fabs %{{.+}} : !cir.float
}

int abs(int);
int testAbs(int x) {
  return abs(x);
  // CHECK: cir.abs %{{.+}} min_is_poison : !s32i
  // LLVM: %{{.+}} = call i32 @llvm.abs.i32(i32 %{{.+}}, i1 true)
  // OGCG: %{{.+}} = call i32 @llvm.abs.i32(i32 %{{.+}}, i1 true)
  // CIR_NO_POISON: cir.abs %{{[^ ]+}} : !s32i
  // LLVM_NO_POISON: %{{.+}} = call i32 @llvm.abs.i32(i32 %{{.+}}, i1 false)
  // OGCG_NO_POISON: %{{.+}} = call i32 @llvm.abs.i32(i32 %{{.+}}, i1 false)
}

long labs(long);
long testLabs(long x) {
  return labs(x);
  // CHECK: cir.abs %{{.+}} min_is_poison : !s64i
  // LLVM: %{{.+}} = call i64 @llvm.abs.i64(i64 %{{.+}}, i1 true)
  // OGCG: %{{.+}} = call i64 @llvm.abs.i64(i64 %{{.+}}, i1 true)
  // CIR_NO_POISON: cir.abs %{{[^ ]+}} : !s64i
  // LLVM_NO_POISON: %{{.+}} = call i64 @llvm.abs.i64(i64 %{{.+}}, i1 false)
  // OGCG_NO_POISON: %{{.+}} = call i64 @llvm.abs.i64(i64 %{{.+}}, i1 false)
}

long long llabs(long long);
long long testLlabs(long long x) {
  return llabs(x);
  // CHECK: cir.abs %{{.+}} min_is_poison : !s64i
  // LLVM: %{{.+}} = call i64 @llvm.abs.i64(i64 %{{.+}}, i1 true)
  // OGCG: %{{.+}} = call i64 @llvm.abs.i64(i64 %{{.+}}, i1 true)
  // CIR_NO_POISON: cir.abs %{{[^ ]+}} : !s64i
  // LLVM_NO_POISON: %{{.+}} = call i64 @llvm.abs.i64(i64 %{{.+}}, i1 false)
  // OGCG_NO_POISON: %{{.+}} = call i64 @llvm.abs.i64(i64 %{{.+}}, i1 false)
}
