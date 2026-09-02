// RUN: %clang_cc1 -cl-std=CL2.0 -triple spir64 -fclangir -emit-llvm -O0 %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -cl-std=CL2.0 -triple spir64 -emit-llvm -O0 %s -o %t.ogcg.ll
// RUN: FileCheck --input-file=%t.ogcg.ll %s --check-prefix=OGCG

// Pointer arguments of an OpenCL/SYCL device kernel carry an `align`
// attribute derived from the pointee type's natural alignment (OpenCL spec
// v3.0.10 section 6.3.5). This matters for local pointer arguments whose
// pointee is over-aligned (e.g. a `double4`): without the alignment the
// SPIR-V/IGC backend can lay out the local memory under-aligned and silently
// drop a subsequent aligned local atomic. clangir must match classic CodeGen
// (OGCG) here.

typedef double double4 __attribute__((ext_vector_type(4)));

// A __global char* pointee is naturally 1-byte aligned.
// A __local double4* pointee is naturally 32-byte aligned.

__kernel void k(__global char *gp, __local double4 *lp) {
  *lp = (double4)(*gp);
}

// LLVM: define {{.*}}spir_kernel void @k(ptr addrspace(1) noundef align 1 %{{.*}}, ptr addrspace(3) noundef align 32 %{{.*}})
// OGCG: define {{.*}}spir_kernel void @k(ptr addrspace(1) noundef align 1 %{{.*}}, ptr addrspace(3) noundef align 32 %{{.*}})
