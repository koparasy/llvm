// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fno-gpu-rdc -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fno-gpu-rdc -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM

// Work-group shared-local-memory (SPIR-V Workgroup / addrspace(3)) accesses and
// the SPIR-V control barrier must survive CIR -> LLVM device lowering. Trivial
// struct/array copies in such kernels lower through the memcpy builtin: emitting
// a *library* `memcpy` call (instead of the llvm.memcpy intrinsic) resolves to
// the SPIR device-library `memcpy` (a `spir_func` definition) once linked, which
// the optimizer folds to `unreachable`, silently deleting the surrounding SLM +
// barrier body. Lower the builtin to the intrinsic (matching classic CodeGen) so
// the body is preserved.

__attribute__((sycl_device)) void __spirv_ControlBarrier(int, int, int);

__attribute__((sycl_device))
void slm_reduce(__attribute__((opencl_local)) int *slm,
                __attribute__((opencl_global)) int *g,
                int *dst, int *src, unsigned long n) {
  __builtin_memcpy(dst, src, n);
  int lid = g[0];
  slm[lid] = dst[0];
  __spirv_ControlBarrier(2, 2, 272);
  g[0] = slm[lid];
}

// The memcpy builtin lowers to cir.libc.memcpy, never a library `@memcpy` call.
// CIR-NOT: cir.call @memcpy
// CIR: cir.func {{.*}}@_Z10slm_reduce
// CIR:   cir.libc.memcpy {{.*}} : !u64i, !cir.ptr<!void, target_address_space(4)> -> !cir.ptr<!void, target_address_space(4)>
// CIR:   cir.store {{.*}} : !s32i, !cir.ptr<!s32i, target_address_space(3)>
// CIR:   cir.call @_Z22__spirv_ControlBarrieriii({{.*}}) {{.*}}: (!s32i {{.*}}, !s32i {{.*}}, !s32i {{.*}}) -> ()
// CIR:   cir.load {{.*}} : !cir.ptr<!s32i, target_address_space(3)>, !s32i

// LLVM-NOT: call {{.*}} @memcpy(
// LLVM: define {{.*}} void @_Z10slm_reduce
// LLVM:   call void @llvm.memcpy.p4.p4.i64(
// LLVM:   store i32 {{.*}}, ptr addrspace(3)
// LLVM:   call spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef 2, i32 noundef 2, i32 noundef 272)
// LLVM:   load i32, ptr addrspace(3)
