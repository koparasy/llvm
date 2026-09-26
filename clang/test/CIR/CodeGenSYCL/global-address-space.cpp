// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown \
// RUN:   -Wno-deprecated-attributes -fclangir -emit-cir \
// RUN:   -mmlir --mlir-print-ir-before=cir-target-lowering \
// RUN:   %s -o %t.cir 2> %t.pre.cir
// RUN: FileCheck %s --check-prefix=CIR-PRE --input-file=%t.pre.cir
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown \
// RUN:   -Wno-deprecated-attributes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s --check-prefix=LLVM --input-file=%t-cir.ll
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown \
// RUN:   -Wno-deprecated-attributes -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --check-prefix=LLVM --input-file=%t.ll

// In SYCL device code, variables with static storage and no address space
// qualifier live in the global address space, both at namespace scope and as
// static locals. Their uses are cast to the generic address space.

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel_entry_point(KernelName)]]
void kernel_single_task(KernelType kf) { kf(); }
template <typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}
struct KN;

union U {
  int i;
  float f;
};

const int table[4] = {1, 2, 3, 4};
constexpr U cu = {.f = 1.0f};

// CIR-PRE-DAG: cir.global "private" constant internal dso_local lang_address_space(offload_global) @_ZL5table = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i]> : !cir.array<!s32i x 4>
// CIR-PRE-DAG: cir.global "private" constant internal dso_local lang_address_space(offload_global) @_ZL2cu = #cir.const_record<{#cir.fp<1.000000e+00> : !cir.float}> : !rec_U
// CIR-PRE-DAG: cir.global "private" constant internal dso_local lang_address_space(offload_global) @_ZZ3getiE5local = #cir.const_array<[#cir.int<7> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i]> : !cir.array<!s32i x 3>
// CIR-PRE-DAG: cir.global "private" constant internal dso_local lang_address_space(offload_global) @_ZZ3getiE1u = #cir.const_record<{#cir.fp<2.000000e+00> : !cir.float}> : !rec_U

// LLVM-DAG: @_ZL5table = internal addrspace(1) constant [4 x i32] [i32 1, i32 2, i32 3, i32 4]
// LLVM-DAG: @_ZL2cu = internal addrspace(1) constant { float } { float 1.000000e+00 }
// LLVM-DAG: @_ZZ3getiE5local = internal addrspace(1) constant [3 x i32] [i32 7, i32 8, i32 9]
// LLVM-DAG: @_ZZ3getiE1u = internal addrspace(1) constant { float } { float 2.000000e+00 }

int get(int i) {
  static const int local[3] = {7, 8, 9};
  static constexpr U u = {.f = 2.0f};
  const int *p = &table[i];
  return local[i] + *p + u.i + cu.i;
}

// CIR-PRE-LABEL: cir.func {{.*}} @_Z3geti
// CIR-PRE:         %[[LOCAL:.*]] = cir.get_global @_ZZ3getiE5local : !cir.ptr<!cir.array<!s32i x 3>, lang_address_space(offload_global)>
// CIR-PRE:         cir.cast address_space %[[LOCAL]] : !cir.ptr<!cir.array<!s32i x 3>, lang_address_space(offload_global)> -> !cir.ptr<!cir.array<!s32i x 3>, lang_address_space(offload_generic)>
// CIR-PRE:         %[[U:.*]] = cir.get_global @_ZZ3getiE1u : !cir.ptr<!rec_U, lang_address_space(offload_global)>
// CIR-PRE:         cir.cast address_space %[[U]] : !cir.ptr<!rec_U, lang_address_space(offload_global)> -> !cir.ptr<!rec_U, lang_address_space(offload_generic)>
// CIR-PRE:         %[[TABLE:.*]] = cir.get_global @_ZL5table : !cir.ptr<!cir.array<!s32i x 4>, lang_address_space(offload_global)>
// CIR-PRE:         cir.cast address_space %[[TABLE]] : !cir.ptr<!cir.array<!s32i x 4>, lang_address_space(offload_global)> -> !cir.ptr<!cir.array<!s32i x 4>, lang_address_space(offload_generic)>

// LLVM-LABEL: define {{.*}} i32 @_Z3geti(
// LLVM:         getelementptr {{.*}}[4 x i32], ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZL5table to ptr addrspace(4))
// LLVM:         getelementptr {{.*}}[3 x i32], ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZZ3getiE5local to ptr addrspace(4))
// LLVM:         load i32, ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZZ3getiE1u to ptr addrspace(4))
// LLVM:         load i32, ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZL2cu to ptr addrspace(4))

void test(int *out) {
  kernel_single_task<KN>([out]() { *out = get(1); });
}
