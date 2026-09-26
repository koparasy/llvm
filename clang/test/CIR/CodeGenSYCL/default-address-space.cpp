// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown \
// RUN:   -Wno-deprecated-attributes -fclangir -emit-cir \
// RUN:   -mmlir --mlir-print-ir-before=cir-target-lowering \
// RUN:   %s -o %t.cir 2> %t.pre.cir
// RUN: FileCheck %s --check-prefix=CIR-PRE --input-file=%t.pre.cir
// RUN: FileCheck %s --check-prefix=CIR --input-file=%t.cir
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown \
// RUN:   -Wno-deprecated-attributes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s --check-prefix=LLVM --input-file=%t-cir.ll
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown \
// RUN:   -Wno-deprecated-attributes -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --check-prefix=LLVM --input-file=%t.ll

// In SYCL device code an unqualified pointer is a generic pointer. CIRGen
// spells it offload_generic, and target lowering rewrites it, including the
// pointer members of records, to the target's generic address space.

struct Node {
  Node *next;
  int *data;
};

union U {
  int *p;
  long l;
};

// CIR-PRE-DAG: !rec_Node = !cir.struct<"Node" {data !cir.ptr<!cir.struct<"Node">, lang_address_space(offload_generic)>, data !cir.ptr<!s32i, lang_address_space(offload_generic)>}>
// CIR-PRE-DAG: !rec_U = !cir.union<"U" {data !cir.ptr<!s32i, lang_address_space(offload_generic)>, data !s64i}>

// CIR-DAG: !rec_Node = !cir.struct<"Node" {data !cir.ptr<!cir.struct<"Node">, target_address_space(4)>, data !cir.ptr<!s32i, target_address_space(4)>}>
// CIR-DAG: !rec_U = !cir.union<"U" {data !cir.ptr<!s32i, target_address_space(4)>, data !s64i}>

// LLVM-DAG: %struct.Node = type { ptr addrspace(4), ptr addrspace(4) }
// LLVM-DAG: %union.U = type { ptr addrspace(4) }

template <typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}
template <typename KernelName, typename KernelType>
[[clang::sycl_kernel_entry_point(KernelName)]]
void kernel_single_task(KernelType kf) { kf(); }
struct KN;

int follow(Node *p, bool c) {
  Node local = *p;
  U u = {p->data};
  int *q = c ? local.data : u.p;
  return *q;
}

// A member is addressed in the address space of the record containing it.
// CIR-PRE-LABEL: cir.func {{.*}} @_Z6followP4Nodeb(%arg0: !cir.ptr<!rec_Node, lang_address_space(offload_generic)>
// CIR-PRE:         cir.get_member %{{.*}}[0] {name = "p"} : !cir.ptr<!rec_U, lang_address_space(offload_generic)> -> !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_generic)>, lang_address_space(offload_generic)>
// CIR-PRE:         cir.get_member %{{.*}}[1] {name = "data"} : !cir.ptr<!rec_Node, lang_address_space(offload_generic)> -> !cir.ptr<!cir.ptr<!s32i, lang_address_space(offload_generic)>, lang_address_space(offload_generic)>

// CIR-LABEL: cir.func {{.*}} @_Z6followP4Nodeb(%arg0: !cir.ptr<!rec_Node, target_address_space(4)>
// CIR:         cir.get_member %{{.*}}[0] {name = "p"} : !cir.ptr<!rec_U, target_address_space(4)> -> !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>, target_address_space(4)>

// LLVM-LABEL: define {{.*}} i32 @_Z6followP4Nodeb(ptr addrspace(4)
// LLVM:         getelementptr inbounds nuw %struct.Node, ptr addrspace(4) %{{.*}}, i32 0, i32 1
// LLVM:         load ptr addrspace(4), ptr addrspace(4)
// LLVM:         store ptr addrspace(4) %{{.*}}, ptr addrspace(4)

void test(Node *p) {
  kernel_single_task<KN>([p]() { follow(p, true); });
}
