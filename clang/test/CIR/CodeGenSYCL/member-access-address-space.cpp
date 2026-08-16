// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fno-gpu-rdc -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -triple spir64-unknown-unknown \
// RUN:   -fno-gpu-rdc -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM

// Under -fsycl-is-device on SPIR(-V), the object pointer of a member function
// lives in the generic address space (addrspace(4)). A member of that object
// (cir.get_member) and the pointer produced by decaying an array member
// (cir.cast array_to_ptrdecay) must inherit that address space. If they were
// emitted in the default address space instead, downstream lowering would try
// to bitcast ptr addrspace(4) -> ptr addrspace(0), which is an illegal
// cross-address-space bitcast (OG emits an addrspacecast / keeps the AS).

struct arr {
  unsigned long d[1];
  unsigned long &operator[](int i) { return d[i]; }
  arr(unsigned long d0) : d{d0} {}
};
struct id : arr {};
struct lam {
  id idx;
  unsigned long run() { return idx[0]; }
};

__attribute__((sycl_device))
unsigned long use(lam l) { return l.run(); }

// get_member on a generic-AS object keeps the member pointer in addrspace(4),
// so base_class_addr and the call operate on generic pointers -- no cross-AS
// bitcast is synthesized.
// CIR-LABEL: cir.func {{.*}}@_ZN3lam3runEv
// CIR: cir.get_member {{.*}} : !cir.ptr<!rec_lam, target_address_space(4)> -> !cir.ptr<!rec_id, target_address_space(4)>
// CIR: cir.base_class_addr {{.*}} : !cir.ptr<!rec_id, target_address_space(4)> -> !cir.ptr<!rec_arr, target_address_space(4)>

// The member access lowers to a GEP that stays in addrspace(4); there is no
// illegal bitcast across address spaces.
// LLVM-LABEL: define {{.*}}@_ZN3lam3runEv
// LLVM: getelementptr {{.*}}ptr addrspace(4)

__attribute__((sycl_device))
void make(unsigned long v) { arr a(v); (void)a; }

// The array-member initializer decays through a generic-AS pointer, so the
// array_to_ptrdecay cast result also stays in addrspace(4).
// CIR-LABEL: cir.func {{.*}}@_ZN3arrC
// CIR: cir.cast array_to_ptrdecay {{.*}} : !cir.ptr<!cir.array<!u64i x 1>, target_address_space(4)> -> !cir.ptr<!u64i, target_address_space(4)>
