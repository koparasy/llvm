// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// An empty C++ class is a padded struct whose only member is a pad byte.  Its
// single member is padding rather than data, so it maps to no fields and
// classifies Ignore -- the argument is dropped from the signature entirely.
struct Empty {};

void take_empty(Empty e) { (void)e; }

// CIR: cir.func {{.*}}@_Z10take_empty5Empty()
// LLVM: define dso_local void @_Z10take_empty5Empty()

// Returning an empty class is likewise ignored: the function returns void.
Empty ret_empty() { return {}; }

// CIR: cir.func {{.*}}@_Z9ret_emptyv()
// LLVM: define dso_local void @_Z9ret_emptyv()

// A packed struct places its int at a byte offset its natural alignment does
// not allow, so the classifier sends the whole struct to memory -- passed
// indirectly by value, exactly as classic CodeGen does.
struct __attribute__((packed)) Packed { char c; int i; };

void take_packed(Packed p) { (void)p; }

// CIR: cir.func {{.*}}@_Z11take_packed6Packed(%arg0: !cir.ptr<!rec_Packed> {{.*}}llvm.byval = !rec_Packed
// LLVM: define dso_local void @_Z11take_packed6Packed(ptr {{.*}}byval(%struct.Packed) align 8 %{{.+}})

// A struct with inter-field padding is not marked padded in CIR (the natural
// struct layout already places the int at offset 4), so it is the control:
// both eightbyte halves land in one INTEGER register.
struct CharInt { char c; int i; };

void take_charint(CharInt s) { (void)s; }

// CIR: cir.func {{.*}}@_Z12take_charint7CharInt(%arg0: !u64i{{.*}})
// LLVM: define dso_local void @_Z12take_charint7CharInt(i64 %{{.+}})
