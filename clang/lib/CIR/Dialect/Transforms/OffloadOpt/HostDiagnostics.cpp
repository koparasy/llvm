//===- HostDiagnostics.cpp - SYCL host-fact diagnostics -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// cir-offload-host-diagnostics: warnings on SYCL host-side bugs that the host
// facts (SYCLHostFacts, the points-to analysis of the host module) prove at a
// kernel launch. The pass runs in the cir-offload-merge combine pipeline after
// the passes that read the host facts and reads the analysis they share
// (SYCLHostFactsCache); it builds no second one. It changes nothing.
//
// The checks. Each check is an entry of the registry below (kChecks): a tag,
// which every warning it emits carries ("[<tag>]") and which the `checks`
// option (--cir-offload-host-diagnostics=<tags|all|none> of
// cir-offload-merge) selects, a function run on the shared context, and
// whether `all` (the default) includes it: a check off by default runs only
// when named. To add a check, add a function and an entry.
//
// The rule every check follows: warn only on a MUST fact. The node the
// warning is about has known origins of the bad kind only, is complete (no
// unknown, incomplete, may-be-shared, may-overlap or collapsed contribution)
// and the fact holds at every record of the launch, that is in every calling
// context the analysis covers. When unsure, stay silent: a missed bug is
// acceptable, a false positive is not. A warning is at the source location of
// the use, with a note at the allocation, and is emitted once per location
// and text (the kernel copies of internalized launch paths share the source
// of their originals).
//
// sycl-host-ptr-in-kernel. SYCL 2020 sec:usm (programming_interface.adoc):
// a device accesses host memory only through a USM allocation or a buffer
// accessor. "Each type of USM allocation has different rules for where that
// memory is accessible. Attempting to dereference a USM pointer on the host
// or on a device in violation of these rules results in undefined behavior"
// (PI 11265-11268); host allocations "allow devices to directly read and
// write host memory inside of a kernel" and "must also be obtained using SYCL
// routines instead of system allocation routines" (PI 11293-11299), shared
// allocations "must also be obtained using SYCL allocation routines instead
// of the system allocator" (PI 11314-11316). The exception is a device with
// aspect::usm_system_allocations, on which the system allocator (malloc/new)
// "may instead be used" for shared allocations (PI 11359-11368, 3643-3647);
// stack objects and globals are not system allocations. A captured pointer is
// a legal kernel parameter (device copyable, sec:kernel.parameter.passing),
// so nothing else rejects the bug.
//
// The check: at each recorded launch, for each pointer parameter of the
// device kernel that the closure member it initializes (cir.sycl_kernel_param
// kind "pointer"; not accessors, local accessors or scalars), warn when
//  - at every record of the launch (each covered calling context) the member
//    must point into host-only memory of one kind
//    (SYCLHostFacts::hostOnlyMemoryOfClosureMember: a host stack object, a
//    global, or a heap allocation that is not USM), and
//  - the device kernel actually dereferences the parameter (DerefFinder: a
//    load, store, copy or atomic through it, or through a pointer derived
//    from it, in the kernel or a function it calls, following the parameter
//    through the closure object's members). A kernel that only passes the
//    pointer on, compares it or ignores it is exempt.
// The warning is at that access, with a note at the kernel (the lambda or
// functor: the access may be in a device function it calls) and one at the
// allocation.
//
// sycl-host-heap-ptr-in-kernel (off by default; only when named): the same
// for a pointer into a host heap allocation (malloc, calloc, operator new,
// std::allocator; not USM). That is not portable, but valid on a device with
// aspect::usm_system_allocations, where "the system allocator (malloc/new)
// may instead be used" for shared allocations (PI 11359-11368, 3643-3647),
// so it is not a bug on every device; the message says so.
//
// sycl-host-deref-device-usm. SYCL 2020 sec:usm, table.usm.allocs: a device
// allocation is "Allocations in device memory that are not accessible by
// the host" (PI 11283-11288), and "attempting to dereference a USM pointer
// on the host or on a device in violation of these rules results in
// undefined behavior" (PI 11265-11268); the host reaches device memory only
// through the queue and handler memory commands (memcpy, copy, memset,
// fill), which the device runs. The check: every load, store, copy, atomic
// or libc memory op (cir.libc.memcpy/memmove/memset/memchr, and calls of the
// C memcpy/memmove/memset/memcmp declarations) in an analysed host function
// whose address operand must point into device USM memory
// (SYCLHostFacts::deviceUSMOfPointer: in every covered calling context, a
// node every origin of which is a malloc_device / aligned_alloc_device call
// or a sycl::malloc / aligned_alloc call with a constant
// usm::alloc::device). Exempt: the SYCL implementation (namespace sycl or a
// SYCL role, which include the queue and handler memory commands), the
// methods of a kernel's closure type and the functions only they call (a
// kernel's body, which the host never runs), and a function that asks the
// runtime for a pointer's kind (sycl::get_pointer_type: its accesses may be
// guarded by the answer). The warning is at the access, with a note at
// each allocation.
// Limitation: the check is not path-sensitive. The must fact is about which
// memory the pointer points to, not whether the access runs: an access the
// program guards by a runtime test of its own (a flag saying "device
// pointer", a kind check other than sycl::get_pointer_type in the same
// function) still warns, e.g. `if (onDevice) q.memcpy(d, s, n); else
// memcpy(d, s, n);` in a helper only ever given device memory.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToAnalysis.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLHostFacts.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLKernelParams.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"

#include <deque>
#include <map>
#include <set>
#include <tuple>

namespace mlir {
#define GEN_PASS_DECL_OFFLOADHOSTDIAGNOSTICS
#define GEN_PASS_DEF_OFFLOADHOSTDIAGNOSTICS
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace cir;

#define DEBUG_TYPE "cir-offload-host-diagnostics"

namespace {

//===----------------------------------------------------------------------===//
// The context and the registry
//===----------------------------------------------------------------------===//

/// What every check reads, and the warnings emitted so far.
struct DiagnosticContext {
  mlir::ModuleOp container;
  mlir::ModuleOp host;
  const cir::SYCLHostFacts &facts;
  mlir::SymbolTableCollection symbols;
  /// (location, text) of every warning emitted, to emit each once.
  std::set<std::pair<const void *, std::string>> emitted;
  unsigned numWarnings = 0;

  DiagnosticContext(mlir::ModuleOp container, const cir::SYCLHostFacts &facts)
      : container(container),
        host(cir::getOffloadContainerHostModule(container)), facts(facts) {}

  /// A warning \p text at \p loc, unless one was emitted there already.
  std::optional<mlir::InFlightDiagnostic> warn(mlir::Location loc,
                                               llvm::StringRef text) {
    if (!emitted.insert({loc.getAsOpaquePointer(), text.str()}).second)
      return std::nullopt;
    ++numWarnings;
    mlir::InFlightDiagnostic diag = mlir::emitWarning(loc);
    diag << text;
    return diag;
  }
};

using CheckFn = void (*)(DiagnosticContext &);

/// One check: its tag, which its warnings carry and the `checks` option
/// names, its function, and whether `all` (the default) selects it; a check
/// that is off by default runs only when named.
struct HostDiagnosticCheck {
  llvm::StringLiteral tag;
  CheckFn run;
  bool onByDefault;
};

void checkHostPtrInKernel(DiagnosticContext &dc);
void checkHostHeapPtrInKernel(DiagnosticContext &dc);
void checkHostDerefDeviceUSM(DiagnosticContext &dc);

/// The checks, in the order they run.
constexpr HostDiagnosticCheck kChecks[] = {
    {"sycl-host-ptr-in-kernel", checkHostPtrInKernel, true},
    {"sycl-host-heap-ptr-in-kernel", checkHostHeapPtrInKernel, false},
    {"sycl-host-deref-device-usm", checkHostDerefDeviceUSM, true},
};

//===----------------------------------------------------------------------===//
// Device side: whether a kernel dereferences a pointer parameter
//===----------------------------------------------------------------------===//

/// Finds an access through a pointer parameter of a device kernel: a load,
/// store, copy, atomic (any op with a read or write effect on the pointer)
/// or a call of an atomic or vector load/store builtin, through the
/// parameter or a pointer derived from it (casts, cir.ptr_stride,
/// cir.get_member, cir.get_element), in the kernel or a function with a body
/// it calls. The pointer is followed through memory where a store puts it at
/// a member path of an object (a local, or what a pointer parameter or a
/// loaded pointer points to): a load of the same path of the same root, a
/// call that passes the root (or a member holding the path) on, and a
/// cir.copy of the object, the caller's object a parameter of an entered
// function points to (a constructor storing the pointer into `this`); and
// one level further, where the address of such
/// an object is itself stored (the `this` slot of operator() at -O0, the
/// kernel's spill of its argument): a load of that slot is the object. A
/// select or phi of the pointer, an integer round trip or deeper
/// indirection is not followed (the answer is then "no access", a missed
/// warning, never a wrong one).
class DerefFinder {
public:
  explicit DerefFinder(mlir::SymbolTableCollection &symbols)
      : symbols(symbols) {}

  /// The first access found through \p arg, or null.
  mlir::Operation *find(mlir::Value arg) {
    reset();
    addValue(arg);
    return run();
  }
  /// The first access found through the pointer member \p path of what
  /// \p root points to, or null.
  mlir::Operation *findAt(mlir::Value root, llvm::ArrayRef<int64_t> path) {
    reset();
    addLocation(root, Path(path.begin(), path.end()));
    return run();
  }

private:
  void reset() {
    deadEnds.clear();
    entered.clear();
    values.clear();
    locations.clear();
    work.clear();
    steps = 0;
  }
  mlir::Operation *run() {
    while (!work.empty() && ++steps < kMaxSteps) {
      Item item = work.front();
      work.pop_front();
      if (item.isValue) {
        if (mlir::Operation *access = visitValue(item.value))
          return access;
      } else {
        visitLocation(item.root, item.path, item.inner);
      }
    }
    return nullptr;
  }

  static constexpr unsigned kMaxSteps = 20000;
  using Path = llvm::SmallVector<int64_t, 4>;
  /// A value that is the pointer, or a location: member `path` of what
  /// `root` points to holds the pointer (no `inner`), or holds the address
  /// of an object whose member `inner` holds it.
  struct Item {
    bool isValue;
    mlir::Value value;
    mlir::Value root;
    Path path;
    std::optional<Path> inner;
  };

  void addValue(mlir::Value value) {
    if (values.insert(value).second)
      work.push_back({true, value, {}, {}, std::nullopt});
  }
  void addLocation(mlir::Value root, const Path &path,
                   std::optional<Path> inner = std::nullopt) {
    std::vector<int64_t> key(path.begin(), path.end());
    key.push_back(-1);
    if (inner)
      key.insert(key.end(), inner->begin(), inner->end());
    else
      key.push_back(-2);
    if (locations.insert({root.getAsOpaquePointer(), key}).second)
      work.push_back({false, {}, root, path, inner});
  }

  cir::FuncOp getCallee(cir::CallOp call) {
    std::optional<llvm::StringRef> name = call.getCallee();
    if (!name)
      return {};
    return symbols.lookupNearestSymbolFrom<cir::FuncOp>(
        call, mlir::StringAttr::get(call.getContext(), *name));
  }

  /// Whether \p callee is a builtin that reads or writes through its
  /// pointer arguments (the SPIR-V atomics and vector loads/stores device
  /// code calls; declarations).
  static bool isAccessBuiltin(cir::FuncOp callee) {
    llvm::StringRef name = callee.getSymName();
    return name.contains("__spirv_Atomic") ||
           name.contains("__spirv_ocl_vload") ||
           name.contains("__spirv_ocl_vstore");
  }

  /// Bind \p value, passed as operands of \p call, to the callee's
  /// parameters: as the pointer (\p rest null), or as the address of an
  /// object whose member \p rest holds it (or, with \p inner, the address
  /// of an object at \p rest of which is the address of an object whose
  /// member \p inner holds it).
  void passToCallee(cir::CallOp call, mlir::Value value, const Path *rest,
                    const std::optional<Path> &inner = std::nullopt) {
    cir::FuncOp callee = getCallee(call);
    if (!callee || callee.isDeclaration()) {
      deadEnd(callee ? ("call of declaration @" + callee.getSymName()).str()
                     : std::string("indirect call"));
      return;
    }
    for (auto [i, operand] : llvm::enumerate(call.getArgOperands())) {
      if (operand != value || i >= callee.getNumArguments())
        continue;
      mlir::Value formal = callee.getArgument(i);
      auto &calls = entered[callee.getOperation()];
      if (!llvm::is_contained(calls, call))
        calls.push_back(call);
      if (rest)
        addLocation(formal, *rest, inner);
      else
        addValue(formal);
    }
  }

  /// The root (a cir.alloca, a block argument or a loaded pointer) and the
  /// member path of the address \p addr, through address casts and
  /// cir.get_member.
  static bool resolve(mlir::Value addr, mlir::Value &root, Path &path) {
    Path reversed;
    for (unsigned depth = 0; depth < 32; ++depth) {
      if (mlir::isa<mlir::BlockArgument>(addr) ||
          addr.getDefiningOp<cir::AllocaOp>() ||
          addr.getDefiningOp<cir::LoadOp>()) {
        root = addr;
        path.assign(reversed.rbegin(), reversed.rend());
        return true;
      }
      if (auto cast = addr.getDefiningOp<cir::CastOp>()) {
        if (cast.getKind() != cir::CastKind::bitcast &&
            cast.getKind() != cir::CastKind::address_space)
          return false;
        addr = cast.getSrc();
        continue;
      }
      if (auto member = addr.getDefiningOp<cir::GetMemberOp>()) {
        reversed.push_back(member.getIndex());
        addr = member.getAddr();
        continue;
      }
      return false;
    }
    return false;
  }

  /// The uses of a value that is the pointer (or one derived from it).
  mlir::Operation *visitValue(mlir::Value value) {
    for (mlir::OpOperand &use : value.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (auto store = mlir::dyn_cast<cir::StoreOp>(user)) {
        if (store.getAddr() == value)
          return user;
        mlir::Value root;
        Path path;
        if (resolve(store.getAddr(), root, path))
          addLocation(root, path);
        continue;
      }
      if (auto cast = mlir::dyn_cast<cir::CastOp>(user)) {
        if (cast.getKind() == cir::CastKind::bitcast ||
            cast.getKind() == cir::CastKind::address_space ||
            cast.getKind() == cir::CastKind::array_to_ptrdecay)
          addValue(cast.getResult());
        continue;
      }
      if (auto stride = mlir::dyn_cast<cir::PtrStrideOp>(user)) {
        if (stride.getBase() == value)
          addValue(stride.getResult());
        continue;
      }
      if (auto member = mlir::dyn_cast<cir::GetMemberOp>(user)) {
        addValue(member.getResult());
        continue;
      }
      // A base or derived subobject of the pointee (sycl::vec's swizzle
      // bases, for instance).
      if (mlir::isa<cir::BaseClassAddrOp, cir::DerivedClassAddrOp>(user)) {
        addValue(user->getResult(0));
        continue;
      }
      // Returned: the result of the calls the walk entered the function
      // through (the pointer came in through one of their operands).
      if (mlir::isa<cir::ReturnOp>(user)) {
        auto func = user->getParentOfType<cir::FuncOp>();
        auto it = func ? entered.find(func.getOperation()) : entered.end();
        if (it != entered.end())
          for (cir::CallOp call : it->second)
            for (mlir::Value result : call->getResults())
              addValue(result);
        else
          deadEnd("cir.return");
        continue;
      }
      if (auto element = mlir::dyn_cast<cir::GetElementOp>(user)) {
        if (element.getBase() == value)
          addValue(element.getResult());
        continue;
      }
      if (auto call = mlir::dyn_cast<cir::CallOp>(user)) {
        cir::FuncOp callee = getCallee(call);
        if (callee && callee.isDeclaration() && isAccessBuiltin(callee))
          return user;
        passToCallee(call, value, nullptr);
        continue;
      }
      // A load, a copy, an atomic, a memset: an op that reads or writes
      // memory through the pointer.
      if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user)) {
        llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 2> list;
        effects.getEffectsOnValue(value, list);
        for (const auto &effect : list)
          if (mlir::isa<mlir::MemoryEffects::Read, mlir::MemoryEffects::Write>(
                  effect.getEffect()))
            return user;
      }
      deadEnd(user->getName().getStringRef());
    }
    return nullptr;
  }

  /// Member \p path of what \p root points to holds the pointer (or, with
  /// \p inner, the address of an object whose member \p inner holds it):
  /// the loads of it, and the calls, copies and stores that pass the object
  /// (or a member that holds the path) on.
  void visitLocation(mlir::Value root, const Path &path,
                     const std::optional<Path> &inner) {
    // A parameter of a function the walk entered: the memory is the caller's
    // object, which the caller reads afterwards (a constructor that stores
    // the pointer into `this`, such as sycl::atomic_ref's).
    if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(root)) {
      auto func = mlir::dyn_cast<cir::FuncOp>(arg.getOwner()->getParentOp());
      auto it = func && arg.getOwner() == &func.getBody().front()
                    ? entered.find(func.getOperation())
                    : entered.end();
      if (it != entered.end())
        for (cir::CallOp call : it->second) {
          if (arg.getArgNumber() >= call.getArgOperands().size())
            continue;
          mlir::Value outerRoot;
          Path outerPath;
          if (resolve(call.getArgOperands()[arg.getArgNumber()], outerRoot,
                      outerPath)) {
            outerPath.append(path.begin(), path.end());
            addLocation(outerRoot, outerPath, inner);
          }
        }
    }
    // (address, member path of it from root), through casts and members on
    // the way to `path`.
    llvm::SmallVector<std::pair<mlir::Value, Path>, 8> stack{{root, Path()}};
    llvm::DenseSet<mlir::Value> seen;
    while (!stack.empty()) {
      auto [addr, here] = stack.pop_back_val();
      if (!seen.insert(addr).second)
        continue;
      Path rest(path.begin() + here.size(), path.end());
      for (mlir::OpOperand &use : addr.getUses()) {
        mlir::Operation *user = use.getOwner();
        if (auto load = mlir::dyn_cast<cir::LoadOp>(user)) {
          if (rest.empty()) {
            if (inner)
              addLocation(load.getResult(), *inner);
            else
              addValue(load.getResult());
          }
          continue;
        }
        if (auto cast = mlir::dyn_cast<cir::CastOp>(user)) {
          if (cast.getKind() == cir::CastKind::bitcast ||
              cast.getKind() == cir::CastKind::address_space)
            stack.push_back({cast.getResult(), here});
          continue;
        }
        if (auto member = mlir::dyn_cast<cir::GetMemberOp>(user)) {
          if (!rest.empty() &&
              static_cast<int64_t>(member.getIndex()) == rest.front()) {
            Path next = here;
            next.push_back(rest.front());
            stack.push_back({member.getResult(), next});
          }
          continue;
        }
        if (auto call = mlir::dyn_cast<cir::CallOp>(user)) {
          passToCallee(call, addr, &rest, inner);
          continue;
        }
        if (auto copy = mlir::dyn_cast<cir::CopyOp>(user)) {
          mlir::Value dstRoot;
          Path dstPath;
          if (copy.getSrc() == addr &&
              resolve(copy.getDst(), dstRoot, dstPath)) {
            dstPath.append(rest.begin(), rest.end());
            addLocation(dstRoot, dstPath, inner);
          }
          continue;
        }
        // The address of the object is stored (one level of indirection):
        // a load of that slot gives the object back.
        if (auto store = mlir::dyn_cast<cir::StoreOp>(user)) {
          mlir::Value slotRoot;
          Path slotPath;
          if (!inner && store.getValue() == addr &&
              resolve(store.getAddr(), slotRoot, slotPath))
            addLocation(slotRoot, slotPath, rest);
          continue;
        }
      }
    }
  }

  /// Where the walk stopped, for -debug-only (why no access was found).
  void deadEnd(llvm::StringRef what) {
    LLVM_DEBUG(++deadEnds[what]);
    (void)what;
  }

public:
  /// The ops the last walk stopped at, most frequent first ("op x count").
  std::string formatDeadEnds() const {
    std::vector<std::pair<unsigned, std::string>> sorted;
    for (const auto &e : deadEnds)
      sorted.push_back({e.second, e.first().str()});
    llvm::sort(sorted, [](const auto &a, const auto &b) { return a > b; });
    std::string out;
    for (const auto &[n, what] : llvm::ArrayRef(sorted).take_front(4))
      out += (out.empty() ? "" : ", ") + what + " x" + std::to_string(n);
    return out;
  }

private:
  llvm::StringMap<unsigned> deadEnds;
  /// The calls the walk entered each function through.
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<cir::CallOp, 2>> entered;
  mlir::SymbolTableCollection &symbols;
  llvm::DenseSet<mlir::Value> values;
  std::set<std::pair<const void *, std::vector<int64_t>>> locations;
  std::deque<Item> work;
  unsigned steps = 0;
};

//===----------------------------------------------------------------------===//
// sycl-host-ptr-in-kernel
//===----------------------------------------------------------------------===//

/// Whether a kernel pointer parameter points into work-group local memory.
bool isLocalPointer(mlir::Type type) {
  auto ptrTy = mlir::dyn_cast<cir::PointerType>(type);
  auto as = ptrTy ? mlir::dyn_cast_if_present<cir::TargetAddressSpaceAttr>(
                        ptrTy.getAddrSpace())
                  : nullptr;
  return as && as.getValue() == 3;
}

/// The member path of a kernel parameter, as source text ("s.p", "a[0]").
std::string formatPath(mlir::ArrayAttr path) {
  std::string out;
  for (mlir::Attribute part : path) {
    llvm::StringRef name = mlir::cast<mlir::StringAttr>(part).getValue();
    if (!out.empty() && !name.starts_with("["))
      out += ".";
    out += name.str();
  }
  return out;
}

/// The name of element \p index of \p rec, from a cir.get_member of it in
/// the module of \p kernel, or "#<index>".
std::string memberName(cir::FuncOp kernel, cir::RecordType rec,
                       unsigned index) {
  std::string out = "#" + std::to_string(index);
  auto module = kernel->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return out;
  module.walk([&](cir::GetMemberOp member) {
    auto ptrTy = mlir::dyn_cast<cir::PointerType>(member.getAddr().getType());
    if (!ptrTy || ptrTy.getPointee() != rec || member.getIndex() != index)
      return mlir::WalkResult::advance();
    out = member.getName().str();
    return mlir::WalkResult::interrupt();
  });
  return out;
}

/// sycl-host-ptr-in-kernel (\p heap false: stack objects and globals) and
/// sycl-host-heap-ptr-in-kernel (\p heap true: heap allocations).
void checkHostMemoryInKernel(DiagnosticContext &dc, bool heap) {
  using HostOnlyMemory = cir::SYCLHostFacts::HostOnlyMemory;
  const cir::pointsto::PointsToAnalysis &analysis = dc.facts.getAnalysis();

  // The candidates of a kernel: a captured pointer (kind "pointer"), or a
  // pointer member of a captured standard-layout struct the kernel takes by
  // value (kind "std_layout", passed byval: element `element` of it).
  struct Candidate {
    unsigned argNo;
    std::optional<unsigned> element;
    mlir::ArrayAttr path;
    std::string name;
  };
  auto candidatesOf = [&](cir::FuncOp kernel) {
    llvm::SmallVector<Candidate, 4> out;
    for (unsigned i = 0, e = kernel.getNumArguments(); i != e; ++i) {
      std::optional<cir::SYCLKernelParam> param =
          cir::getSYCLKernelParam(kernel, i);
      if (!param)
        continue;
      mlir::Type type = kernel.getArgumentTypes()[i];
      auto ptrTy = mlir::dyn_cast<cir::PointerType>(type);
      if (!ptrTy || isLocalPointer(type))
        continue;
      if (param->kind == "pointer") {
        out.push_back({i, std::nullopt, param->path, formatPath(param->path)});
        continue;
      }
      if (param->kind != "std_layout" || !kernel.getArgAttr(i, "llvm.byval"))
        continue;
      auto rec = mlir::dyn_cast<cir::RecordType>(ptrTy.getPointee());
      if (!rec || rec.isIncomplete() || rec.isUnion())
        continue;
      for (unsigned j = 0, n = rec.getNumElements(); j != n; ++j) {
        mlir::Type elem = rec.getElementType(j);
        if (!mlir::isa<cir::PointerType>(elem) || isLocalPointer(elem))
          continue;
        out.push_back(
            {i, j, param->path,
             formatPath(param->path) + "." + memberName(kernel, rec, j)});
      }
    }
    return out;
  };

  // One group per (launch call, device kernel, candidate): the facts of its
  // records (one per covered calling context of the launch).
  struct Group {
    cir::FuncOp kernel;
    Candidate candidate;
    unsigned records = 0;
    unsigned hostOnly = 0;
    HostOnlyMemory first;
    bool mixed = false;
  };
  llvm::MapVector<
      std::tuple<mlir::Operation *, mlir::Operation *, unsigned, int64_t>,
      Group>
      groups;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<Candidate, 4>> cache;
  for (const cir::pointsto::LaunchRecord &launch : analysis.getLaunches()) {
    for (mlir::StringAttr name : launch.kernelNames) {
      for (cir::FuncOp kernel : dc.facts.getDeviceKernels(name.getValue())) {
        if (kernel.getCallingConv() != cir::CallingConv::SpirKernel ||
            kernel.getBody().empty())
          continue;
        auto it = cache.find(kernel.getOperation());
        if (it == cache.end())
          it = cache.try_emplace(kernel.getOperation(), candidatesOf(kernel))
                   .first;
        for (const Candidate &c : it->second) {
          Group &g = groups[{launch.call, kernel.getOperation(), c.argNo,
                             c.element ? int64_t(*c.element) : -1}];
          g.kernel = kernel;
          g.candidate = c;
          ++g.records;
          llvm::SmallVector<unsigned, 1> elements;
          if (c.element)
            elements.push_back(*c.element);
          HostOnlyMemory memory =
              dc.facts.hostOnlyMemoryOfClosureMember(launch, c.path, elements);
          LLVM_DEBUG(llvm::dbgs()
                     << "host-ptr-in-kernel: @" << kernel.getSymName()
                     << " arg " << c.argNo << " '" << c.name << "' at "
                     << launch.call->getLoc() << ": "
                     << (memory.kind == HostOnlyMemory::Kind::Unknown
                             ? memory.why
                             : std::string("host-only"))
                     << "\n");
          if (memory.kind == HostOnlyMemory::Kind::Unknown)
            continue;
          if (!g.hostOnly)
            g.first = memory;
          else if (g.first.kind != memory.kind)
            g.mixed = true;
          ++g.hostOnly;
        }
      }
    }
  }

  DerefFinder finder(dc.symbols);
  for (auto &entry : groups) {
    Group &g = entry.second;
    // The fact must hold at every record of the launch.
    if (!g.records || g.hostOnly != g.records || g.mixed)
      continue;
    if (heap != (g.first.kind == HostOnlyMemory::Kind::Heap))
      continue; // The other tag's.
    mlir::Value arg = g.kernel.getBody().front().getArgument(g.candidate.argNo);
    mlir::Operation *access =
        g.candidate.element
            ? finder.findAt(arg, {static_cast<int64_t>(*g.candidate.element)})
            : finder.find(arg);
    LLVM_DEBUG(llvm::dbgs() << "host-ptr-in-kernel: @" << g.kernel.getSymName()
                            << " '" << g.candidate.name << "'"
                            << (access ? " dereferenced"
                                       : " not dereferenced (stops at: " +
                                             finder.formatDeadEnds() + ")")
                            << "\n");
    if (!access)
      continue; // The kernel does not dereference it: exempt.
    llvm::StringRef what;
    switch (g.first.kind) {
    case HostOnlyMemory::Kind::Stack:
      what = "stack variable";
      break;
    case HostOnlyMemory::Kind::Global:
      what = "global";
      break;
    case HostOnlyMemory::Kind::Heap:
      what = "heap allocation";
      break;
    case HostOnlyMemory::Kind::Unknown:
      continue;
    }
    std::string text =
        heap ? ("kernel argument '" + g.candidate.name +
                "' points to host heap memory (" + what +
                ") that the device can access only if it has "
                "aspect::usm_system_allocations (SYCL 2020 sec:usm) "
                "[sycl-host-heap-ptr-in-kernel]")
                   .str()
             : ("kernel argument '" + g.candidate.name +
                "' points to host memory (" + what +
                ") that the device cannot access "
                "[sycl-host-ptr-in-kernel]")
                   .str();
    std::optional<mlir::InFlightDiagnostic> diag =
        dc.warn(access->getLoc(), text);
    if (!diag)
      continue;
    // The kernel (its lambda or functor), which the access may be far from
    // (a device function it calls).
    diag->attachNote(g.kernel.getLoc())
        << "the kernel that captures the pointer is defined here";
    // A note at each allocation site of the node (one, or a few for an
    // allocation function with several sites).
    llvm::DenseSet<const void *> noted;
    for (const cir::pointsto::Origin &origin :
         llvm::ArrayRef(g.first.node->getOrigins()).take_front(4)) {
      mlir::Location loc = dc.container.getLoc();
      std::string note;
      switch (origin.kind) {
      case cir::pointsto::Origin::Kind::Alloca: {
        auto alloca = mlir::cast<cir::AllocaOp>(origin.site);
        loc = alloca.getLoc();
        note = ("the stack variable '" + alloca.getName() +
                "' the pointer points to is allocated here")
                   .str();
        break;
      }
      case cir::pointsto::Origin::Kind::GlobalSym: {
        llvm::StringRef sym = origin.sym.getLeafReference().getValue();
        if (mlir::Operation *global =
                dc.symbols.lookupSymbolIn(dc.host, origin.sym))
          loc = global->getLoc();
        note = ("the global '" + sym + "' the pointer points to is " +
                "defined here")
                   .str();
        break;
      }
      case cir::pointsto::Origin::Kind::HeapCall:
        loc = origin.site->getLoc();
        note = "the heap memory the pointer points to is allocated here";
        break;
      default:
        continue;
      }
      if (noted.insert(loc.getAsOpaquePointer()).second)
        diag->attachNote(loc) << note;
    }
  }
}

void checkHostPtrInKernel(DiagnosticContext &dc) {
  checkHostMemoryInKernel(dc, /*heap=*/false);
}

void checkHostHeapPtrInKernel(DiagnosticContext &dc) {
  checkHostMemoryInKernel(dc, /*heap=*/true);
}

//===----------------------------------------------------------------------===//
// sycl-host-deref-device-usm
//===----------------------------------------------------------------------===//

/// Whether \p func is part of the SYCL implementation: a SYCL host-API role,
/// or namespace sycl (a lambda in a function of it too; a launch-path copy
/// keeps its original's name with a suffix).
bool isSYCLImplementation(cir::FuncOp func) {
  if (func->hasAttr(cir::CIRDialect::getSYCLHostAPIAttrName()))
    return true;
  llvm::StringRef name = func.getSymName();
  return name.starts_with("_ZN4sycl") || name.starts_with("_ZNK4sycl") ||
         name.starts_with("_ZZN4sycl") || name.starts_with("_ZZNK4sycl");
}

/// The C library functions that read or write through every pointer
/// argument (when called as declarations; CIRGen usually emits the
/// cir.libc ops for them instead).
bool isLibcMemoryFunction(llvm::StringRef name) {
  return llvm::is_contained({"memcpy", "memmove", "memset", "memcmp"}, name);
}

/// "file:line" of \p loc (the first file location in it), or "".
std::string formatFileLine(mlir::Location loc) {
  std::string out;
  loc->walk([&](mlir::Location inner) {
    auto file = mlir::dyn_cast<mlir::FileLineColLoc>(inner);
    if (!file)
      return mlir::WalkResult::advance();
    out = (llvm::sys::path::filename(file.getFilename().getValue()) + ":" +
           llvm::Twine(file.getLine()))
              .str();
    return mlir::WalkResult::interrupt();
  });
  return out;
}

/// Whether a host function accesses memory through one of its pointer
/// parameters: a load, store, copy, atomic or libc memory op whose address
/// is the parameter or derived from it (casts, cir.ptr_stride,
/// cir.get_member, cir.get_element, base and derived class addresses), also
/// after a spill to a local slot and a reload (the -O0 shape of every
/// parameter), or in a function it passes the pointer to that does so
/// (functions with a body only; not the SYCL implementation but its value
/// types). A select of the pointer, an escape into memory other than a
/// local slot, or a call of a declaration is not followed: "no access"
/// then, a missed warning. The body is the one the TU runs: a definition
/// that another TU cannot replace (external, internal, or ODR, whose
/// definitions are the same in every TU).
class HostAccessSummary {
public:
  explicit HostAccessSummary(mlir::SymbolTableCollection &symbols)
      : symbols(symbols) {}

  /// Whether \p func is a method of a SYCL value type (sycl::vec and its
  /// detail::vec_base storage, sycl::marray, sycl::half), plain host code.
  static bool isSYCLValueType(cir::FuncOp func) {
    llvm::StringRef name = func.getSymName();
    for (llvm::StringRef prefix :
         {"_ZN4sycl3_V13vec", "_ZNK4sycl3_V13vec", "_ZN4sycl3_V16marray",
          "_ZNK4sycl3_V16marray", "_ZN4sycl3_V16detail8vec_base",
          "_ZNK4sycl3_V16detail8vec_base", "_ZN4sycl3_V16detail9half_impl4half",
          "_ZNK4sycl3_V16detail9half_impl4half"})
      if (name.starts_with(prefix))
        return true;
    return false;
  }

  /// The first op of \p func (or a function it calls) that accesses memory
  /// through its parameter \p index, or null.
  mlir::Operation *accessThrough(cir::FuncOp func, unsigned index,
                                 unsigned depth = 0) {
    auto key = std::make_pair(func.getOperation(), index);
    if (auto it = memo.find(key); it != memo.end())
      return it->second;
    memo[key] = nullptr; // A recursive call: no access (yet).
    mlir::Operation *found = nullptr;
    if (depth < 8 && eligible(func) && index < func.getNumArguments() &&
        !func.getBody().empty())
      found = walk(func.getArgument(index), depth);
    memo[key] = found;
    return found;
  }

private:
  bool eligible(cir::FuncOp func) {
    if (func.isDeclaration() ||
        func->hasAttr(cir::CIRDialect::getSYCLHostAPIAttrName()))
      return false;
    if (isSYCLImplementation(func) && !isSYCLValueType(func))
      return false;
    switch (func.getLinkage()) {
    case cir::GlobalLinkageKind::ExternalLinkage:
    case cir::GlobalLinkageKind::AvailableExternallyLinkage:
    case cir::GlobalLinkageKind::LinkOnceODRLinkage:
    case cir::GlobalLinkageKind::WeakODRLinkage:
    case cir::GlobalLinkageKind::InternalLinkage:
    case cir::GlobalLinkageKind::PrivateLinkage:
      return true;
    default:
      return false;
    }
  }

  mlir::Operation *walk(mlir::Value start, unsigned depth) {
    llvm::SmallVector<mlir::Value, 8> work{start};
    llvm::DenseSet<mlir::Value> seen{start};
    auto add = [&](mlir::Value v) {
      if (seen.insert(v).second)
        work.push_back(v);
    };
    for (unsigned steps = 0; !work.empty() && steps < 4096; ++steps) {
      mlir::Value value = work.pop_back_val();
      for (mlir::OpOperand &use : value.getUses()) {
        mlir::Operation *user = use.getOwner();
        if (auto store = mlir::dyn_cast<cir::StoreOp>(user)) {
          if (store.getAddr() == value)
            return user;
          // Spilled to a local slot: its loads give the pointer back.
          if (auto slot = store.getAddr().getDefiningOp<cir::AllocaOp>())
            for (mlir::Operation *slotUser : slot->getUsers())
              if (auto load = mlir::dyn_cast<cir::LoadOp>(slotUser))
                add(load.getResult());
          continue;
        }
        if (auto cast = mlir::dyn_cast<cir::CastOp>(user)) {
          if (cast.getKind() == cir::CastKind::bitcast ||
              cast.getKind() == cir::CastKind::address_space ||
              cast.getKind() == cir::CastKind::array_to_ptrdecay)
            add(cast.getResult());
          continue;
        }
        if (auto stride = mlir::dyn_cast<cir::PtrStrideOp>(user)) {
          if (stride.getBase() == value)
            add(stride.getResult());
          continue;
        }
        if (auto element = mlir::dyn_cast<cir::GetElementOp>(user)) {
          if (element.getBase() == value)
            add(element.getResult());
          continue;
        }
        if (mlir::isa<cir::GetMemberOp, cir::BaseClassAddrOp,
                      cir::DerivedClassAddrOp>(user)) {
          add(user->getResult(0));
          continue;
        }
        if (auto call = mlir::dyn_cast<cir::CallOp>(user)) {
          std::optional<llvm::StringRef> name = call.getCallee();
          cir::FuncOp callee =
              name ? symbols.lookupNearestSymbolFrom<cir::FuncOp>(
                         call, mlir::StringAttr::get(call.getContext(), *name))
                   : cir::FuncOp();
          if (!callee)
            continue;
          if (callee.isDeclaration()) {
            if (isLibcMemoryFunction(callee.getSymName()))
              return user;
            continue;
          }
          for (auto [i, operand] : llvm::enumerate(call.getArgOperands()))
            if (operand == value)
              if (mlir::Operation *inner = accessThrough(callee, i, depth + 1))
                return inner;
          continue;
        }
        if (mlir::isa<cir::CIRCallOpInterface>(user))
          continue;
        if (auto effects =
                mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user)) {
          llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 2> list;
          effects.getEffectsOnValue(value, list);
          for (const auto &effect : list)
            if (mlir::isa<mlir::MemoryEffects::Read,
                          mlir::MemoryEffects::Write>(effect.getEffect()))
              return user;
        }
      }
    }
    return nullptr;
  }

  mlir::SymbolTableCollection &symbols;
  std::map<std::pair<mlir::Operation *, unsigned>, mlir::Operation *> memo;
};

void checkHostDerefDeviceUSM(DiagnosticContext &dc) {
  const cir::pointsto::PointsToAnalysis &analysis = dc.facts.getAnalysis();

  // The functions exempt: the SYCL implementation, a function that asks for
  // a pointer's USM kind, and the kernels' bodies (the methods of a launch's
  // closure type, and what only they call).
  llvm::DenseSet<mlir::Type> closureTypes;
  for (const cir::pointsto::LaunchRecord &launch : analysis.getAllLaunches())
    if (launch.closureType)
      closureTypes.insert(launch.closureType);
  llvm::DenseSet<mlir::Operation *> exempt, kernelBodies;
  auto calleeOf = [&](mlir::Operation *op) -> cir::FuncOp {
    auto call = mlir::dyn_cast<cir::CIRCallOpInterface>(op);
    if (!call || call.isIndirect())
      return {};
    auto sym = op->getAttrOfType<mlir::FlatSymbolRefAttr>("callee");
    return sym ? dc.symbols.lookupNearestSymbolFrom<cir::FuncOp>(op, sym)
               : cir::FuncOp();
  };
  for (const auto &fn : analysis.getFunctions()) {
    cir::FuncOp func = fn->func;
    if (isSYCLImplementation(func)) {
      exempt.insert(func);
      continue;
    }
    bool asksKind = false;
    func.walk([&](mlir::Operation *op) {
      cir::FuncOp callee = calleeOf(op);
      if (callee && isSYCLImplementation(callee) &&
          callee.getSymName().contains("16get_pointer_type")) {
        asksKind = true;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    if (asksKind)
      exempt.insert(func);
  }
  // Kernel bodies: a method of a closure type (its `this`), then, to a
  // fixpoint, a function every use of which is in a kernel body.
  for (const auto &fn : analysis.getFunctions()) {
    cir::FuncOp func = fn->func;
    if (!func.getNumArguments())
      continue;
    auto ptrTy = mlir::dyn_cast<cir::PointerType>(func.getArgumentTypes()[0]);
    if (ptrTy && closureTypes.contains(ptrTy.getPointee()))
      kernelBodies.insert(func);
  }
  if (!kernelBodies.empty()) {
    // The functions each symbol is used in (calls, address-of), once.
    llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<mlir::Operation *, 2>>
        usersOf;
    if (std::optional<mlir::SymbolTable::UseRange> uses =
            mlir::SymbolTable::getSymbolUses(&dc.host.getBodyRegion()))
      for (const mlir::SymbolTable::SymbolUse &use : *uses) {
        mlir::Operation *user = use.getUser();
        auto func = mlir::isa<cir::FuncOp>(user)
                        ? mlir::cast<cir::FuncOp>(user)
                        : user->getParentOfType<cir::FuncOp>();
        auto &list = usersOf[use.getSymbolRef().getRootReference()];
        mlir::Operation *holder = func ? func.getOperation() : user;
        if (!llvm::is_contained(list, holder))
          list.push_back(holder);
      }
    for (bool changed = true; changed;) {
      changed = false;
      for (const auto &fn : analysis.getFunctions()) {
        cir::FuncOp func = fn->func;
        if (kernelBodies.contains(func))
          continue;
        auto it = usersOf.find(func.getSymNameAttr());
        if (it == usersOf.end() || it->second.empty())
          continue;
        if (llvm::all_of(it->second, [&](mlir::Operation *user) {
              return kernelBodies.contains(user);
            })) {
          kernelBodies.insert(func);
          changed = true;
        }
      }
    }
  }

  // The pointer operands \p op reads or writes memory through.
  auto accessedPointers = [&](mlir::Operation *op,
                              llvm::SmallVectorImpl<mlir::Value> &out) {
    if (mlir::isa<cir::CIRCallOpInterface>(op)) {
      cir::FuncOp callee = calleeOf(op);
      if (callee && callee.isDeclaration() &&
          isLibcMemoryFunction(callee.getSymName()))
        for (mlir::Value operand : op->getOperands())
          if (mlir::isa<cir::PointerType>(operand.getType()))
            out.push_back(operand);
      return;
    }
    auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
    if (!effects)
      return;
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 4> list;
    effects.getEffects(list);
    for (const auto &effect : list) {
      mlir::Value value = effect.getValue();
      if (!value || !mlir::isa<cir::PointerType>(value.getType()) ||
          !mlir::isa<mlir::MemoryEffects::Read, mlir::MemoryEffects::Write>(
              effect.getEffect()))
        continue;
      if (!llvm::is_contained(out, value))
        out.push_back(value);
    }
  };

  // Warn at \p at about the access of \p ptr (in \p func), if it must
  // point into device USM memory; \p callee, \p access: the access is in
  // that callee (a call site lifted from it, see below).
  HostAccessSummary summary(dc.symbols);
  auto check = [&](cir::FuncOp func, mlir::Operation *at, mlir::Value ptr,
                   cir::FuncOp callee, mlir::Operation *access) {
    cir::SYCLHostFacts::DeviceUSMMemory memory =
        dc.facts.deviceUSMOfPointer(ptr);
    // -debug-only: the verdict of every access that may reach USM memory
    // (the others are the bulk of a TU's accesses).
    LLVM_DEBUG({
      if (memory.reachesUSM)
        llvm::dbgs() << "host-deref-device-usm: @" << func.getSymName() << " "
                     << at->getName() << " at " << at->getLoc() << ": "
                     << (memory.nodes.empty() ? memory.why
                                              : std::string("device USM"))
                     << (callee ? (" (in @" + callee.getSymName() + ")").str()
                                : std::string())
                     << "\n";
    });
    if (memory.nodes.empty())
      return;
    // The allocation sites, for the text and the notes.
    llvm::SmallVector<mlir::Operation *, 4> sites;
    for (const cir::pointsto::Node *node : memory.nodes)
      for (const cir::pointsto::Origin &origin : node->getOrigins())
        if (origin.site && !llvm::is_contained(sites, origin.site))
          sites.push_back(origin.site);
    if (sites.empty())
      return;
    std::string where = formatFileLine(sites.front()->getLoc());
    if (where.empty())
      where = "an unknown location";
    if (sites.size() > 1)
      where += " (and " + std::to_string(sites.size() - 1) + " other site" +
               (sites.size() > 2 ? "s" : "") + ")";
    std::string text = "host access to device USM memory allocated at " +
                       where + " [sycl-host-deref-device-usm]";
    std::optional<mlir::InFlightDiagnostic> diag = dc.warn(at->getLoc(), text);
    if (!diag)
      return;
    if (callee && access)
      diag->attachNote(access->getLoc())
          << "the called function accesses the memory here";
    for (mlir::Operation *site : llvm::ArrayRef(sites).take_front(4))
      diag->attachNote(site->getLoc())
          << "the device USM memory is allocated here; device allocations "
             "are not accessible on the host (SYCL 2020 sec:usm)";
  };

  for (const auto &fn : analysis.getFunctions()) {
    cir::FuncOp func = fn->func;
    if (exempt.contains(func) || kernelBodies.contains(func))
      continue;
    func.walk([&](mlir::Operation *op) {
      llvm::SmallVector<mlir::Value, 2> pointers;
      accessedPointers(op, pointers);
      for (mlir::Value ptr : pointers)
        check(func, op, ptr, {}, nullptr);
      // A call of a function whose own accesses the check cannot judge,
      // because its formals are incomplete (an ODR or externally visible
      // function: other TUs may call it), or because it is exempt but a
      // SYCL value type (sycl::vec, half, marray: their methods are plain
      // host code): if the callee accesses memory through a pointer
      // argument, the access is judged at the call, where the argument is
      // known.
      auto call = mlir::dyn_cast<cir::CallOp>(op);
      if (!call)
        return;
      cir::FuncOp callee = calleeOf(op);
      if (!callee || callee.isDeclaration() || kernelBodies.contains(callee))
        return;
      if (isSYCLImplementation(callee) &&
          !HostAccessSummary::isSYCLValueType(callee))
        return;
      const cir::pointsto::TUFunction *tuCallee = analysis.getFunction(callee);
      if (tuCallee && tuCallee->formalsComplete &&
          !isSYCLImplementation(callee))
        return; // Judged in the callee, in its calling contexts.
      for (auto [i, operand] : llvm::enumerate(call.getArgOperands()))
        if (mlir::isa<cir::PointerType>(operand.getType()))
          if (mlir::Operation *access = summary.accessThrough(callee, i))
            check(func, op, operand, callee, access);
    });
  }
}

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

struct OffloadHostDiagnosticsPass
    : public impl::OffloadHostDiagnosticsBase<OffloadHostDiagnosticsPass> {
  void runOnOperation() override;
};

void OffloadHostDiagnosticsPass::runOnOperation() {
  // Diagnostics only: nothing below changes the IR.
  markAllAnalysesPreserved();
  mlir::ModuleOp container = getOperation();

  // The checks selected.
  llvm::SmallVector<const HostDiagnosticCheck *, 4> selected;
  llvm::SmallVector<llvm::StringRef, 4> names;
  llvm::StringRef(checks).split(names, ',', /*MaxSplit=*/-1,
                                /*KeepEmpty=*/false);
  for (llvm::StringRef name : names) {
    name = name.trim();
    if (name == "none")
      continue;
    if (name == "all") {
      for (const HostDiagnosticCheck &check : kChecks)
        if (check.onByDefault)
          selected.push_back(&check);
      continue;
    }
    const HostDiagnosticCheck *found = llvm::find_if(
        kChecks, [&](const HostDiagnosticCheck &c) { return c.tag == name; });
    if (found == std::end(kChecks)) {
      std::string known;
      for (const HostDiagnosticCheck &check : kChecks)
        known += (known.empty() ? "" : ", ") + check.tag.str();
      container.emitError()
          << "unknown host diagnostic '" << name
          << "' (expected all, none or one of: " << known << ")";
      return signalPassFailure();
    }
    selected.push_back(found);
  }
  if (selected.empty() || !cir::isOffloadContainer(container))
    return;

  std::optional<cir::HostFactsMode> mode = cir::parseHostFactsMode(hostFacts);
  if (!mode) {
    container.emitError() << "unknown host-facts mode '" << hostFacts
                          << "' (expected ipo or none)";
    return signalPassFailure();
  }
  if (*mode == cir::HostFactsMode::None)
    return;
  // The analysis the host-facts passes share (SYCLHostFactsCache): the same
  // options give the same key.
  cir::HostFactsOptions options;
  options.mode = *mode;
  options.assumeClosedTU = hostFactsClosedTU;
  options.bottomUp = hostFactsBottomUp;
  options.subobjects = hostFactsSubobjects;
  options.budgetScale = hostFactsBudgetScale;
  options.relevanceSlice = hostFactsSlice;
  options.maxSeconds = hostFactsMaxSeconds;
  options.maxMemoryMB = hostFactsMaxMemoryMB;
  options.internalize = hostFactsInternalize;
  cir::SYCLHostFactsCache &factsCache = getAnalysis<cir::SYCLHostFactsCache>();
  unsigned builtBefore = factsCache.getNumBuilt();
  const cir::SYCLHostFacts &facts = factsCache.get(options);
  numHostFactsBuilt += factsCache.getNumBuilt() - builtBefore;

  llvm::StringSet<> ran;
  DiagnosticContext dc(container, facts);
  for (const HostDiagnosticCheck *check : selected)
    if (ran.insert(check->tag).second)
      check->run(dc);
  numWarnings += dc.numWarnings;
}

} // namespace

std::unique_ptr<Pass> mlir::createOffloadHostDiagnosticsPass() {
  return std::make_unique<OffloadHostDiagnosticsPass>();
}
