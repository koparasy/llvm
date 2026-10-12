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

#include <deque>
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

/// The checks, in the order they run.
constexpr HostDiagnosticCheck kChecks[] = {
    {"sycl-host-ptr-in-kernel", checkHostPtrInKernel, true},
    {"sycl-host-heap-ptr-in-kernel", checkHostHeapPtrInKernel, false},
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
