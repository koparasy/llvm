//===- DetectHostDeviceClosureLayoutMismatch.cpp - closure ABI check ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Warns when a SYCL kernel's captured-closure layout diverges between the host
// module and the device module of a merged cir.offload.container
// ([sycl-closure-layout-mismatch]).
//
// A SYCL kernel lambda is compiled twice: once for the host and once for the
// SPIR-V device. Preprocessor divergence (`#ifdef __SYCL_DEVICE_ONLY__`) or
// divergent `[=]` implicit captures can make the two closures disagree in a
// field's type/layout while keeping the same total size. The
// `sizeof(KernelType) == Info.KernelSize` static_assert in `sycl/queue.hpp`
// only checks total size, so an equal-size / different-type divergence compiles
// clean and produces silent wrong results at runtime (observed: a kernel that
// should write 111 wrote 0). Only the joined host+device container sees both
// layouts, so only a container pass can catch it.
//
// For each SYCL binding of the KernelBindingTable:
//  - the HOST closure record is the record a host `hostkernel.ctor` call
//    (detail::HostKernelRef / detail::HostKernel constructor) wraps: on the
//    direct launch path the constructor of the HostKernelRef the launch
//    passes, on the handler path the HostKernel built next to the
//    `getDeviceKernelInfo<Kernel>()` marker (in the marker's function or its
//    direct callers: handler::StoreLambda<Name, Type> both builds the
//    HostKernel and calls setDeviceKernelInfo<Name>). All launch sites must
//    agree on one record, else the kernel is not checked. The record is never
//    matched by source location or name: one lambda of a function template
//    has a closure per instantiation, all at the same location;
//  - the DEVICE closure record is the type of the closure object CIRGen tags
//    `cir.sycl_kernel_object` in the spir_kernel.
//
// The two records are compared by LAYOUT, each with its own module's
// DataLayout: both are flattened to their leaves (offset, size, kind: integer
// of width N -- signedness ignored --, bool, a float kind, pointer -- address
// space and pointee ignored, both are device-lowering artifacts --, vector or
// array of K elements of a kind, bit-field unit); records are recursed into
// (record names are never compared), and pad/empty members are skipped, so a
// padding-only difference is not a divergence. SYCL library types (`sycl::`,
// e.g. half, vec, id, accessor, stream) and unions are opaque leaves compared
// by size only: their host and device representations differ by design
// (sycl::half is uint16_t on the host and _Float16 on the device). Each
// offset where the leaves disagree gives one warning per captured field,
// anchored at the device get_member of the field, naming both kinds
// ("host: int32 at offset 8, device: float at offset 8"); a differing leaf
// count gives one more warning at the kernel.
//
// Accessors: a captured accessor or local_accessor is decomposed into device
// kernel parameters (`cir.sycl_kernel_param` kind accessor/local_accessor,
// parts ptr / access_range / mem_range / offset). Its host field is the opaque
// leaf at the offset of the parameter's closure path; the element type and
// dimensions in the host accessor's template arguments are checked against
// the pointee of the device `ptr` parameter and the rank of the device
// `access_range` parameter. Whatever cannot be decided (an element type that
// is not a builtin scalar or a record, an unparsable name, a missing part) is
// not checked; with `report-unchecked` a remark says so (debug aid, off by
// default).
//
// Analysis + warning ONLY: the pass never mutates the module. It runs at the
// head of the cir-offload-merge combine pipeline, BEFORE the mem2reg/sccp that
// would promote the device closure alloca away, so the check sees the
// as-emitted closures.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Location.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLKernelParams.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/SYCLPointsToModels.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "cir-offload-closure-layout"

namespace mlir {
#define GEN_PASS_DECL_OFFLOADDETECTHOSTDEVICECLOSURELAYOUTMISMATCH
#define GEN_PASS_DEF_OFFLOADDETECTHOSTDEVICECLOSURELAYOUTMISMATCH
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace cir;

namespace {

constexpr llvm::StringLiteral kTag = "[sycl-closure-layout-mismatch]";

//===----------------------------------------------------------------------===//
// Host closure recovery
//===----------------------------------------------------------------------===//

// Peel the offset-0 reinterprets CIRGen inserts on a closure `this`
// (address-space / bitcast casts and base_class_addr views) so a closure
// operand resolves to the object pointer it names.
static mlir::Value peelClosurePtr(mlir::Value v) {
  for (;;) {
    if (auto bca = v.getDefiningOp<cir::BaseClassAddrOp>()) {
      v = bca.getSrcAddr();
      continue;
    }
    if (auto cast = v.getDefiningOp<cir::CastOp>()) {
      if (cast.getKind() == cir::CastKind::address_space ||
          cast.getKind() == cir::CastKind::bitcast) {
        v = cast.getSrc();
        continue;
      }
    }
    return v;
  }
}

static cir::RecordType pointeeRecord(mlir::Type type) {
  auto ptr = mlir::dyn_cast<cir::PointerType>(type);
  return ptr ? mlir::dyn_cast<cir::RecordType>(ptr.getPointee())
             : cir::RecordType();
}

// The cir.sycl_host_api role of the direct callee of `call`, or empty.
static llvm::StringRef calleeRole(cir::CallOp call,
                                  mlir::SymbolTableCollection &symbols) {
  if (call.isIndirect())
    return {};
  auto fn =
      symbols.lookupNearestSymbolFrom<cir::FuncOp>(call, call.getCalleeAttr());
  auto role = fn ? fn->getAttrOfType<mlir::StringAttr>(
                       cir::CIRDialect::getSYCLHostAPIAttrName())
                 : nullptr;
  return role ? role.getValue() : llvm::StringRef();
}

// The closure record a HostKernel / HostKernelRef constructor call wraps: the
// record its second argument (`const KernelType &`) points to.
static cir::RecordType closureOfHostKernelCtor(cir::CallOp ctor) {
  mlir::OperandRange args = ctor.getArgOperands();
  if (args.size() < 2)
    return {};
  return pointeeRecord(peelClosurePtr(args[1]).getType());
}

// The host closure record of a launch site's closure operand: on the direct
// path the operand is (a view of) the HostKernelRef a hostkernel.ctor call
// constructed, which wraps the closure; on the handler path the operand is
// already the closure the HostKernel was built from.
static cir::RecordType
hostClosureOfOperand(mlir::Value operand,
                     mlir::SymbolTableCollection &symbols) {
  if (!operand)
    return {};
  mlir::Value obj = peelClosurePtr(operand);
  for (mlir::Operation *user : obj.getUsers()) {
    auto ctor = mlir::dyn_cast<cir::CallOp>(user);
    if (ctor && calleeRole(ctor, symbols) == "hostkernel.ctor" &&
        !ctor.getArgOperands().empty() && ctor.getArgOperands()[0] == obj)
      return closureOfHostKernelCtor(ctor);
  }
  return pointeeRecord(obj.getType());
}

// Direct callers of each host function, built on first use.
class HostCallers {
public:
  HostCallers(mlir::ModuleOp host, mlir::SymbolTableCollection &symbols)
      : host(host), symbols(symbols) {}

  llvm::ArrayRef<cir::FuncOp> callersOf(cir::FuncOp fn) {
    if (!built) {
      built = true;
      host.walk([&](cir::CallOp call) {
        if (call.isIndirect())
          return;
        auto caller = call->getParentOfType<cir::FuncOp>();
        auto callee = symbols.lookupNearestSymbolFrom<cir::FuncOp>(
            call, call.getCalleeAttr());
        if (!caller || !callee)
          return;
        llvm::SmallVector<cir::FuncOp, 2> &list = callers[callee];
        if (!llvm::is_contained(list, caller))
          list.push_back(caller);
      });
    }
    auto it = callers.find(fn);
    return it == callers.end() ? llvm::ArrayRef<cir::FuncOp>()
                               : llvm::ArrayRef<cir::FuncOp>(it->second);
  }

private:
  mlir::ModuleOp host;
  mlir::SymbolTableCollection &symbols;
  bool built = false;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<cir::FuncOp, 2>> callers;
};

// The closure records of the hostkernel.ctor calls in `fn`.
static void ctorClosuresIn(cir::FuncOp fn, mlir::SymbolTableCollection &symbols,
                           llvm::SmallVectorImpl<cir::RecordType> &out) {
  fn.walk([&](cir::CallOp call) {
    if (calleeRole(call, symbols) != "hostkernel.ctor")
      return;
    cir::RecordType rec = closureOfHostKernelCtor(call);
    out.push_back(rec); // A null record makes the result ambiguous.
  });
}

// The host closure record of a handler-path launch whose closure operand was
// not recovered: the HostKernel built in the function of the launch's
// getDeviceKernelInfo<Kernel>() marker or, when none is, in its direct
// callers (handler::StoreLambda<Name, Type> builds the HostKernel and calls
// setDeviceKernelInfo<Name>, which holds the marker). Null unless exactly one
// closure record is found.
static cir::RecordType
hostClosureNearMarker(cir::CallOp marker, HostCallers &callers,
                      mlir::SymbolTableCollection &symbols) {
  if (!marker)
    return {};
  auto fn = marker->getParentOfType<cir::FuncOp>();
  if (!fn)
    return {};
  llvm::SmallVector<cir::RecordType, 2> found;
  ctorClosuresIn(fn, symbols, found);
  if (found.empty())
    for (cir::FuncOp caller : callers.callersOf(fn))
      ctorClosuresIn(caller, symbols, found);
  if (found.empty())
    return {};
  for (cir::RecordType rec : found)
    if (!rec || rec != found.front())
      return {};
  return found.front();
}

// The DEVICE closure record of a spir_kernel: the type of the closure object
// CIRGen tags with `cir.sycl_kernel_object`.
static cir::RecordType deviceClosureRecord(cir::FuncOp kernel) {
  mlir::Value obj = cir::getSYCLKernelObject(kernel);
  if (!obj)
    return {};
  auto alloca = obj.getDefiningOp<cir::AllocaOp>();
  return alloca ? mlir::dyn_cast<cir::RecordType>(alloca.getAllocaType())
                : cir::RecordType();
}

//===----------------------------------------------------------------------===//
// Layout flattening
//===----------------------------------------------------------------------===//

// A record's C++ name without the device-lowering prefix.
static llvm::StringRef recordName(cir::RecordType rec) {
  if (!rec || !rec.getName())
    return {};
  llvm::StringRef name = rec.getName().getValue();
  name.consume_front("__cir_lowered.");
  return name;
}

// Whether a record is compared by size only: a union (its variants overlay),
// an incomplete record, or a SYCL library type, whose host and device
// representations differ by design.
static bool isOpaqueRecord(cir::RecordType rec) {
  return rec.isUnion() || rec.isIncomplete() ||
         recordName(rec).starts_with("sycl::");
}

// One leaf of a flattened closure.
struct Leaf {
  uint64_t offset = 0;
  uint64_t size = 0;
  // The kind in plain terms ("int32", "float", "pointer", "array of 4 x
  // int32", "sycl::half"): equal kinds of equal size are the same leaf.
  std::string kind;
  // Compared by size only (an opaque record, a type with no plain kind).
  bool opaque = false;
  // The opaque record, for the accessor check.
  mlir::Type type;
  // The index of the closure member the leaf is in.
  unsigned top = 0;
};

class Flattener {
public:
  explicit Flattener(const mlir::DataLayout &layout) : layout(layout) {}

  void flatten(cir::RecordType closure, llvm::SmallVectorImpl<Leaf> &out) {
    llvm::ArrayRef<cir::RecordMemberKind> kinds = closure.getMemberKinds();
    for (auto [i, member] : llvm::enumerate(closure.getMembers())) {
      if (i < kinds.size() && (kinds[i] == cir::RecordMemberKind::Pad ||
                               kinds[i] == cir::RecordMemberKind::Empty))
        continue;
      add(member, closure.getElementOffset(layout, i), i, out);
    }
  }

private:
  const mlir::DataLayout &layout;
  llvm::DenseMap<mlir::Type, std::string> signatures;

  uint64_t sizeOf(mlir::Type type) { return layout.getTypeSize(type); }

  // The plain kind of a non-record type, or empty if it has none.
  static std::string scalarKind(mlir::Type type) {
    if (auto i = mlir::dyn_cast<cir::IntType>(type))
      return "int" + std::to_string(i.getWidth());
    if (mlir::isa<cir::BoolType>(type))
      return "bool";
    if (mlir::isa<cir::SingleType>(type))
      return "float";
    if (mlir::isa<cir::DoubleType>(type))
      return "double";
    if (mlir::isa<cir::FP16Type>(type))
      return "half";
    if (mlir::isa<cir::BF16Type>(type))
      return "bfloat16";
    if (mlir::isa<cir::FP80Type>(type))
      return "x86_fp80";
    if (mlir::isa<cir::FP128Type>(type))
      return "fp128";
    if (mlir::isa<cir::LongDoubleType>(type))
      return "long double";
    if (mlir::isa<cir::PointerType>(type))
      return "pointer";
    if (mlir::isa<cir::BitFieldType>(type))
      return "bit-field unit";
    return {};
  }

  // The kind of an array/vector/complex element, or of a record element as
  // the list of its leaves. Empty if some part has no plain kind.
  std::string signature(mlir::Type type) {
    auto it = signatures.find(type);
    if (it != signatures.end())
      return it->second;
    std::string sig;
    if (auto rec = mlir::dyn_cast<cir::RecordType>(type)) {
      if (isOpaqueRecord(rec)) {
        sig = recordName(rec).str();
        if (sig.empty())
          sig = "<opaque>";
        sig += "(" + std::to_string(sizeOf(type)) + " bytes)";
      } else {
        llvm::SmallVector<Leaf> leaves;
        flatten(rec, leaves);
        sig = "{";
        for (auto [i, leaf] : llvm::enumerate(leaves)) {
          if (leaf.opaque && leaf.kind.empty()) {
            sig.clear();
            break;
          }
          sig += (i ? ", " : "") + leaf.kind + " at " +
                 std::to_string(leaf.offset);
        }
        if (!sig.empty())
          sig += "}";
      }
    } else if (std::string k = kindOf(type); !k.empty()) {
      sig = k;
    }
    signatures[type] = sig;
    return sig;
  }

  // The plain kind of any non-record type, or empty.
  std::string kindOf(mlir::Type type) {
    if (std::string k = scalarKind(type); !k.empty())
      return k;
    auto elementsOf = [&](mlir::Type elt, uint64_t n,
                          llvm::StringRef what) -> std::string {
      std::string sig = signature(elt);
      if (sig.empty())
        return {};
      return (what + " of " + llvm::Twine(n) + " x " + sig).str();
    };
    if (auto arr = mlir::dyn_cast<cir::ArrayType>(type))
      return elementsOf(arr.getElementType(), arr.getSize(), "array");
    if (auto vec = mlir::dyn_cast<cir::VectorType>(type))
      return elementsOf(vec.getElementType(), vec.getSize(), "vector");
    if (auto cplx = mlir::dyn_cast<cir::ComplexType>(type)) {
      std::string sig = signature(cplx.getElementType());
      return sig.empty() ? std::string() : "complex of " + sig;
    }
    return {};
  }

  void add(mlir::Type type, uint64_t offset, unsigned top,
           llvm::SmallVectorImpl<Leaf> &out) {
    if (auto rec = mlir::dyn_cast<cir::RecordType>(type)) {
      if (!isOpaqueRecord(rec)) {
        llvm::ArrayRef<cir::RecordMemberKind> kinds = rec.getMemberKinds();
        for (auto [i, member] : llvm::enumerate(rec.getMembers())) {
          if (i < kinds.size() && (kinds[i] == cir::RecordMemberKind::Pad ||
                                   kinds[i] == cir::RecordMemberKind::Empty))
            continue;
          add(member, offset + rec.getElementOffset(layout, i), top, out);
        }
        return;
      }
      Leaf leaf;
      leaf.offset = offset;
      leaf.size = sizeOf(type);
      leaf.kind = recordName(rec).str();
      leaf.opaque = true;
      leaf.type = type;
      leaf.top = top;
      out.push_back(leaf);
      return;
    }
    Leaf leaf;
    leaf.offset = offset;
    leaf.size = sizeOf(type);
    leaf.kind = kindOf(type);
    leaf.opaque = leaf.kind.empty();
    leaf.type = type;
    leaf.top = top;
    out.push_back(leaf);
  }
};

static bool sameLeaf(const Leaf &a, const Leaf &b) {
  if (a.size != b.size)
    return false;
  // An opaque leaf is compared by size only.
  if (a.opaque || b.opaque)
    return true;
  return a.kind == b.kind;
}

static std::string describe(const Leaf *leaf, uint64_t offset) {
  if (!leaf)
    return "no field at offset " + std::to_string(offset);
  std::string kind = leaf->kind.empty() ? "an unknown type" : leaf->kind;
  if (leaf->opaque && !leaf->kind.empty())
    kind += " (" + std::to_string(leaf->size) + " bytes)";
  return kind + " at offset " + std::to_string(leaf->offset);
}

//===----------------------------------------------------------------------===//
// Accessor check
//===----------------------------------------------------------------------===//

// The top-level template arguments of a C++ class name `T<a, b<c, d>, e>`,
// or none if the name does not have that shape.
static std::optional<llvm::SmallVector<llvm::StringRef, 6>>
templateArgs(llvm::StringRef name, llvm::StringRef &templateName) {
  size_t open = name.find('<');
  if (open == llvm::StringRef::npos || !name.ends_with(">"))
    return std::nullopt;
  templateName = name.take_front(open);
  llvm::StringRef body = name.drop_front(open + 1).drop_back();
  llvm::SmallVector<llvm::StringRef, 6> args;
  int depth = 0;
  size_t start = 0;
  for (size_t i = 0; i < body.size(); ++i) {
    char c = body[i];
    if (c == '<' || c == '(')
      ++depth;
    else if (c == '>' || c == ')')
      --depth;
    else if (c == ',' && depth == 0) {
      args.push_back(body.slice(start, i).trim());
      start = i + 1;
    }
    if (depth < 0)
      return std::nullopt;
  }
  if (depth != 0)
    return std::nullopt;
  args.push_back(body.drop_front(start).trim());
  return args;
}

// The C++ spellings of a builtin scalar element type, as a template argument
// of an accessor name prints it.
static llvm::SmallVector<llvm::StringRef, 2> scalarSpellings(mlir::Type type) {
  if (mlir::isa<cir::BoolType>(type))
    return {"bool"};
  if (mlir::isa<cir::SingleType>(type))
    return {"float"};
  if (mlir::isa<cir::DoubleType>(type))
    return {"double"};
  if (mlir::isa<cir::FP16Type>(type))
    return {"_Float16", "__fp16"};
  auto i = mlir::dyn_cast<cir::IntType>(type);
  if (!i)
    return {};
  bool s = i.isSigned();
  switch (i.getWidth()) {
  case 8:
    return s ? llvm::SmallVector<llvm::StringRef, 2>{"signed char", "char"}
             : llvm::SmallVector<llvm::StringRef, 2>{"unsigned char", "char"};
  case 16:
    return {s ? "short" : "unsigned short"};
  case 32:
    return {s ? "int" : "unsigned int"};
  case 64:
    return s ? llvm::SmallVector<llvm::StringRef, 2>{"long", "long long"}
             : llvm::SmallVector<llvm::StringRef, 2>{"unsigned long",
                                                     "unsigned long long"};
  default:
    return {};
  }
}

static bool isBuiltinScalarSpelling(llvm::StringRef s) {
  static const char *const kSpellings[] = {
      "bool",   "float",          "double",      "_Float16",
      "__fp16", "char",           "signed char", "unsigned char",
      "short",  "unsigned short", "int",         "unsigned int",
      "long",   "unsigned long",  "long long",   "unsigned long long"};
  return llvm::is_contained(kSpellings, s);
}

static llvm::StringRef stripConst(llvm::StringRef s) {
  s = s.trim();
  s.consume_front("const ");
  s.consume_back(" const");
  return s.trim();
}

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

struct OffloadDetectHostDeviceClosureLayoutMismatchPass
    : public impl::OffloadDetectHostDeviceClosureLayoutMismatchBase<
          OffloadDetectHostDeviceClosureLayoutMismatchPass> {
  using Base::Base;
  void runOnOperation() override;

private:
  void checkKernel(cir::FuncOp kernel, cir::RecordType hostRec,
                   cir::RecordType devRec, const mlir::DataLayout &hostLayout,
                   const mlir::DataLayout &devLayout,
                   const cir::SYCLClosureLayout &devFields);
  void unchecked(mlir::Location loc, const llvm::Twine &why) {
    LLVM_DEBUG(llvm::dbgs()
               << "closure-layout: " << loc << ": " << why << "\n");
    if (reportUnchecked)
      mlir::emitRemark(loc) << why;
  }
};

void OffloadDetectHostDeviceClosureLayoutMismatchPass::checkKernel(
    cir::FuncOp kernel, cir::RecordType hostRec, cir::RecordType devRec,
    const mlir::DataLayout &hostLayout, const mlir::DataLayout &devLayout,
    const cir::SYCLClosureLayout &devFields) {
  // Per-field get_member ops of the device closure: their names label the
  // diagnostic and their locations anchor it at the source line of the
  // capture rather than at the whole kernel.
  llvm::DenseMap<uint64_t, cir::GetMemberOp> fieldOps;
  kernel.walk([&](cir::GetMemberOp gm) {
    if (gm.getAddr().getType().getPointee() == devRec)
      fieldOps.try_emplace(gm.getIndex(), gm);
  });
  auto nameOf = [&](uint64_t i) -> std::string {
    auto it = fieldOps.find(i);
    if (it != fieldOps.end())
      return "'" + it->second.getName().str() + "'";
    return "#" + std::to_string(i);
  };
  auto locOf = [&](uint64_t i) -> mlir::Location {
    auto it = fieldOps.find(i);
    return it != fieldOps.end() ? it->second.getLoc() : kernel.getLoc();
  };
  auto warn = [&](mlir::Location loc, const llvm::Twine &what) {
    ++numWarnings;
    mlir::InFlightDiagnostic diag = mlir::emitWarning(loc);
    diag << "SYCL kernel captured-closure layout differs between host and "
            "device ("
         << what
         << "); likely a #ifdef __SYCL_DEVICE_ONLY__ or divergent [=] "
            "capture -- undefined behavior "
         << kTag;
    if (loc != kernel.getLoc())
      diag.attachNote(kernel.getLoc()) << "in the SYCL kernel defined here";
  };

  llvm::SmallVector<Leaf> host, dev;
  Flattener(hostLayout).flatten(hostRec, host);
  Flattener(devLayout).flatten(devRec, dev);
  LLVM_DEBUG({
    llvm::dbgs() << "closure-layout: @" << kernel.getSymName()
                 << " checked: " << host.size() << " host / " << dev.size()
                 << " device leaves; fields";
    for (unsigned i = 0, e = devRec.getNumElements(); i != e; ++i)
      llvm::dbgs() << " " << nameOf(i);
    llvm::dbgs() << "\n";
  });

  // The device member at a byte offset (for the anchor of a host-only leaf).
  auto devMemberAt = [&](uint64_t offset) -> std::optional<unsigned> {
    for (const Leaf &leaf : dev)
      if (leaf.offset <= offset && offset < leaf.offset + leaf.size)
        return leaf.top;
    std::optional<unsigned> best;
    for (unsigned i = 0, e = devRec.getNumElements(); i != e; ++i)
      if (devRec.getElementOffset(devLayout, i) <= offset)
        best = i;
    return best;
  };

  // Walk both leaf lists by offset; one warning per device member.
  llvm::DenseSet<unsigned> warned;
  auto mismatch = [&](const Leaf *h, const Leaf *d) {
    uint64_t offset = d ? d->offset : h->offset;
    std::optional<unsigned> member = d ? d->top : devMemberAt(offset);
    if (member && !warned.insert(*member).second)
      return;
    std::string what = "captured field " +
                       (member ? nameOf(*member) : std::string("<unknown>")) +
                       ": host: " + describe(h, offset) +
                       ", device: " + describe(d, offset);
    warn(member ? locOf(*member) : kernel.getLoc(), what);
  };
  size_t i = 0, j = 0;
  while (i < host.size() || j < dev.size()) {
    if (i < host.size() && j < dev.size() && host[i].offset == dev[j].offset) {
      if (!sameLeaf(host[i], dev[j]))
        mismatch(&host[i], &dev[j]);
      ++i, ++j;
    } else if (j == dev.size() ||
               (i < host.size() && host[i].offset < dev[j].offset)) {
      mismatch(&host[i], nullptr);
      ++i;
    } else {
      mismatch(nullptr, &dev[j]);
      ++j;
    }
  }
  if (host.size() != dev.size())
    warn(kernel.getLoc(), "field count differs: host " +
                              llvm::Twine(host.size()) + ", device " +
                              llvm::Twine(dev.size()) + " fields");

  // Accessors: the host template arguments against the device parameters.
  struct AccessorParts {
    llvm::StringRef kind;
    mlir::ArrayAttr path;
    mlir::Type ptr, accessRange;
    unsigned numPtr = 0;
  };
  llvm::SmallVector<AccessorParts, 4> accessors;
  for (unsigned a = 0, e = kernel.getNumArguments(); a != e; ++a) {
    std::optional<cir::SYCLKernelParam> param =
        cir::getSYCLKernelParam(kernel, a);
    if (!param ||
        (param->kind != "accessor" && param->kind != "local_accessor"))
      continue;
    auto it = llvm::find_if(accessors, [&](const AccessorParts &p) {
      return p.path == param->path && p.kind == param->kind;
    });
    if (it == accessors.end()) {
      accessors.push_back({param->kind, param->path, {}, {}, 0});
      it = std::prev(accessors.end());
    }
    mlir::Type argType = kernel.getArgumentTypes()[a];
    if (param->accessorPart == "ptr") {
      ++it->numPtr;
      it->ptr = argType;
    } else if (param->accessorPart == "access_range") {
      it->accessRange = argType;
    }
  }
  for (const AccessorParts &acc : accessors) {
    std::string field =
        "'" +
        llvm::join(
            llvm::map_range(acc.path.getAsRange<mlir::StringAttr>(),
                            [](mlir::StringAttr s) { return s.getValue(); }),
            ".") +
        "'";
    mlir::Location loc = kernel.getLoc();
    std::optional<std::pair<int64_t, mlir::Type>> devField =
        devFields.getField(devRec, acc.path, devLayout);
    if (!devField) {
      ++numAccessorsUnchecked;
      unchecked(loc, "SYCL kernel captured " + acc.kind + " " + field +
                         " not checked: its closure path is not known");
      continue;
    }
    if (!acc.path.empty())
      if (auto top = mlir::dyn_cast<mlir::StringAttr>(acc.path[0]))
        for (auto &[idx, gm] : fieldOps)
          if (gm.getName() == top.getValue())
            loc = gm.getLoc();
    const Leaf *hostLeaf = nullptr;
    for (const Leaf &leaf : host)
      if (leaf.offset == static_cast<uint64_t>(devField->first))
        hostLeaf = &leaf;
    auto hostAcc =
        hostLeaf ? mlir::dyn_cast_if_present<cir::RecordType>(hostLeaf->type)
                 : cir::RecordType();
    llvm::StringRef hostTemplate;
    std::optional<llvm::SmallVector<llvm::StringRef, 6>> args =
        hostAcc ? templateArgs(recordName(hostAcc), hostTemplate)
                : std::nullopt;
    llvm::StringRef expected = acc.kind == "accessor"
                                   ? "sycl::_V1::accessor"
                                   : "sycl::_V1::local_accessor";
    if (!args || args->size() < 2 || hostTemplate != expected ||
        acc.numPtr != 1) {
      ++numAccessorsUnchecked;
      unchecked(loc, "SYCL kernel captured " + acc.kind + " " + field +
                         " not checked: no host " + acc.kind +
                         " of a known shape at its offset");
      continue;
    }
    // Dimensions: the host template argument against the rank of the device
    // access range (range<N>). A 0-dimensional accessor is lowered with
    // 1-dimensional ranges.
    if (auto range =
            pointeeRecord(acc.accessRange)
                ? pointeeRecord(acc.accessRange)
                : mlir::dyn_cast_if_present<cir::RecordType>(acc.accessRange)) {
      llvm::StringRef rangeTemplate;
      std::optional<llvm::SmallVector<llvm::StringRef, 6>> rangeArgs =
          templateArgs(recordName(range), rangeTemplate);
      unsigned hostDims = 0, devDims = 0;
      if (rangeArgs && rangeArgs->size() == 1 &&
          rangeTemplate == "sycl::_V1::range" &&
          !(*args)[1].getAsInteger(10, hostDims) &&
          !(*rangeArgs)[0].getAsInteger(10, devDims) &&
          std::max(hostDims, 1u) != devDims) {
        ++numAccessorsChecked;
        warn(loc, "captured " + acc.kind + " " + field +
                      ": host: " + llvm::Twine(hostDims) +
                      " dimensions, device: " + llvm::Twine(devDims) +
                      " dimensions");
        continue;
      }
    }
    // Element type: the host template argument against the pointee of the
    // device data pointer.
    llvm::StringRef hostElt = stripConst((*args)[0]);
    auto ptrTy = mlir::dyn_cast_if_present<cir::PointerType>(acc.ptr);
    if (!ptrTy) {
      ++numAccessorsUnchecked;
      unchecked(loc, "SYCL kernel captured " + acc.kind + " " + field +
                         " not checked: its data pointer is not a pointer");
      continue;
    }
    mlir::Type pointee = ptrTy.getPointee();
    std::string devElt;
    bool differs = false;
    if (auto rec = mlir::dyn_cast<cir::RecordType>(pointee)) {
      // A record element: its name against the host spelling (a builtin
      // scalar spelling never names a record).
      devElt = recordName(rec).str();
      differs = isBuiltinScalarSpelling(hostElt) || devElt != hostElt;
    } else if (isBuiltinScalarSpelling(hostElt)) {
      // A scalar element against a builtin spelling. Anything else (an enum,
      // a typedef the name kept) is not decided.
      llvm::SmallVector<llvm::StringRef, 2> spellings =
          scalarSpellings(pointee);
      if (!spellings.empty()) {
        devElt = spellings.front().str();
        differs = !llvm::is_contained(spellings, hostElt);
      }
    }
    if (devElt.empty()) {
      ++numAccessorsUnchecked;
      unchecked(loc, "SYCL kernel captured " + acc.kind + " " + field +
                         " not checked: element type '" + hostElt +
                         "' cannot be compared");
      continue;
    }
    ++numAccessorsChecked;
    if (differs)
      warn(loc, "captured " + acc.kind + " " + field +
                    ": host: element type '" + hostElt +
                    "', device: element type '" + devElt + "'");
  }
}

void OffloadDetectHostDeviceClosureLayoutMismatchPass::runOnOperation() {
  mlir::ModuleOp container = getOperation();
  if (!cir::isOffloadContainer(container))
    return;

  // Analysis only: nothing below mutates the IR.
  markAllAnalysesPreserved();

  mlir::ModuleOp hostModule = cir::getOffloadContainerHostModule(container);
  if (!hostModule)
    return;
  cir::KernelBindingTable &table = getAnalysis<cir::KernelBindingTable>();
  mlir::SymbolTableCollection symbols;
  HostCallers callers(hostModule, symbols);
  mlir::DataLayout hostLayout(hostModule);
  llvm::DenseMap<mlir::Operation *, std::unique_ptr<mlir::DataLayout>> layouts;
  llvm::DenseMap<mlir::Operation *, std::unique_ptr<cir::SYCLClosureLayout>>
      closureLayouts;

  for (const auto &entry : table) {
    const cir::KernelBinding &binding = entry.second;
    if (binding.kind != cir::BindingKind::SYCLName)
      continue;

    // The host closure record: every launch site must give the same one.
    cir::RecordType hostRec;
    bool ambiguous = binding.launchSites.empty();
    for (const cir::LaunchSite &site : binding.launchSites) {
      cir::RecordType rec =
          hostClosureOfOperand(site.getSYCLClosure(), symbols);
      if (!rec)
        rec = hostClosureNearMarker(site.syclMarkerCall, callers, symbols);
      if (!rec || (hostRec && rec != hostRec)) {
        ambiguous = true;
        break;
      }
      hostRec = rec;
    }

    for (cir::FuncOp kernel : binding.deviceKernels) {
      if (kernel.getCallingConv() != cir::CallingConv::SpirKernel ||
          kernel.getBody().empty())
        continue;
      cir::RecordType devRec = deviceClosureRecord(kernel);
      if (!devRec || devRec.isIncomplete()) {
        ++numKernelsUnchecked;
        unchecked(kernel.getLoc(), "SYCL kernel closure layout not checked: "
                                   "no device closure object");
        continue;
      }
      if (ambiguous || !hostRec || hostRec.isIncomplete()) {
        ++numKernelsUnchecked;
        unchecked(kernel.getLoc(), "SYCL kernel closure layout not checked: "
                                   "no unique host closure record");
        continue;
      }
      auto devModule = kernel->getParentOfType<mlir::ModuleOp>();
      std::unique_ptr<mlir::DataLayout> &devLayout = layouts[devModule];
      if (!devLayout)
        devLayout = std::make_unique<mlir::DataLayout>(devModule);
      std::unique_ptr<cir::SYCLClosureLayout> &devFields =
          closureLayouts[devModule];
      if (!devFields)
        devFields = std::make_unique<cir::SYCLClosureLayout>(devModule);
      ++numKernelsChecked;
      checkKernel(kernel, hostRec, devRec, hostLayout, *devLayout, *devFields);
    }
  }
}

} // namespace

std::unique_ptr<Pass>
mlir::createOffloadDetectHostDeviceClosureLayoutMismatchPass() {
  return std::make_unique<OffloadDetectHostDeviceClosureLayoutMismatchPass>();
}
