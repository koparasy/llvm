//===- PointsToModels.cpp - Library models for CIR points-to --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The registry and the generic C/C++ models. Every model states the section
// of the C standard (ISO/IEC 9899:2018, "C17") or the C++ standard
// (ISO/IEC 14882:2020, "C++20") it relies on, or the Itanium C++ ABI for the
// __cxa_* runtime that implements C++ semantics.
//
// Which callees a model accepts:
//
//  - The replaceable C++ allocation functions (operator new / delete) only
//    when they are declarations: a program may replace them
//    ([replacement.functions]), and a replacement may keep the pointer it
//    returns somewhere, which a body would show and the model would not.
//  - The C library functions also when the module defines them with external
//    linkage: defining a C library name with external linkage is undefined
//    behaviour (C17 7.1.3p2, C++20 [extern.names]), so the body is the
//    library's and the model describes it.
//  - Everything else only as a declaration.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToModels.h"
#include "mlir/IR/SymbolTable.h"
#include "clang/CIR/Dialect/Analysis/CIRPointerDecomposition.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace cir;
using namespace cir::pointsto;

//===----------------------------------------------------------------------===//
// ModelContext
//===----------------------------------------------------------------------===//

std::optional<int64_t> ModelContext::getConstantArg(const CallSiteRecord &site,
                                                    unsigned index) const {
  if (site.synthetic)
    return std::nullopt;
  auto call = dyn_cast_if_present<CIRCallOpInterface>(site.call);
  if (!call || index >= call.getNumArgOperands())
    return std::nullopt;
  return getConstantIndex(call.getArgOperand(index));
}

void ModelContext::forEachPointerArg(const CallSiteRecord &site,
                                     llvm::function_ref<void(Cell)> fn) {
  for (const std::optional<Cell> &arg : site.args)
    if (arg)
      fn(*arg);
  for (const auto &entry : site.aggregateArgs)
    for (const auto &slot : entry.second)
      if (slot.second)
        fn(slot.second);
}

cir::FuncOp ModelContext::lookupFunction(Operation *at,
                                         FlatSymbolRefAttr sym) const {
  if (!at || !sym)
    return {};
  if (symbols)
    return symbols->lookupNearestSymbolFrom<cir::FuncOp>(at, sym);
  return SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(at, sym);
}

const Reason *ModelContext::makeReason(const CallSiteRecord &site,
                                       Reason::Kind kind, StringRef detail) {
  return getGraph().makeReason(kind, site.call, detail);
}

void ModelContext::modRef(std::optional<Cell> cell, bool mod, bool ref) {
  if (!cell)
    return;
  NodeFlags flags = NodeFlags::None;
  if (mod) {
    flags |= NodeFlags::Mod;
    if (writes)
      writes->push_back({nullptr, *cell, /*reachable=*/false});
  }
  if (ref)
    flags |= NodeFlags::Ref;
  getGraph().addFlags(*cell, flags);
}

void ModelContext::writeUnknown(std::optional<Cell> cell,
                                std::optional<uint64_t> size) {
  if (!cell)
    return;
  if (!size)
    return modRef(cell, /*mod=*/true, /*ref=*/false);
  getGraph().addFlags(*cell, NodeFlags::Mod);
  if (writes)
    writes->push_back({nullptr, *cell, /*reachable=*/false, *size});
}

Cell ModelContext::makeHeapObject(const CallSiteRecord &site,
                                  std::optional<int64_t> size) {
  Graph &graph = getGraph();
  Node *node = graph.makeNode(Origin{Origin::Kind::HeapCall, site.call},
                              NodeFlags::Heap);
  if (size && *size > 0)
    graph.growSize(node, static_cast<uint64_t>(*size));
  return Cell(node);
}

void ModelContext::setResult(const CallSiteRecord &site, Cell cell) {
  if (!site.result)
    return;
  Graph &graph = getGraph();
  graph.unify(*site.result, cell,
              graph.makeReason(Reason::Kind::MergedAt, site.call));
}

void ModelContext::setResultUnknown(const CallSiteRecord &site,
                                    NodeFlags status, const Reason *why) {
  Graph &graph = getGraph();
  NodeFlags flags = NodeFlags::UnknownObj | status;
  if (site.result)
    graph.addFlags(*site.result, flags, why);
  if (site.resultSlots)
    for (const auto &slot : *site.resultSlots)
      if (slot.second)
        graph.addFlags(slot.second, flags, why);
}

void ModelContext::applyUnknownCall(CallSiteRecord &site, const Reason *why) {
  // SOUNDNESS: the same rule as LocalBuilder::applyUnknownCall.
  forEachPointerArg(site, [&](Cell cell) {
    defer(DeferredEffect::Kind::Reachable, cell,
          NodeFlags::External | NodeFlags::Mod | NodeFlags::Ref, why);
  });
  setResultUnknown(site, NodeFlags::External | NodeFlags::Incomplete, why);
  site.clobbersGlobals = true;
}

void ModelContext::addIndirectCall(const CallSiteRecord &site, Cell callee,
                                   ArrayRef<std::optional<Cell>> args) {
  assert(extraCalls && "addIndirectCall without a call sink");
  CallSiteRecord call;
  call.call = site.call;
  call.synthetic = true;
  call.calleePtr = callee;
  call.args.assign(args.begin(), args.end());
  call.kind = CallKind::IndirectDeferred;
  extraCalls->push_back(std::move(call));
}

void cir::pointsto::finishModelWrites(const PointsToModel &model,
                                      const CallSiteRecord &site,
                                      SmallVectorImpl<ModelWrite> &writes,
                                      size_t first) {
  for (size_t i = first; i < writes.size(); ++i)
    writes[i].op = site.call;
  if (model.declaresAllWrites())
    return;
  // SOUNDNESS: the model may write what it is handed in ways it does not
  // report (see PointsToModel::declaresAllWrites).
  ModelContext::forEachPointerArg(
      site, [&](Cell cell) { writes.push_back({site.call, cell, false}); });
  if (site.result)
    writes.push_back({site.call, *site.result, false});
  if (site.resultSlots)
    for (const auto &slot : *site.resultSlots)
      if (slot.second)
        writes.push_back({site.call, slot.second, false});
}

//===----------------------------------------------------------------------===//
// Registry
//===----------------------------------------------------------------------===//

PointsToModel::~PointsToModel() = default;

PointsToModelRegistry::PointsToModelRegistry() = default;
PointsToModelRegistry::~PointsToModelRegistry() = default;

const PointsToModel *
PointsToModelRegistry::addRoleModel(StringRef role,
                                    std::unique_ptr<PointsToModel> model) {
  const PointsToModel *m = model.get();
  models.push_back(std::move(model));
  byRole[role].push_back(m);
  return m;
}

const PointsToModel *
PointsToModelRegistry::addSymbolModel(ArrayRef<StringRef> symbols,
                                      std::unique_ptr<PointsToModel> model) {
  const PointsToModel *m = model.get();
  models.push_back(std::move(model));
  for (StringRef sym : symbols)
    bySymbol[sym].push_back(m);
  return m;
}

const PointsToModel *
PointsToModelRegistry::addPredicateModel(std::unique_ptr<PointsToModel> model) {
  const PointsToModel *m = model.get();
  models.push_back(std::move(model));
  predicates.push_back(m);
  return m;
}

const PointsToModel *PointsToModelRegistry::lookup(CIRCallOpInterface call,
                                                   cir::FuncOp callee,
                                                   StringRef calleeName) const {
  bool defined = callee && !callee.isDeclaration();
  auto accepts = [&](const PointsToModel *m) {
    return (!defined || m->replacesDefinition()) && m->matches(call, callee);
  };
  if (callee)
    if (auto role = callee->getAttrOfType<StringAttr>(
            CIRDialect::getSYCLHostAPIAttrName())) {
      auto it = byRole.find(role.getValue());
      if (it != byRole.end())
        for (const PointsToModel *m : it->second)
          if (accepts(m))
            return m;
    }
  auto it = bySymbol.find(calleeName);
  if (it != bySymbol.end())
    for (const PointsToModel *m : it->second)
      if (accepts(m))
        return m;
  for (const PointsToModel *m : predicates)
    if (accepts(m))
      return m;
  return fallback ? fallback->lookup(call, callee, calleeName) : nullptr;
}

//===----------------------------------------------------------------------===//
// Generic models
//===----------------------------------------------------------------------===//

namespace {

/// Which definitions of a matched symbol a model accepts.
enum class Accept : uint8_t {
  /// Declarations only.
  Declarations,
  /// Declarations, and definitions with external linkage (a C library name:
  /// defining one is UB, C17 7.1.3p2).
  LibraryDefinitions,
};

/// The number of arguments of the call, or of the callee for a synthetic
/// site.
unsigned getNumArgs(CIRCallOpInterface call, cir::FuncOp callee) {
  if (call)
    return call.getNumArgOperands();
  return callee ? callee.getFunctionType().getNumInputs() : 0;
}

/// A model matched by symbol name, with a check of the definition rule and
/// of the minimum argument count.
class SymbolModel : public PointsToModel {
public:
  SymbolModel(StringRef name, StringRef citation, Accept accept,
              unsigned minArgs)
      : modelName(name), citation(citation), accept(accept), minArgs(minArgs) {}

  bool matches(CIRCallOpInterface call, cir::FuncOp callee) const override {
    if (callee && !callee.isDeclaration()) {
      if (accept != Accept::LibraryDefinitions)
        return false;
      if (callee.isPrivate() || cir::isLocalLinkage(callee.getLinkage()))
        return false;
    }
    return getNumArgs(call, callee) >= minArgs;
  }

  StringRef name() const override { return modelName; }
  StringRef specCitation() const override { return citation; }
  /// A C library function defined with external linkage is the library's
  /// (C17 7.1.3p2): the model describes the body.
  bool replacesDefinition() const override {
    return accept == Accept::LibraryDefinitions;
  }
  /// Every generic model reports its writes (audited for T22: the
  /// allocators, the string functions, the pointer-free and escaping
  /// functions write only through modRef, defer and addMemTransfer).
  bool declaresAllWrites() const override { return true; }

private:
  StringRef modelName;
  StringRef citation;
  Accept accept;
  unsigned minArgs;
};

//===--------------------------------------------------------------------===//
// Allocators
//===--------------------------------------------------------------------===//

/// How an allocation function is called.
enum class AllocKind : uint8_t {
  /// f(size, ...): operator new, malloc, __cxa_allocate_exception.
  SizeArg0,
  /// calloc(count, size).
  Calloc,
  /// aligned_alloc(alignment, size).
  SizeArg1,
  /// posix_memalign(void **out, alignment, size): the result is written
  /// through the first argument.
  PosixMemalign,
  /// realloc(p, size): the contents of p move to the result.
  Realloc,
  /// strdup(s), strndup(s, n): a copy of a string, which holds no pointer.
  StrDup,
};

/// SOUNDNESS: an allocation function returns storage disjoint from every
/// other live object (C17 7.22.3p1, "each such allocation shall yield a
/// pointer to an object disjoint from any other object"; C++20
/// [basic.stc.dynamic.allocation]p2, "the value returned ... shall be
/// distinct from any other value previously returned and not since
/// deallocated"), so a call site gets a fresh Heap node. The storage holds
/// no pointer until one is stored into it (indeterminate or zero values,
/// C17 7.22.3.4p2 and 7.22.3.2p2), so the node starts without links.
class AllocatorModel : public SymbolModel {
public:
  AllocatorModel(StringRef name, StringRef citation, Accept accept,
                 AllocKind kind)
      : SymbolModel(name, citation, accept, getMinArgs(kind)), kind(kind) {}

  void apply(CallSiteRecord &site, ModelContext &ctx) const override {
    Graph &graph = ctx.getGraph();
    std::optional<int64_t> size;
    switch (kind) {
    case AllocKind::SizeArg0:
      size = ctx.getConstantArg(site, 0);
      break;
    case AllocKind::Calloc: {
      std::optional<int64_t> n = ctx.getConstantArg(site, 0);
      std::optional<int64_t> m = ctx.getConstantArg(site, 1);
      int64_t product;
      if (n && m && !llvm::MulOverflow(*n, *m, product))
        size = product;
      break;
    }
    case AllocKind::SizeArg1:
    case AllocKind::Realloc:
      size = ctx.getConstantArg(site, 1);
      break;
    case AllocKind::PosixMemalign:
      size = ctx.getConstantArg(site, 2);
      break;
    case AllocKind::StrDup:
      break;
    }
    // Pointer arguments other than the ones handled below (the nothrow tag
    // of operator new) are only read.
    for (auto [i, arg] : llvm::enumerate(site.args))
      if (arg && !(kind == AllocKind::PosixMemalign && i == 0) &&
          !(kind == AllocKind::Realloc && i == 0))
        ctx.modRef(arg, /*mod=*/false, /*ref=*/true);

    Cell heap = ctx.makeHeapObject(site, size);
    // The stored-value queries: calloc's storage starts zeroed (C17
    // 7.22.3.2p2) and strdup's holds a copy of the string, so neither starts
    // indeterminate. Both are reported as writes of unknown value.
    if (kind == AllocKind::Calloc || kind == AllocKind::StrDup)
      ctx.modRef(heap, /*mod=*/true, /*ref=*/false);
    switch (kind) {
    case AllocKind::PosixMemalign:
      // C17 does not have posix_memalign; POSIX.1-2017 specifies it: "upon
      // successful completion, the value pointed to by memptr shall be a
      // multiple of alignment" and the storage is as from malloc().
      if (std::optional<Cell> out = ModelContext::getArg(site, 0)) {
        ctx.modRef(out, /*mod=*/true, /*ref=*/false);
        graph.setLink(*out, heap,
                      ctx.makeReason(site, Reason::Kind::CollapsedAt));
      }
      return;
    case AllocKind::Realloc:
      // C17 7.22.3.5p2: realloc deallocates the old object and returns a
      // new one with the same contents up to the lesser size. The pointers
      // the old object held are the new object's: unifying the two nodes
      // keeps every link, and the old pointer is dead after the call.
      if (std::optional<Cell> old = ModelContext::getArg(site, 0)) {
        ctx.modRef(old, /*mod=*/true, /*ref=*/true);
        graph.unify(heap, *old,
                    ctx.makeReason(site, Reason::Kind::MergedAt, "realloc"));
      }
      break;
    default:
      break;
    }
    ctx.setResult(site, heap);
  }

private:
  static unsigned getMinArgs(AllocKind kind) {
    switch (kind) {
    case AllocKind::SizeArg0:
    case AllocKind::StrDup:
      return 1;
    case AllocKind::Calloc:
    case AllocKind::SizeArg1:
    case AllocKind::Realloc:
      return 2;
    case AllocKind::PosixMemalign:
      return 3;
    }
    llvm_unreachable("unknown allocation kind");
  }

  AllocKind kind;
};

/// The Itanium manglings of the global replaceable allocation and
/// deallocation functions ([new.delete.single], [new.delete.array]), for a
/// 64-bit (m) and a 32-bit (j) size_t.
struct OperatorNewDeleteNames {
  std::vector<std::string> newNames, deleteNames;

  OperatorNewDeleteNames() {
    for (StringRef sizeT : {"m", "j"})
      for (StringRef op : {"nw", "na"})
        for (StringRef suffix : {"", "RKSt9nothrow_t", "St11align_val_t",
                                 "St11align_val_tRKSt9nothrow_t"})
          newNames.push_back(("_Z" + op + sizeT + suffix).str());
    for (StringRef op : {"dl", "da"}) {
      for (StringRef suffix : {"", "RKSt9nothrow_t", "St11align_val_t",
                               "St11align_val_tRKSt9nothrow_t"})
        deleteNames.push_back(("_Z" + op + "Pv" + suffix).str());
      // Sized deallocation, C++14 and later.
      for (StringRef sizeT : {"m", "j"})
        for (StringRef suffix : {"", "St11align_val_t"})
          deleteNames.push_back(("_Z" + op + "Pv" + sizeT + suffix).str());
    }
  }
};

SmallVector<StringRef> toRefs(const std::vector<std::string> &names) {
  return SmallVector<StringRef>(names.begin(), names.end());
}

//===--------------------------------------------------------------------===//
// Deallocators and functions without a pointer effect
//===--------------------------------------------------------------------===//

/// What a function without a pointer effect does with its pointer
/// arguments.
enum class ArgAccess : uint8_t {
  /// Nothing.
  None,
  /// Reads the memory they point to.
  Ref,
  /// Writes (invalidates) the memory they point to: a deallocation.
  Mod,
  /// Reads and writes the memory they point to (not what it points to).
  ModRef,
  /// Reads and writes everything reachable from them, without keeping
  /// them.
  ModRefReachable,
};

/// A function that neither creates, stores, returns nor keeps a pointer the
/// caller can see: its only effect is Mod/Ref on its pointer arguments. A
/// pointer result, if any, points to an unknown object (External).
class NoPointerEffectModel : public SymbolModel {
public:
  NoPointerEffectModel(StringRef name, StringRef citation, Accept accept,
                       ArgAccess access)
      : SymbolModel(name, citation, accept, /*minArgs=*/0), access(access) {}

  void apply(CallSiteRecord &site, ModelContext &ctx) const override {
    switch (access) {
    case ArgAccess::None:
      break;
    case ArgAccess::Ref:
    case ArgAccess::Mod:
    case ArgAccess::ModRef:
      ModelContext::forEachPointerArg(site, [&](Cell cell) {
        ctx.modRef(cell, access != ArgAccess::Ref, access != ArgAccess::Mod);
      });
      break;
    case ArgAccess::ModRefReachable: {
      const Reason *why = nullptr;
      ModelContext::forEachPointerArg(site, [&](Cell cell) {
        ctx.defer(DeferredEffect::Kind::Reachable, cell,
                  NodeFlags::Mod | NodeFlags::Ref, why);
      });
      break;
    }
    }
    if (site.result || site.resultSlots)
      ctx.setResultUnknown(site, NodeFlags::External | NodeFlags::Incomplete,
                           ctx.makeReason(site, Reason::Kind::EscapedTo,
                                          "result of " + name().str()));
  }

private:
  ArgAccess access;
};

/// A function that keeps its pointer arguments to use them later (it
/// registers a callback and its argument): everything reachable from them
/// escapes, but nothing else happens.
class EscapeArgsModel : public SymbolModel {
public:
  using SymbolModel::SymbolModel;

  void apply(CallSiteRecord &site, ModelContext &ctx) const override {
    const Reason *why = ctx.makeReason(site, Reason::Kind::EscapedTo,
                                       ("kept by " + name()).str());
    ModelContext::forEachPointerArg(site, [&](Cell cell) {
      ctx.defer(DeferredEffect::Kind::Reachable, cell,
                NodeFlags::External | NodeFlags::Mod | NodeFlags::Ref, why);
    });
    if (site.result || site.resultSlots)
      ctx.setResultUnknown(site, NodeFlags::External | NodeFlags::Incomplete,
                           why);
  }
};

/// Whether \p name is the Itanium mangling of a function `std::__throw_*`
/// (`_ZSt<len><name>...`), libstdc++'s out-of-line throw helpers.
bool isStdThrowHelper(StringRef name) {
  if (!name.consume_front("_ZSt"))
    return false;
  size_t len;
  if (name.consumeInteger(10, len) || len > name.size())
    return false;
  return name.take_front(len).starts_with("__throw_");
}

/// std::__throw_bad_alloc(), std::__throw_logic_error(const char *), ...:
/// [res.on.exception.handling] lets a library function throw; these helpers
/// construct and throw a standard exception from a message they only read,
/// and do not return. The exception object is the library's.
class StdThrowModel : public NoPointerEffectModel {
public:
  StdThrowModel()
      : NoPointerEffectModel(
            "std-throw",
            "C++20 [res.on.exception.handling]; libstdc++ functexcept.h",
            Accept::Declarations, ArgAccess::Ref) {}

  bool matches(CIRCallOpInterface call, cir::FuncOp callee) const override {
    if (!callee || !callee.isDeclaration() ||
        !isStdThrowHelper(callee.getSymName()))
      return false;
    return NoPointerEffectModel::matches(call, callee);
  }
};

//===--------------------------------------------------------------------===//
// C string and memory functions
//===--------------------------------------------------------------------===//

enum class StringKind : uint8_t {
  /// memcpy(d, s, n), memmove(d, s, n): a copy of n bytes; returns d.
  Copy,
  /// memset(d, c, n): writes bytes, no pointer; returns d.
  Set,
  /// strcpy(d, s): a copy of an unknown length; returns d.
  StrCpy,
  /// strncpy(d, s, n): a copy of at most n bytes; returns d.
  StrNCpy,
  /// strcat(d, s), strncat(d, s, n): a copy to an unknown offset in d;
  /// returns d.
  StrCat,
  /// strlen, strcmp, strncmp, memcmp: read only.
  Read,
  /// strchr, strrchr, strstr, memchr: read only; returns a pointer into the
  /// first argument.
  Search,
};

/// C17 7.24 (<string.h>). SOUNDNESS: the copying functions copy bytes, and
/// a pointer copied byte-wise is a pointer (C17 6.2.6.1p4, the object
/// representation); they are given the same byte-range closure as
/// cir.libc.memcpy. The search functions return a pointer into the object
/// they search (C17 7.24.5.1-8) or null.
class StringModel : public SymbolModel {
public:
  StringModel(StringRef name, StringRef citation, StringKind kind)
      : SymbolModel(name, citation, Accept::LibraryDefinitions,
                    getMinArgs(kind)),
        kind(kind) {}

  void apply(CallSiteRecord &site, ModelContext &ctx) const override {
    Graph &graph = ctx.getGraph();
    std::optional<Cell> dst = ModelContext::getArg(site, 0);
    std::optional<Cell> src = ModelContext::getArg(site, 1);
    switch (kind) {
    case StringKind::Copy:
      if (dst && src)
        ctx.addMemTransfer(site, *dst, *src, ctx.getConstantArg(site, 2));
      break;
    case StringKind::Set:
      ctx.modRef(dst, /*mod=*/true, /*ref=*/false);
      if (std::optional<int64_t> n = ctx.getConstantArg(site, 2);
          dst && n && *n > 0)
        graph.growSize(dst->getNode(), dst->getOffset() + *n);
      break;
    case StringKind::StrCpy:
      if (dst && src)
        ctx.addMemTransfer(site, *dst, *src, std::nullopt);
      break;
    case StringKind::StrNCpy:
      if (dst && src)
        ctx.addMemTransfer(site, *dst, *src, ctx.getConstantArg(site, 2));
      // Past the end of the source string strncpy writes zeros, not the
      // source's bytes (C17 7.24.2.4p3): a write of unknown value too.
      ctx.modRef(dst, /*mod=*/true, /*ref=*/false);
      break;
    case StringKind::StrCat:
      // The copy starts at the terminating null of d, an offset we don't
      // know: collapse d, so that the closure relates its single slot to
      // every slot of s.
      if (dst && src) {
        graph.collapse(dst->getNode(),
                       ctx.makeReason(site, Reason::Kind::CollapsedAt,
                                      "appended at an unknown offset"));
        ctx.addMemTransfer(site, *dst, *src, std::nullopt);
      }
      break;
    case StringKind::Read:
      ModelContext::forEachPointerArg(
          site, [&](Cell cell) { ctx.modRef(cell, false, true); });
      return;
    case StringKind::Search:
      ModelContext::forEachPointerArg(
          site, [&](Cell cell) { ctx.modRef(cell, false, true); });
      if (dst) {
        // At an unknown offset in the searched object.
        graph.collapse(dst->getNode(),
                       ctx.makeReason(site, Reason::Kind::CollapsedAt,
                                      "pointer into the searched range"));
        ctx.setResult(site, *dst);
      } else if (site.result) {
        ctx.setResultUnknown(site, NodeFlags::Incomplete,
                             ctx.makeReason(site, Reason::Kind::EscapedTo,
                                            "search of a null pointer"));
      }
      return;
    }
    // Every copying and setting function returns its first argument.
    if (dst)
      ctx.setResult(site, *dst);
  }

private:
  static unsigned getMinArgs(StringKind kind) {
    switch (kind) {
    case StringKind::Copy:
    case StringKind::Set:
    case StringKind::StrNCpy:
      return 3;
    case StringKind::StrCpy:
    case StringKind::StrCat:
    case StringKind::Search:
      return 2;
    case StringKind::Read:
      return 1;
    }
    llvm_unreachable("unknown string kind");
  }

  StringKind kind;
};

//===--------------------------------------------------------------------===//
// #cir.memory_effects
//===--------------------------------------------------------------------===//

/// The effects the callee and the call site promise: the meet of the two
/// attributes, class by class (each is a valid promise on its own).
std::optional<MemoryEffectsAttr> getMemoryEffects(CIRCallOpInterface call,
                                                  cir::FuncOp callee) {
  StringRef attrName = CIRDialect::getMemoryEffectsAttrName();
  MemoryEffectsAttr a =
      callee ? callee->getAttrOfType<MemoryEffectsAttr>(attrName) : nullptr;
  MemoryEffectsAttr b =
      call ? call->getAttrOfType<MemoryEffectsAttr>(attrName) : nullptr;
  if (!a && !b)
    return std::nullopt;
  if (!a || !b)
    return a ? a : b;
  auto meet = [](ModRefInfo x, ModRefInfo y) {
    return static_cast<ModRefInfo>(static_cast<uint32_t>(x) &
                                   static_cast<uint32_t>(y));
  };
  return MemoryEffectsAttr::get(
      a.getContext(), meet(a.getOther(), b.getOther()),
      meet(a.getArgMem(), b.getArgMem()),
      meet(a.getInaccessibleMem(), b.getInaccessibleMem()),
      meet(a.getErrnoMem(), b.getErrnoMem()),
      meet(a.getTargetMem0(), b.getTargetMem0()),
      meet(a.getTargetMem1(), b.getTargetMem1()));
}

bool hasMod(ModRefInfo info) {
  return static_cast<uint32_t>(info) & static_cast<uint32_t>(ModRefInfo::Mod);
}
bool hasRef(ModRefInfo info) {
  return static_cast<uint32_t>(info) & static_cast<uint32_t>(ModRefInfo::Ref);
}

/// What a #cir.memory_effects attribute buys. Only the `other` and `arg_mem`
/// classes concern memory the module can see: `inaccessible_mem` is memory
/// the module cannot reach, `errno_mem` holds an int, and the target classes
/// are target state, so none of them holds a pointer the analysis tracks.
/// The promise is about memory accesses only; a pointer result may still be
/// one the callee was given or read, so its possible targets become External
/// (see apply):
///
///  - `none` (other = arg_mem = none): no effect on memory. With a pointer
///    result, the argument objects become External.
///  - `readOnly` (no class written): Ref on every node reachable from a
///    pointer argument. With a pointer result, those nodes also become
///    External.
///  - `argMemOnly` (other not written, arg_mem written): the callee may store
///    one argument's object, or a pointer it read, into another argument's
///    memory. CIR has no nocapture attribute, so this is the unknown-call rule
///    over what the arguments reach, sparing only the globals.
///  - `other` written: no promise worth using; the unknown-call rule applies.
///
/// Only declarations are modelled: a definition's body is analysed instead.
class MemoryEffectsModel : public PointsToModel {
public:
  enum class Class : uint8_t { None, ReadOnly, ArgMemOnly };

  static std::optional<Class> classify(MemoryEffectsAttr effects) {
    ModRefInfo other = effects.getOther();
    ModRefInfo arg = effects.getArgMem();
    if (hasMod(other))
      return std::nullopt;
    if (!hasRef(other) && !hasRef(arg) && !hasMod(arg))
      return Class::None;
    if (!hasMod(arg))
      return Class::ReadOnly;
    return Class::ArgMemOnly;
  }

  bool matches(CIRCallOpInterface call, cir::FuncOp callee) const override {
    if (callee && !callee.isDeclaration())
      return false;
    std::optional<MemoryEffectsAttr> effects = getMemoryEffects(call, callee);
    return effects && classify(*effects);
  }

  // SOUNDNESS: a memory_effects promise limits what the callee reads and
  // writes, not what it returns. A pointer result may be an argument (at any
  // offset) or, if the callee reads memory, any pointer it read. Code that
  // later stores through such a result writes the argument's object behind
  // the analysis's back, so the objects the result may point to must be
  // External, not just the fresh result node.
  void apply(CallSiteRecord &site, ModelContext &ctx) const override {
    auto call = dyn_cast_if_present<CIRCallOpInterface>(site.call);
    cir::FuncOp callee;
    if (site.callee)
      callee = ctx.lookupFunction(site.call, site.callee);
    MemoryEffectsAttr effects =
        *getMemoryEffects(site.synthetic ? CIRCallOpInterface() : call, callee);
    Class cls = *classify(effects);
    bool returnsPointer = site.result || site.resultSlots;
    switch (cls) {
    case Class::None: {
      // No memory is read, so a pointer result is computed from the argument
      // pointers themselves (or is a global's address): their objects may be
      // returned, but nothing reachable through them.
      const Reason *why =
          ctx.makeReason(site, Reason::Kind::EscapedTo,
                         "may be returned by a memory_effects(none) call");
      if (returnsPointer)
        ModelContext::forEachPointerArg(site, [&](Cell cell) {
          ctx.getGraph().addFlags(cell, NodeFlags::External, why);
        });
      ctx.setResultUnknown(site, NodeFlags::External | NodeFlags::Incomplete,
                           why);
      return;
    }
    case Class::ReadOnly: {
      // Nothing is written, so nothing is published through memory; but a
      // pointer result may be anything read from what the arguments reach.
      const Reason *why = ctx.makeReason(site, Reason::Kind::EscapedTo,
                                         "may be returned by a read-only call");
      NodeFlags reach = NodeFlags::Ref;
      if (returnsPointer)
        reach |= NodeFlags::External;
      ModelContext::forEachPointerArg(site, [&](Cell cell) {
        ctx.defer(DeferredEffect::Kind::Reachable, cell, reach,
                  returnsPointer ? why : nullptr);
      });
      ctx.setResultUnknown(site, NodeFlags::External | NodeFlags::Incomplete,
                           why);
      return;
    }
    case Class::ArgMemOnly: {
      // The callee may store a pointer to one argument's object, or any
      // pointer it read, into another argument's memory. Without a nocapture
      // attribute in CIR that is the unknown-call rule restricted to what the
      // arguments reach: only the globals are spared.
      NodeFlags access = NodeFlags::External | NodeFlags::Mod;
      if (hasRef(effects.getArgMem()))
        access |= NodeFlags::Ref;
      const Reason *why = ctx.makeReason(site, Reason::Kind::EscapedTo,
                                         "written by an argmem-only callee");
      ModelContext::forEachPointerArg(site, [&](Cell cell) {
        ctx.defer(DeferredEffect::Kind::Reachable, cell, access, why);
      });
      ctx.setResultUnknown(site, NodeFlags::External | NodeFlags::Incomplete,
                           why);
      return;
    }
    }
  }

  StringRef name() const override { return "memory-effects"; }
  /// Writes only through defer (argMemOnly) or not at all.
  bool declaresAllWrites() const override { return true; }
  StringRef specCitation() const override {
    return "#cir.memory_effects (CIRAttrs.td), the semantics of LLVM's "
           "memory(...) function attribute";
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// registerGenericModels
//===----------------------------------------------------------------------===//

void cir::pointsto::registerGenericModels(PointsToModelRegistry &registry) {
  static const OperatorNewDeleteNames newDelete;
  auto sym = [&](ArrayRef<StringRef> symbols,
                 std::unique_ptr<PointsToModel> model) {
    registry.addSymbolModel(symbols, std::move(model));
  };

  // Allocation.
  sym(toRefs(newDelete.newNames),
      std::make_unique<AllocatorModel>(
          "operator-new",
          "C++20 [basic.stc.dynamic.allocation]p2, [new.delete.single], "
          "[new.delete.array]",
          Accept::Declarations, AllocKind::SizeArg0));
  sym({"malloc"}, std::make_unique<AllocatorModel>(
                      "malloc", "C17 7.22.3p1, 7.22.3.4",
                      Accept::LibraryDefinitions, AllocKind::SizeArg0));
  sym({"calloc"}, std::make_unique<AllocatorModel>(
                      "calloc", "C17 7.22.3p1, 7.22.3.2",
                      Accept::LibraryDefinitions, AllocKind::Calloc));
  sym({"aligned_alloc"}, std::make_unique<AllocatorModel>(
                             "aligned_alloc", "C17 7.22.3p1, 7.22.3.1",
                             Accept::LibraryDefinitions, AllocKind::SizeArg1));
  sym({"posix_memalign"},
      std::make_unique<AllocatorModel>(
          "posix_memalign", "POSIX.1-2017 posix_memalign; C17 7.22.3p1",
          Accept::LibraryDefinitions, AllocKind::PosixMemalign));
  sym({"realloc"}, std::make_unique<AllocatorModel>(
                       "realloc", "C17 7.22.3p1, 7.22.3.5p2",
                       Accept::LibraryDefinitions, AllocKind::Realloc));
  sym({"strdup", "strndup"},
      std::make_unique<AllocatorModel>(
          "strdup", "C2x 7.24.6.4-5 (strdup, strndup); POSIX.1-2017",
          Accept::LibraryDefinitions, AllocKind::StrDup));
  // Itanium C++ ABI 2.4.2: __cxa_allocate_exception returns fresh storage for
  // the exception object.
  sym({"__cxa_allocate_exception"},
      std::make_unique<AllocatorModel>(
          "cxa-allocate-exception",
          "Itanium C++ ABI 2.4.2; C++20 [except.throw]p4", Accept::Declarations,
          AllocKind::SizeArg0));

  // Deallocation. C17 7.22.3.3, C++20 [new.delete.single]p10: the pointer
  // becomes invalid; nothing is read through it that the caller can see, and
  // no pointer is created. Mod on the freed node.
  sym(toRefs(newDelete.deleteNames),
      std::make_unique<NoPointerEffectModel>(
          "operator-delete", "C++20 [new.delete.single], [new.delete.array]",
          Accept::Declarations, ArgAccess::Mod));
  sym({"free"},
      std::make_unique<NoPointerEffectModel>(
          "free", "C17 7.22.3.3", Accept::LibraryDefinitions, ArgAccess::Mod));
  sym({"__cxa_free_exception"},
      std::make_unique<NoPointerEffectModel>(
          "cxa-free-exception", "Itanium C++ ABI 2.4.2", Accept::Declarations,
          ArgAccess::Mod));

  // C string and memory functions (C17 7.24).
  sym({"memcpy", "memmove"},
      std::make_unique<StringModel>("memcpy", "C17 7.24.2.1-2, 6.2.6.1p4",
                                    StringKind::Copy));
  sym({"memset"},
      std::make_unique<StringModel>("memset", "C17 7.24.6.1", StringKind::Set));
  sym({"strcpy"}, std::make_unique<StringModel>("strcpy", "C17 7.24.2.3",
                                                StringKind::StrCpy));
  sym({"strncpy"}, std::make_unique<StringModel>("strncpy", "C17 7.24.2.4",
                                                 StringKind::StrNCpy));
  sym({"strcat", "strncat"},
      std::make_unique<StringModel>("strcat", "C17 7.24.3.1-2",
                                    StringKind::StrCat));
  sym({"strlen", "strcmp", "strncmp", "memcmp"},
      std::make_unique<StringModel>("string-read",
                                    "C17 7.24.4.1-2, 7.24.4.4, 7.24.6.3",
                                    StringKind::Read));
  sym({"strchr", "strrchr", "strstr", "memchr"},
      std::make_unique<StringModel>(
          "string-search", "C17 7.24.5.1, 7.24.5.2, 7.24.5.5, 7.24.5.7",
          StringKind::Search));

  // The Itanium C++ ABI runtime that implements C++ semantics, minus the
  // functions that call user code on objects (__cxa_vec_*) or keep pointers
  // (below): it touches only its arguments and its own state.
  sym({"__cxa_guard_acquire", "__cxa_guard_release", "__cxa_guard_abort"},
      std::make_unique<NoPointerEffectModel>(
          "cxa-guard", "Itanium C++ ABI 3.3.2; C++20 [stmt.dcl]p3",
          Accept::Declarations, ArgAccess::ModRef));
  sym({"__cxa_begin_catch", "__cxa_end_catch", "__cxa_rethrow",
       "__cxa_get_exception_ptr", "__cxa_current_exception_type",
       "__cxa_call_unexpected", "__cxa_pure_virtual", "__cxa_deleted_virtual",
       "__cxa_bad_cast", "__cxa_bad_typeid", "__cxa_throw_bad_array_new_length",
       "__cxa_uncaught_exception", "__cxa_uncaught_exceptions",
       "__cxa_get_globals", "__cxa_get_globals_fast", "__cxa_finalize",
       "__cxa_demangle"},
      std::make_unique<NoPointerEffectModel>(
          "cxa-runtime", "Itanium C++ ABI 2.5, 3.3.5; C++20 [except.handle]",
          Accept::Declarations, ArgAccess::ModRef));
  // __cxa_atexit(f, obj, dso) and the thread_local variants keep f and obj
  // to call f(obj) at exit (Itanium C++ ABI 3.3.6; C++20
  // [basic.start.term]): they escape.
  sym({"__cxa_atexit", "__cxa_thread_atexit", "__cxa_thread_atexit_impl",
       "atexit", "at_quick_exit"},
      std::make_unique<EscapeArgsModel>(
          "cxa-atexit",
          "Itanium C++ ABI 3.3.6; C++20 [basic.start.term]; C17 7.22.4.2",
          Accept::Declarations, /*minArgs=*/1));
  // __cxa_throw(obj, tinfo, dtor) hands the exception object and its
  // destructor to whichever handler catches it: they escape. (The handout
  // lists __cxa_* as pointer-free; this one is not.)
  sym({"__cxa_throw"},
      std::make_unique<EscapeArgsModel>(
          "cxa-throw", "Itanium C++ ABI 2.4.3; C++20 [except.throw]p3",
          Accept::Declarations, /*minArgs=*/1));

  registry.addPredicateModel(std::make_unique<StdThrowModel>());

  // Termination. They do not return; exit runs the atexit functions and the
  // static destructors (C17 7.22.4.4, C++20 [support.start.term]), which
  // were registered (escaped) or are part of the globals graph.
  sym({"abort", "exit", "_Exit", "quick_exit", "_ZSt9terminatev"},
      std::make_unique<NoPointerEffectModel>(
          "terminate",
          "C17 7.22.4.1, 7.22.4.4-5, 7.22.4.7; C++20 [except.terminate]",
          Accept::LibraryDefinitions, ArgAccess::None));

  // Formatted and character output (C17 7.21.6, 7.21.7). The arguments are
  // read (%s) and may be written (%n, the sprintf buffer, the FILE); the
  // va_list of the v* functions reaches the variadic arguments. Nothing is
  // kept, so nothing escapes. SOUNDNESS: Mod and Ref on everything reachable
  // from a pointer argument covers %n through a pointer in a va_list.
  sym({"printf", "fprintf", "sprintf", "snprintf", "vprintf", "vfprintf",
       "vsprintf", "vsnprintf", "puts", "fputs", "fputc", "putc", "putchar",
       "fwrite", "fflush"},
      std::make_unique<NoPointerEffectModel>(
          "printf", "C17 7.21.6.1-13, 7.21.7.3-9, 7.21.8.2, 7.21.5.2",
          Accept::LibraryDefinitions, ArgAccess::ModRefReachable));

  registry.addPredicateModel(std::make_unique<MemoryEffectsModel>());
}
