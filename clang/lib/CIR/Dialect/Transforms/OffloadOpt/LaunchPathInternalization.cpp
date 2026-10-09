//===- LaunchPathInternalization.cpp - Clone host paths from roots --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// See LaunchPathInternalization.h for what this does and why it is sound.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Transforms/OffloadOpt/LaunchPathInternalization.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Interfaces/CIROpInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

#include <deque>

using namespace mlir;
using namespace cir;

bool cir::isClosedRoot(cir::FuncOp func) {
  if (!func || func.isDeclaration())
    return false;
  // C++20 [basic.start.main]p3: main shall not be used within a program.
  if (func.getSymName() == "main")
    return true;
  // Run by the implementation at program start or exit, with no arguments
  // ([basic.start.dynamic], [basic.start.term]).
  return func->hasAttr("global_ctor_priority") ||
         func->hasAttr("global_dtor_priority");
}

namespace {

/// Blocked: an address-of nobody vouched for (not followed; only to tell why
/// a sink is not reached).
enum class EdgeKind : uint8_t { Call, Address, Exact, Guard, Thread, Blocked };

struct Edge {
  EdgeKind kind;
  Operation *op;
  unsigned from, to;
  /// Thread: the std::thread::_State_impl constructors from the one P calls
  /// to the one that stores the vtable (the complete constructor calls the
  /// base one), as function indices, and the vtable.
  SmallVector<unsigned, 2> ctors;
  cir::GlobalOp vtable;
};

constexpr StringLiteral kStartThread = "_ZNSt6thread15_M_start_thread";
constexpr StringLiteral kStateImpl = "_ZNSt6thread11_State_implI";
constexpr StringLiteral kStateImplVTable = "_ZTVNSt6thread11_State_implI";

/// The call graph of the module the paths are cut from: the functions with a
/// body, the edges of the kinds of the file comment, and what decides
/// clean-ness.
class PathGraph {
public:
  PathGraph(ModuleOp module, const InternalizationOptions &options)
      : module(module), context(module.getContext()), options(options),
        symbols(module) {
    for (auto func : module.getOps<cir::FuncOp>())
      if (!func.isDeclaration()) {
        index[func] = funcs.size();
        byName[func.getSymNameAttr()] = funcs.size();
        funcs.push_back(func);
      }
    out.resize(funcs.size());
    in.resize(funcs.size());
    numOps.resize(funcs.size(), 0);
    root.resize(funcs.size(), false);
    for (auto [i, func] : llvm::enumerate(funcs)) {
      root[i] = isClosedRoot(func);
      func.walk([&](Operation *) { ++numOps[i]; });
    }
    collectVTableSlots();
    collectUses();
    for (unsigned i = 0, e = funcs.size(); i != e; ++i)
      collectEdges(i);
    computeClean();
  }

  ModuleOp module;
  MLIRContext *context;
  const InternalizationOptions &options;
  SymbolTable symbols;
  SmallVector<cir::FuncOp> funcs;
  DenseMap<Operation *, unsigned> index;
  DenseMap<StringAttr, unsigned> byName;
  SmallVector<Edge> edges;
  SmallVector<SmallVector<unsigned, 4>> out, in;
  SmallVector<uint64_t> numOps;
  SmallVector<bool> root, clean;
  unsigned numUnvouched = 0;

  std::optional<unsigned> lookup(StringRef name) const {
    auto it = byName.find(StringAttr::get(context, name));
    if (it == byName.end())
      return std::nullopt;
    return it->second;
  }

private:
  /// For each slot index i: the functions some vtable address point of the
  /// module has at slot i (the candidates of `get_virtual_fn_addr[i]`).
  DenseMap<int64_t, SmallVector<unsigned, 2>> slotTargets;
  /// The functions whose address is taken (a non-call use).
  SmallVector<unsigned> addressTaken;
  /// Per function: the functions that use its symbol (through local globals
  /// too), and whether some use is not in a function (or in a global
  /// another TU can name).
  SmallVector<llvm::SmallSetVector<unsigned, 4>> users;
  SmallVector<bool> unknownUse;

  void addEdge(Edge edge) {
    out[edge.from].push_back(edges.size());
    in[edge.to].push_back(edges.size());
    edges.push_back(edge);
  }

  cir::GlobalOp lookupGlobal(ModuleOp, StringRef name) {
    return symbols.lookup<cir::GlobalOp>(name);
  }

  void collectVTableSlots() {
    llvm::DenseSet<std::tuple<Operation *, int32_t, int32_t>> seen;
    module.walk([&](cir::VTableAddrPointOp ap) {
      cir::GlobalOp vt = lookupGlobal(module, ap.getName());
      if (!vt)
        return;
      cir::AddressPointAttr point = ap.getAddressPoint();
      if (!seen.insert({vt, point.getIndex(), point.getOffset()}).second)
        return;
      std::optional<Attribute> init = vt.getInitialValue();
      auto table = init ? dyn_cast<cir::VTableAttr>(*init) : nullptr;
      if (!table || point.getIndex() < 0 ||
          (size_t)point.getIndex() >= table.getData().size())
        return;
      auto arr =
          dyn_cast<cir::ConstArrayAttr>(table.getData()[point.getIndex()]);
      auto elts = arr ? dyn_cast<ArrayAttr>(arr.getElts()) : nullptr;
      if (!elts)
        return;
      for (int64_t slot = 0; point.getOffset() + slot < (int64_t)elts.size();
           ++slot) {
        auto view =
            dyn_cast<cir::GlobalViewAttr>(elts[point.getOffset() + slot]);
        if (!view)
          continue;
        if (std::optional<unsigned> f =
                lookup(view.getSymbol().getRootReference()))
          if (!llvm::is_contained(slotTargets[slot], *f))
            slotTargets[slot].push_back(*f);
      }
    });
  }

  void collectUses() {
    users.resize(funcs.size());
    unknownUse.resize(funcs.size(), false);
    std::optional<SymbolTable::UseRange> uses =
        SymbolTable::getSymbolUses(&module.getBodyRegion());
    if (!uses) {
      unknownUse.assign(funcs.size(), true);
      return;
    }
    DenseMap<StringAttr, SmallVector<Operation *, 2>> usersOf;
    for (const SymbolTable::SymbolUse &use : *uses)
      usersOf[use.getSymbolRef().getRootReference()].push_back(use.getUser());
    // The function users of a symbol, through local globals.
    DenseMap<StringAttr, std::pair<llvm::SmallSetVector<unsigned, 4>, bool>>
        memo;
    std::function<const std::pair<llvm::SmallSetVector<unsigned, 4>, bool> &(
        StringAttr, unsigned)>
        usersOfSymbol = [&](StringAttr name, unsigned depth)
        -> const std::pair<llvm::SmallSetVector<unsigned, 4>, bool> & {
      auto [it, inserted] = memo.try_emplace(name);
      if (!inserted)
        return it->second; // done, or in progress (a cycle of globals)
      llvm::SmallSetVector<unsigned, 4> result;
      bool unknown = depth > 8;
      for (Operation *user : usersOf.lookup(name)) {
        if (auto global = dyn_cast<cir::GlobalOp>(user)) {
          if (!cir::isLocalLinkage(global.getLinkage())) {
            unknown = true;
            continue;
          }
          const auto &[more, moreUnknown] =
              usersOfSymbol(global.getSymNameAttr(), depth + 1);
          result.insert(more.begin(), more.end());
          unknown |= moreUnknown;
          continue;
        }
        auto func = user->getParentOfType<cir::FuncOp>();
        auto it2 = func ? index.find(func) : index.end();
        if (it2 == index.end()) {
          unknown = true;
          continue;
        }
        result.insert(it2->second);
      }
      auto &slot = memo[name];
      slot.first = std::move(result);
      slot.second = unknown;
      return slot;
    };
    for (auto [i, func] : llvm::enumerate(funcs)) {
      const auto &[set, unknown] = usersOfSymbol(func.getSymNameAttr(), 0);
      users[i] = set;
      unknownUse[i] = unknown;
    }
    // Address-taken functions: a use that is not a direct call of it.
    llvm::DenseSet<unsigned> taken;
    for (const SymbolTable::SymbolUse &use : *uses) {
      std::optional<unsigned> f =
          lookup(use.getSymbolRef().getRootReference().getValue());
      if (!f)
        continue;
      if (auto call = dyn_cast<CIRCallOpInterface>(use.getUser()))
        if (auto callee = call->getAttrOfType<FlatSymbolRefAttr>("callee"))
          if (callee.getAttr() == funcs[*f].getSymNameAttr())
            continue;
      if (taken.insert(*f).second)
        addressTaken.push_back(*f);
    }
    llvm::sort(addressTaken);
  }

  /// Whether \p call may call \p target directly with its operands (the same
  /// arity, each operand of the parameter's type or both pointers, the same
  /// result).
  bool isCompatible(cir::CallOp call, cir::FuncOp target) const {
    cir::FuncType type = target.getFunctionType();
    if (type.isVarArg() || type.getNumInputs() != call.getNumArgOperands())
      return false;
    for (auto [arg, param] : llvm::zip(call.getArgOperands(), type.getInputs()))
      if (arg.getType() != param && !(isa<cir::PointerType>(arg.getType()) &&
                                      isa<cir::PointerType>(param)))
        return false;
    if (call->getNumResults() == 0)
      return type.hasVoidReturn();
    return !type.hasVoidReturn() &&
           type.getReturnType() == call->getResult(0).getType();
  }

  void collectEdges(unsigned from) {
    cir::FuncOp func = funcs[from];
    SmallVector<Operation *, 2> ctorCalls;
    bool startsThread = false;
    func.walk([&](Operation *op) {
      if (auto call = dyn_cast<cir::CallOp>(op)) {
        if (!call.isIndirect()) {
          StringRef callee = *call.getCallee();
          if (callee.starts_with(kStartThread))
            startsThread = true;
          std::optional<unsigned> to = lookup(callee);
          if (!to)
            return;
          addEdge({EdgeKind::Call, op, from, *to});
          if (isStateImplCtor(callee))
            ctorCalls.push_back(op);
          return;
        }
        if (!options.guardIndirectCalls ||
            call->hasAttr(CIRDialect::getMustTailAttrName()))
          return;
        // A call of a function's address (cir.get_global, through casts;
        // after mem2reg and SCCP, a local function pointer that is only
        // ever one function): a direct call of it.
        Value callee = call.getIndirectCall();
        while (auto cast = callee.getDefiningOp<cir::CastOp>())
          callee = cast.getSrc();
        if (auto addr = callee.getDefiningOp<cir::GetGlobalOp>()) {
          if (std::optional<unsigned> to = lookup(addr.getName()))
            if (isCompatible(call, funcs[*to]))
              addEdge({EdgeKind::Exact, op, from, *to});
          return;
        }
        // A virtual call: the callee is a load of a vtable slot; its
        // candidates are that slot of every vtable (an override takes its own
        // `this`: pointer-castable operands). Any other call through a
        // pointer: the address-taken functions of exactly the pointee type
        // (calling a function through a pointer to another function type is
        // undefined, C++20 [expr.call]p6), so that no guard makes a direct
        // call the program never makes, which the analysis would bind.
        ArrayRef<unsigned> candidates = addressTaken;
        bool isVirtual = false;
        if (auto load = call.getIndirectCall().getDefiningOp<cir::LoadOp>())
          if (auto slot = load.getAddr()
                              .getDefiningOp<cir::VTableGetVirtualFnAddrOp>()) {
            auto it = slotTargets.find(slot.getIndex());
            candidates = it == slotTargets.end() ? ArrayRef<unsigned>()
                                                 : ArrayRef(it->second);
            isVirtual = true;
          }
        auto calleeType =
            dyn_cast<cir::PointerType>(call.getIndirectCall().getType());
        for (unsigned to : candidates)
          if (isVirtual ? isCompatible(call, funcs[to])
                        : calleeType && calleeType.getPointee() ==
                                            funcs[to].getFunctionType())
            addEdge({EdgeKind::Guard, op, from, to});
        return;
      }
      if (auto addr = dyn_cast<cir::GetGlobalOp>(op)) {
        std::optional<unsigned> to = lookup(addr.getName());
        if (!to)
          return;
        if (options.vouchAddress && options.vouchAddress(addr, func)) {
          addEdge({EdgeKind::Address, op, from, *to});
        } else {
          addEdge({EdgeKind::Blocked, op, from, *to});
          ++numUnvouched;
        }
      }
    });
    if (!options.redirectThreadStates || !startsThread)
      return;
    // std::thread: the state constructor's vtable holds the _M_run the
    // thread runs. The vtable is stored by the base-object constructor,
    // which the complete-object one calls.
    for (Operation *op : ctorCalls) {
      SmallVector<unsigned, 2> ctors{
          *lookup(*cast<cir::CallOp>(op).getCallee())};
      cir::GlobalOp vtable;
      while (!vtable && ctors.size() <= 3) {
        std::optional<unsigned> next;
        funcs[ctors.back()].walk([&](Operation *inner) {
          if (auto ap = dyn_cast<cir::VTableAddrPointOp>(inner)) {
            if (ap.getName().starts_with(kStateImplVTable))
              vtable = lookupGlobal(module, ap.getName());
          } else if (auto call = dyn_cast<cir::CallOp>(inner)) {
            if (!call.isIndirect() && isStateImplCtor(*call.getCallee()))
              next = lookup(*call.getCallee());
          }
        });
        if (vtable || !next || llvm::is_contained(ctors, *next))
          break;
        ctors.push_back(*next);
      }
      std::optional<Attribute> init =
          vtable ? vtable.getInitialValue() : std::nullopt;
      if (!init)
        continue;
      std::optional<unsigned> run;
      init->walk([&](cir::GlobalViewAttr view) {
        StringRef name = view.getSymbol().getRootReference();
        if (name.ends_with("6_M_runEv"))
          run = lookup(name);
      });
      if (run)
        addEdge({EdgeKind::Thread, op, from, *run, std::move(ctors), vtable});
    }
  }

  /// A constructor of std::thread::_State_impl<F> (complete or base object).
  static bool isStateImplCtor(StringRef name) {
    return name.starts_with(kStateImpl) &&
           (name.contains("C1E") || name.contains("C2E"));
  }

  void computeClean() {
    // Greatest fixpoint: start with every root and every local function
    // whose uses are all in functions, remove one with a user that is not
    // clean, until nothing changes.
    clean.assign(funcs.size(), false);
    for (auto [i, func] : llvm::enumerate(funcs))
      clean[i] = root[i] || (cir::isLocalLinkage(func.getLinkage()) &&
                             !unknownUse[i] && !users[i].empty());
    bool changed = true;
    while (changed) {
      changed = false;
      for (unsigned i = 0, e = funcs.size(); i != e; ++i) {
        if (!clean[i] || root[i])
          continue;
        if (llvm::any_of(users[i], [&](unsigned u) { return !clean[u]; })) {
          clean[i] = false;
          changed = true;
        }
      }
    }
  }
};

/// The functions of the paths to a set of sinks: the chain, its copies and
/// its cost.
struct Plan {
  llvm::SmallVector<bool> chain, cloned, isSink, inR;
  /// The constructors the vtable redirect copies that are not copies of the
  /// chain.
  llvm::SmallSetVector<unsigned, 2> ctorCopies;
  unsigned numFunctions = 0;
  uint64_t numOps = 0;

  bool inWorld(const PathGraph &g, unsigned f) const {
    return chain[f] && (g.root[f] || g.clean[f] || cloned[f]);
  }
};

Plan makePlan(const PathGraph &g, ArrayRef<unsigned> sinks,
              const InternalizationOptions &options) {
  Plan plan;
  size_t n = g.funcs.size();
  plan.chain.assign(n, false);
  plan.cloned.assign(n, false);
  plan.isSink.assign(n, false);
  plan.inR.assign(n, false);
  std::deque<unsigned> work;
  for (unsigned s : sinks) {
    plan.isSink[s] = true;
    if (!plan.inR[s]) {
      plan.inR[s] = true;
      work.push_back(s);
    }
  }
  while (!work.empty()) {
    unsigned f = work.front();
    work.pop_front();
    for (unsigned e : g.in[f]) {
      if (g.edges[e].kind == EdgeKind::Blocked)
        continue;
      unsigned from = g.edges[e].from;
      if (!plan.inR[from]) {
        plan.inR[from] = true;
        work.push_back(from);
      }
    }
  }
  for (unsigned r = 0; r != n; ++r)
    if (g.root[r] && plan.inR[r]) {
      plan.chain[r] = true;
      work.push_back(r);
    }
  while (!work.empty()) {
    unsigned f = work.front();
    work.pop_front();
    // A sink ends its paths unless sinks are copied too.
    if (plan.isSink[f] && !options.cloneSinks && !g.root[f])
      continue;
    for (unsigned e : g.out[f]) {
      if (g.edges[e].kind == EdgeKind::Blocked)
        continue;
      unsigned to = g.edges[e].to;
      if (plan.inR[to] && !plan.chain[to]) {
        plan.chain[to] = true;
        work.push_back(to);
      }
    }
  }
  for (unsigned f = 0; f != n; ++f)
    if (plan.chain[f] && !g.root[f] && !g.clean[f] &&
        (options.cloneSinks || !plan.isSink[f])) {
      plan.cloned[f] = true;
      ++plan.numFunctions;
      plan.numOps += g.numOps[f];
    }
  for (const Edge &edge : g.edges)
    if (edge.kind == EdgeKind::Thread && plan.inWorld(g, edge.from) &&
        plan.cloned[edge.to])
      for (unsigned ctor : edge.ctors)
        if (!plan.cloned[ctor] && plan.ctorCopies.insert(ctor)) {
          ++plan.numFunctions;
          plan.numOps += g.numOps[ctor];
        }
  return plan;
}

bool withinBudget(const Plan &plan, const InternalizationOptions &options) {
  return plan.numFunctions <= options.maxFunctions &&
         plan.numOps <= options.maxOps;
}

/// Make \p symbol (a copy) local to this TU.
void makeLocal(Operation *symbol) {
  if (auto func = dyn_cast<cir::FuncOp>(symbol)) {
    func.setLinkage(cir::GlobalLinkageKind::InternalLinkage);
    func.removeComdatAttr();
  } else if (auto global = dyn_cast<cir::GlobalOp>(symbol)) {
    global.setLinkage(cir::GlobalLinkageKind::InternalLinkage);
    global.setComdat(false);
  }
  SymbolTable::setSymbolVisibility(symbol, SymbolTable::Visibility::Private);
}

/// A direct call of \p copy with the operands, attributes and result of the
/// indirect call \p call, at \p b's insertion point (operands bitcast to the
/// copy's parameter types where they differ: pointers only, isCompatible).
cir::CallOp makeDirectCall(OpBuilder &b, cir::CallOp call, cir::FuncOp copy) {
  auto copyCall = cast<cir::CallOp>(b.clone(*call));
  copyCall->eraseOperand(0);
  copyCall.setCalleeAttr(FlatSymbolRefAttr::get(copy.getSymNameAttr()));
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(copyCall);
  for (auto [i, param] : llvm::enumerate(copy.getFunctionType().getInputs())) {
    Value arg = copyCall->getOperand(i);
    if (arg.getType() != param)
      copyCall->setOperand(i, cir::CastOp::create(b, call.getLoc(), param,
                                                  cir::CastKind::bitcast, arg));
  }
  return copyCall;
}

/// Replace the indirect call \p call, whose callee is the address of
/// \p original, by a direct call of its copy \p copy.
void devirtualizeCall(cir::CallOp call, cir::FuncOp original,
                      cir::FuncOp copy) {
  OpBuilder b(call);
  cir::CallOp direct = makeDirectCall(b, call, copy);
  direct->setAttr(kInternalizedEdgeAttr, original.getSymNameAttr());
  call->replaceAllUsesWith(direct);
  call->erase();
}

/// Guard the indirect call \p call with a direct call of \p copy when its
/// callee is \p original (see the file comment).
void guardCall(cir::CallOp call, cir::FuncOp original, cir::FuncOp copy) {
  cir::CIRBaseBuilderTy builder(*call->getContext());
  builder.setInsertionPoint(call);
  Location loc = call.getLoc();
  Value callee = call.getIndirectCall();
  Value addr = cir::GetGlobalOp::create(
      builder, loc, cir::PointerType::get(original.getFunctionType()),
      original.getSymName());
  if (addr.getType() != callee.getType())
    addr = builder.createBitcast(loc, addr, callee.getType());
  Value same = builder.createCompare(loc, cir::CmpOpKind::eq, callee, addr);
  auto direct = [&](OpBuilder &b, Location) -> Value {
    cir::CallOp copyCall = makeDirectCall(b, call, copy);
    return copyCall->getNumResults() ? copyCall->getResult(0) : Value();
  };
  StringAttr tag = original.getSymNameAttr();
  if (call->getNumResults() == 0) {
    auto guard = cir::IfOp::create(
        builder, loc, same, /*withElseRegion=*/true,
        [&](OpBuilder &b, Location l) {
          direct(b, l);
          cir::YieldOp::create(b, l);
        },
        [&](OpBuilder &b, Location l) { cir::YieldOp::create(b, l); });
    guard->setAttr(kInternalizedGuardAttr, tag);
    call->moveBefore(guard.getElseRegion().front().getTerminator());
    return;
  }
  Value result = call->getResult(0);
  auto guard = cir::TernaryOp::create(
      builder, loc, same,
      [&](OpBuilder &b, Location l) {
        cir::YieldOp::create(b, l, ValueRange{direct(b, l)});
      },
      [&](OpBuilder &b, Location l) {
        cir::YieldOp::create(b, l, ValueRange{result});
      });
  guard->setAttr(kInternalizedGuardAttr, tag);
  result.replaceUsesWithIf(guard.getResult(), [&](OpOperand &use) {
    return !guard->isAncestor(use.getOwner());
  });
  call->moveBefore(guard.getFalseRegion().front().getTerminator());
}

} // namespace

InternalizationResult
cir::internalizeLaunchPaths(ModuleOp module, ArrayRef<cir::FuncOp> sinkFuncs,
                            const InternalizationOptions &options) {
  InternalizationResult result;
  PathGraph g(module, options);
  SmallVector<unsigned> sinks;
  for (cir::FuncOp sink : sinkFuncs) {
    auto it = g.index.find(sink);
    if (it != g.index.end() && !llvm::is_contained(sinks, it->second))
      sinks.push_back(it->second);
  }

  // The budget: the whole plan if it fits, else the sinks one at a time.
  Plan plan = makePlan(g, sinks, options);
  SmallVector<unsigned> accepted(sinks);
  if (!withinBudget(plan, options)) {
    accepted.clear();
    plan = makePlan(g, accepted, options);
    for (unsigned s : sinks) {
      SmallVector<unsigned> trial(accepted);
      trial.push_back(s);
      Plan next = makePlan(g, trial, options);
      if (withinBudget(next, options)) {
        accepted = std::move(trial);
        plan = std::move(next);
        continue;
      }
      result.skipped.push_back(g.funcs[s]);
      if (result.budgetWhy.empty()) {
        llvm::raw_string_ostream os(result.budgetWhy);
        os << "with @" << g.funcs[s].getSymName() << " the paths need "
           << next.numFunctions << " function copies and " << next.numOps
           << " ops (budget " << options.maxFunctions << " and "
           << options.maxOps << ")";
      }
    }
  }
  result.numUnvouchedAddresses = g.numUnvouched;

  // The copies, after their originals, in module order.
  SymbolTable symbols(module);
  DenseMap<unsigned, std::pair<cir::FuncOp, IRMapping>> copies;
  auto copyFunction = [&](unsigned f) -> cir::FuncOp {
    auto [it, inserted] = copies.try_emplace(f);
    if (!inserted)
      return it->second.first;
    cir::FuncOp original = g.funcs[f];
    auto copy = cast<cir::FuncOp>(original->clone(it->second.second));
    copy.setSymName((original.getSymName() + options.suffix).str());
    makeLocal(copy);
    copy->setAttr(kInternalizedAttr, original.getSymNameAttr());
    symbols.insert(copy, std::next(Block::iterator(original)));
    it->second.first = copy;
    result.copies[original] = copy;
    ++result.numCloned;
    result.numOps += g.numOps[f];
    return copy;
  };
  for (unsigned f = 0, e = g.funcs.size(); f != e; ++f)
    if (plan.cloned[f])
      copyFunction(f);
  for (unsigned f = 0, e = g.funcs.size(); f != e; ++f)
    if (plan.chain[f] && !g.root[f] && g.clean[f])
      ++result.numInPlace;

  // The op an edge of the plan is at: in the copy of its function, if any.
  auto opOf = [&](const Edge &edge) -> Operation * {
    auto it = copies.find(edge.from);
    if (it == copies.end())
      return edge.op;
    return it->second.second.lookup(edge.op);
  };

  // Redirect the edges from the world (roots, clean functions of the
  // chain, copies) into copies. Guards last: they move the calls.
  DenseMap<Operation *, SmallVector<unsigned, 2>> guards;
  SmallVector<std::pair<Operation *, unsigned>> exact;
  DenseMap<Operation *, cir::GlobalOp> vtableCopies;
  for (Edge edge : g.edges) {
    if (!plan.inWorld(g, edge.from) || !plan.cloned[edge.to])
      continue;
    Operation *op = opOf(edge);
    StringAttr original = g.funcs[edge.to].getSymNameAttr();
    cir::FuncOp copy = copies[edge.to].first;
    switch (edge.kind) {
    case EdgeKind::Call:
      cast<cir::CallOp>(op).setCalleeAttr(
          FlatSymbolRefAttr::get(copy.getSymNameAttr()));
      op->setAttr(kInternalizedEdgeAttr, original);
      ++result.numRedirected;
      break;
    case EdgeKind::Address:
      cast<cir::GetGlobalOp>(op).setNameAttr(
          FlatSymbolRefAttr::get(copy.getSymNameAttr()));
      op->setAttr(kInternalizedEdgeAttr, original);
      ++result.numRedirected;
      break;
    case EdgeKind::Exact:
      exact.push_back({op, edge.to});
      break;
    case EdgeKind::Guard:
      guards[op].push_back(edge.to);
      break;
    case EdgeKind::Blocked:
      // An address nobody vouched for keeps naming the original.
      break;
    case EdgeKind::Thread: {
      // V' (once per vtable) with _M_run -> its copy, and the constructor
      // copy that stores it.
      cir::GlobalOp &vt = vtableCopies[edge.vtable];
      if (!vt) {
        vt = cast<cir::GlobalOp>(edge.vtable->clone());
        vt.setSymName((edge.vtable.getSymName() + options.suffix).str());
        makeLocal(vt);
        vt->setAttr(kInternalizedAttr, edge.vtable.getSymNameAttr());
        symbols.insert(vt, std::next(Block::iterator(edge.vtable)));
        // The slot of _M_run in the initializer (symbol uses in a global's
        // own attributes are not "nested" uses for replaceAllSymbolUses).
        AttrTypeReplacer replacer;
        replacer.addReplacement(
            [&](FlatSymbolRefAttr ref) -> std::optional<Attribute> {
              if (ref.getAttr() == original)
                return FlatSymbolRefAttr::get(copy.getSymNameAttr());
              return std::nullopt;
            });
        vt.setInitialValueAttr(replacer.replace(*vt.getInitialValue()));
        ++result.numVTables;
      }
      // The constructor copies: each calls the next copy, the last stores
      // V'.
      SmallVector<cir::FuncOp, 2> ctorCopies;
      for (unsigned ctor : edge.ctors)
        ctorCopies.push_back(copyFunction(ctor));
      for (unsigned i = 0; i + 1 < edge.ctors.size(); ++i) {
        StringAttr next = g.funcs[edge.ctors[i + 1]].getSymNameAttr();
        ctorCopies[i].walk([&](cir::CallOp call) {
          if (!call.isIndirect() && call.getCalleeAttr().getAttr() == next) {
            call.setCalleeAttr(
                FlatSymbolRefAttr::get(ctorCopies[i + 1].getSymNameAttr()));
            call->setAttr(kInternalizedEdgeAttr, next);
          }
        });
      }
      (void)SymbolTable::replaceAllSymbolUses(edge.vtable.getSymNameAttr(),
                                              vt.getSymNameAttr(),
                                              ctorCopies.back().getOperation());
      cast<cir::CallOp>(op).setCalleeAttr(
          FlatSymbolRefAttr::get(ctorCopies.front().getSymNameAttr()));
      op->setAttr(kInternalizedEdgeAttr,
                  g.funcs[edge.ctors.front()].getSymNameAttr());
      ++result.numRedirected;
      break;
    }
    }
  }
  for (auto [op, to] : exact) {
    devirtualizeCall(cast<cir::CallOp>(op), g.funcs[to], copies[to].first);
    ++result.numRedirected;
  }
  for (auto &[op, targets] : guards)
    for (unsigned to : targets) {
      guardCall(cast<cir::CallOp>(op), g.funcs[to], copies[to].first);
      ++result.numGuards;
    }

  // The sinks reached, their calls in the world and their roots.
  for (unsigned s : accepted) {
    InternalizationResult::Sink sink;
    sink.sink = g.funcs[s];
    for (unsigned e : g.in[s]) {
      const Edge &edge = g.edges[e];
      if (edge.kind != EdgeKind::Call || !plan.inWorld(g, edge.from))
        continue;
      sink.calls.push_back(opOf(edge));
      sink.throughClone |= plan.cloned[edge.from];
    }
    if (sink.calls.empty())
      continue;
    // The roots it is reached from (forward over the plan's chain).
    for (unsigned r = 0, e = g.funcs.size(); r != e; ++r) {
      if (!g.root[r] || !plan.chain[r])
        continue;
      llvm::SmallVector<bool> seen(g.funcs.size(), false);
      std::deque<unsigned> work{r};
      seen[r] = true;
      bool reaches = false;
      while (!work.empty() && !reaches) {
        unsigned f = work.front();
        work.pop_front();
        for (unsigned ei : g.out[f]) {
          if (g.edges[ei].kind == EdgeKind::Blocked)
            continue;
          unsigned to = g.edges[ei].to;
          if (to == s && plan.inWorld(g, f)) {
            reaches = true;
            break;
          }
          if (!plan.chain[to] || seen[to] ||
              (plan.isSink[to] && !options.cloneSinks))
            continue;
          seen[to] = true;
          work.push_back(to);
        }
      }
      if (reaches)
        sink.roots.push_back(g.funcs[r].getSymName().str());
    }
    result.sinks.push_back(std::move(sink));
  }
  // Why the other sinks are not reached: no closed root reaches them at all,
  // or only through an edge that cannot be redirected.
  for (unsigned s : sinks) {
    if (llvm::any_of(result.sinks,
                     [&](const InternalizationResult::Sink &k) {
                       return k.sink == g.funcs[s];
                     }) ||
        llvm::is_contained(result.skipped, g.funcs[s]))
      continue;
    // Forward from the roots over every edge, blocked ones included.
    llvm::SmallVector<int> via(g.funcs.size(), -1); // the blocked edge used
    llvm::SmallVector<bool> seen(g.funcs.size(), false);
    std::deque<unsigned> work;
    for (unsigned r = 0, e = g.funcs.size(); r != e; ++r)
      if (g.root[r]) {
        seen[r] = true;
        work.push_back(r);
      }
    while (!work.empty()) {
      unsigned f = work.front();
      work.pop_front();
      for (unsigned ei : g.out[f]) {
        unsigned to = g.edges[ei].to;
        if (seen[to])
          continue;
        seen[to] = true;
        via[to] = g.edges[ei].kind == EdgeKind::Blocked ? (int)ei : via[f];
        work.push_back(to);
      }
    }
    std::string why;
    llvm::raw_string_ostream os(why);
    if (!seen[s]) {
      os << "no closed root (main, a dynamic initializer) reaches it";
    } else if (via[s] >= 0) {
      const Edge &blocked = g.edges[via[s]];
      os << "it is reached from a closed root only through the address of @"
         << g.funcs[blocked.to].getSymName() << " taken in @"
         << g.funcs[blocked.from].getSymName()
         << ", which code we don't see may call or compare";
    } else {
      os << "its paths from a closed root are not redirectable";
    }
    result.unreached.push_back({g.funcs[s], why});
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Prune
//===----------------------------------------------------------------------===//

namespace {

/// Replace the guard \p guard (a cir.if or cir.ternary made by guardCall) by
/// the contents of its false branch.
void unwrapGuard(Operation *guard) {
  Region &falseRegion = guard->getRegion(1);
  Block &block = falseRegion.front();
  Operation *yield = block.getTerminator();
  if (guard->getNumResults())
    guard->getResult(0).replaceAllUsesWith(yield->getOperand(0));
  yield->erase();
  // The compare and the address only feed the guard; leave them to DCE.
  guard->getBlock()->getOperations().splice(Block::iterator(guard),
                                            block.getOperations());
  guard->erase();
}

} // namespace

/// The top-level op of \p module that holds \p op (a function, a global).
static Operation *topLevelOp(ModuleOp module, Operation *op) {
  while (op && op->getParentOp() != module.getOperation())
    op = op->getParentOp();
  return op;
}

unsigned cir::pruneInternalizedPaths(ModuleOp module,
                                     function_ref<bool(cir::FuncOp)> keep) {
  SmallVector<Operation *> copies;
  for (Operation &op : module.getBody()->getOperations())
    if (op.hasAttr(kInternalizedAttr))
      copies.push_back(&op);
  if (copies.empty())
    return 0;
  DenseMap<StringAttr, Operation *> copyByName;
  for (Operation *copy : copies)
    copyByName[SymbolTable::getSymbolName(copy)] = copy;
  // The symbol uses of the module, by the top-level op that holds them (one
  // walk; a global's initializer is a use by the global itself).
  auto collectRefs = [&] {
    DenseMap<Operation *, SmallVector<Operation *, 4>> refs;
    if (std::optional<SymbolTable::UseRange> uses =
            SymbolTable::getSymbolUses(&module.getBodyRegion()))
      for (const SymbolTable::SymbolUse &use : *uses)
        if (Operation *target =
                copyByName.lookup(use.getSymbolRef().getRootReference()))
          refs[topLevelOp(module, use.getUser())].push_back(target);
    return refs;
  };
  DenseMap<Operation *, SmallVector<Operation *, 4>> refs = collectRefs();

  // A copy is needed if it is kept (a kept getter copy), calls a kept
  // function, or refers to a needed copy: backward closure.
  llvm::SmallSetVector<Operation *, 16> needed;
  for (Operation *copy : copies) {
    auto func = dyn_cast<cir::FuncOp>(copy);
    if (!func)
      continue;
    bool callsKept = keep(func);
    func.walk([&](cir::CallOp call) {
      if (callsKept || call.isIndirect())
        return;
      if (Operation *target = copyByName.lookup(call.getCalleeAttr().getAttr()))
        if (auto callee = dyn_cast<cir::FuncOp>(target))
          callsKept = keep(callee);
    });
    if (callsKept)
      needed.insert(copy);
  }
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *copy : copies) {
      if (needed.contains(copy))
        continue;
      if (llvm::any_of(refs.lookup(copy),
                       [&](Operation *t) { return needed.contains(t); })) {
        needed.insert(copy);
        changed = true;
      }
    }
  }
  // Restore the edges into copies that are not needed, in every function.
  auto isDropped = [&](StringAttr name) {
    Operation *copy = copyByName.lookup(name);
    return copy && !needed.contains(copy);
  };
  SmallVector<Operation *> unwrap;
  module.walk([&](Operation *op) {
    if (op->hasAttr(kInternalizedGuardAttr)) {
      // The true branch calls the copy of the candidate.
      bool drop = false;
      op->getRegion(0).walk([&](cir::CallOp call) {
        if (!call.isIndirect() && isDropped(call.getCalleeAttr().getAttr()))
          drop = true;
      });
      if (drop)
        unwrap.push_back(op);
      return;
    }
    auto original = op->getAttrOfType<StringAttr>(kInternalizedEdgeAttr);
    if (!original)
      return;
    if (auto call = dyn_cast<cir::CallOp>(op)) {
      if (!call.isIndirect() && isDropped(call.getCalleeAttr().getAttr()))
        call.setCalleeAttr(FlatSymbolRefAttr::get(original));
    } else if (auto addr = dyn_cast<cir::GetGlobalOp>(op)) {
      if (isDropped(addr.getNameAttr().getAttr()))
        addr.setNameAttr(FlatSymbolRefAttr::get(original));
    }
  });
  // Inner guards first (a later candidate's guard is nested in the false
  // branch of an earlier one).
  for (Operation *op : llvm::reverse(unwrap))
    unwrapGuard(op);

  // Erase the copies nothing uses any more: count the uses of each copy by
  // other top-level ops, and erase from the ones with none, releasing what
  // they refer to.
  refs = collectRefs();
  DenseMap<Operation *, unsigned> liveUses;
  for (auto &[holder, targets] : refs)
    for (Operation *target : targets)
      if (target != holder)
        ++liveUses[target];
  SmallVector<Operation *> work;
  for (Operation *copy : copies)
    if (!needed.contains(copy) && !liveUses.lookup(copy))
      work.push_back(copy);
  llvm::DenseSet<Operation *> erasedOps;
  while (!work.empty()) {
    Operation *copy = work.pop_back_val();
    if (!erasedOps.insert(copy).second)
      continue;
    for (Operation *target : refs.lookup(copy))
      if (target != copy && !erasedOps.contains(target) &&
          --liveUses[target] == 0 && !needed.contains(target))
        work.push_back(target);
    copy->erase();
  }
  return erasedOps.size();
}

void cir::stripInternalizationTags(ModuleOp module) {
  module.walk([](Operation *op) {
    op->removeAttr(kInternalizedAttr);
    op->removeAttr(kInternalizedEdgeAttr);
    op->removeAttr(kInternalizedGuardAttr);
  });
}
