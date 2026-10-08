//===- PointsToVerifier.cpp - Self-check of points-to graphs --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToVerifier.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/LocalBuilder.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToAnalysis.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Interfaces/CIROpInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace cir::pointsto;

static llvm::cl::opt<bool> clVerify(
    "cir-points-to-verify",
    llvm::cl::desc("Check the invariants of every CIR points-to graph built"),
    llvm::cl::init(false));

bool cir::pointsto::isPointsToVerifyEnabled() { return clVerify; }

namespace {
class Verifier {
public:
  explicit Verifier(const FunctionGraph &fg)
      : fg(fg), dataLayout(LocalBuilder::getDataLayoutFor(fg.getFunction())) {}

  LogicalResult run();

private:
  /// Check that \p value has a cell or slots; \p report emits the error.
  void checkValue(Value value, function_ref<InFlightDiagnostic()> report);

  const FunctionGraph &fg;
  mlir::DataLayout dataLayout;
  llvm::DenseMap<Type, SmallVector<int64_t, 2>> offsetCache;
  bool ok = true;
};
} // namespace

void Verifier::checkValue(Value value,
                          function_ref<InFlightDiagnostic()> report) {
  Type type = value.getType();
  if (isPointerValueType(type)) {
    if (!fg.lookup(value) && !isNullPointerConstant(value)) {
      report() << "points-to: pointer value has no cell";
      ok = false;
    }
    return;
  }
  auto [it, inserted] = offsetCache.try_emplace(type);
  if (inserted)
    it->second = getPointerSlotOffsets(type, dataLayout);
  const SmallVector<int64_t, 2> &offsets = it->second;
  if (offsets.empty())
    return;
  const AggregateSlots *slots = fg.lookupSlots(value);
  if (!slots) {
    report() << "points-to: aggregate value holding pointers has no slot "
                "record";
    ok = false;
    return;
  }
  bool match = slots->size() == offsets.size();
  for (size_t i = 0; match && i != offsets.size(); ++i)
    match = (*slots)[i].first == offsets[i];
  if (!match) {
    report() << "points-to: slot record does not match the pointer offsets "
                "of the value's type";
    ok = false;
  }
}

LogicalResult Verifier::run() {
  cir::FuncOp func = fg.getFunction();

  llvm::DenseMap<Operation *, unsigned> records;
  // Synthetic sites (cir.delete_array, or a call a model makes on behalf of a
  // modelled call) come on top of the call's own record.
  for (const CallSiteRecord &site : fg.callSites)
    if (!site.synthetic)
      ++records[site.call];

  func.walk([&](Block *block) {
    for (BlockArgument arg : block->getArguments())
      checkValue(arg, [&] {
        return emitError(arg.getLoc())
               << "block argument #" << arg.getArgNumber() << ": ";
      });
  });
  func.walk([&](Operation *op) {
    if (op == func.getOperation())
      return;
    for (OpResult result : op->getResults())
      checkValue(result, [&] {
        return op->emitError()
               << "result #" << result.getResultNumber() << ": ";
      });
    if (isa<cir::CIRCallOpInterface>(op)) {
      unsigned count = records.lookup(op);
      if (count != 1) {
        op->emitError() << "points-to: call has " << count
                        << " call-site records, expected 1";
        ok = false;
      }
    }
  });

  std::string errors;
  llvm::raw_string_ostream errorStream(errors);
  if (!fg.getGraph().verify(&errorStream)) {
    func.emitError() << "points-to graph failed to verify:\n" << errors;
    ok = false;
  }
  return success(ok);
}

LogicalResult cir::pointsto::verifyFunctionGraph(const FunctionGraph &fg) {
  return Verifier(fg).run();
}

LogicalResult
cir::pointsto::verifyPointsToAnalysis(const PointsToAnalysis &analysis) {
  ModuleOp module = analysis.getModule();
  const Graph &graph = analysis.getGraph();
  bool ok = true;
  std::string errors;
  llvm::raw_string_ostream os(errors);
  auto nodeName = [](const Node *node) {
    return "n" + std::to_string(node->getId());
  };

  // Every cell held resolves to a live node.
  size_t numDangling = 0;
  auto checkCell = [&](Cell cell) {
    if (cell && cell.getNode()->isForwarded())
      ++numDangling;
  };
  auto checkSlots = [&](const AggregateSlots &slots) {
    for (auto [off, cell] : slots)
      checkCell(cell);
  };
  for (auto [value, cell] : analysis.scalarMap)
    checkCell(cell);
  for (const auto &[value, slots] : analysis.aggregateMap)
    checkSlots(slots);
  for (const std::unique_ptr<TUFunction> &fn : analysis.getFunctions()) {
    for (const std::optional<Cell> &formal : fn->formals)
      if (formal)
        checkCell(*formal);
    for (const std::optional<AggregateSlots> &slots : fn->formalSlots)
      if (slots)
        checkSlots(*slots);
    if (fn->returnCell)
      checkCell(*fn->returnCell);
    if (fn->returnSlots)
      checkSlots(*fn->returnSlots);
    for (const TUCallSite &call : fn->callSites) {
      if (call.site.calleePtr)
        checkCell(*call.site.calleePtr);
      for (const std::optional<Cell> &arg : call.site.args)
        if (arg)
          checkCell(*arg);
      if (call.site.result)
        checkCell(*call.site.result);
    }
  }
  if (numDangling) {
    os << numDangling << " cells resolve to a forwarded node\n";
    ok = false;
  }

  for (const Node *node : graph.nodes()) {
    if (node->isForwarded()) {
      os << nodeName(node) << " is live and forwarded\n";
      ok = false;
    }
    // Status propagation is closed.
    for (NodeFlags flag : {NodeFlags::External, NodeFlags::Incomplete}) {
      if (!node->hasFlags(flag))
        continue;
      for (auto [off, target] : node->getLinks())
        if (!target.getNode()->hasFlags(flag)) {
          os << nodeName(node) << " is " << getFlagName(flag) << " but "
             << nodeName(target.getNode()) << ", which it links to at " << off
             << ", is not\n";
          ok = false;
        }
      for (const GhostLink &ghost : node->getGhostLinks())
        if (!ghost.target.getNode()->hasFlags(flag)) {
          os << nodeName(node) << " is " << getFlagName(flag) << " but "
             << nodeName(ghost.target.getNode()) << ", which its ghost link "
             << ghost.field.getName() << " at " << ghost.offset
             << " targets, is not\n";
          ok = false;
        }
      // Subobjects: to the children, and up from a child at offset 0.
      for (const SubobjectLink &sub : node->getSubobjects())
        if (!sub.child.getNode()->hasFlags(flag)) {
          os << nodeName(node) << " is " << getFlagName(flag) << " but "
             << nodeName(sub.child.getNode()) << ", its child at " << sub.offset
             << ", is not\n";
          ok = false;
        }
      for (const ParentLink &parent : graph.getParents(node))
        if (parent.offset == 0 && !parent.parent->hasFlags(flag)) {
          os << nodeName(node) << " is " << getFlagName(flag) << " but "
             << nodeName(parent.parent)
             << ", which it is at offset 0 of, is not\n";
          ok = false;
        }
    }
    if (node->hasFlags(NodeFlags::Summary)) {
      const Reason *why = node->getReason(NodeFlags::Summary);
      if (!why || why->kind != Reason::Kind::SummaryObject) {
        os << nodeName(node) << " is Summary without a SummaryObject reason\n";
        ok = false;
      }
    }
  }

  // Resolved indirect calls call only the targets found (T09's rule).
  bool anyClobber = false;
  for (const std::unique_ptr<TUFunction> &fn : analysis.getFunctions())
    for (const TUCallSite &call : fn->callSites) {
      anyClobber |= call.site.clobbersGlobals;
      if (call.resolution != CallResolution::Indirect)
        continue;
      auto report = [&]() -> InFlightDiagnostic {
        ok = false;
        return call.site.call->emitError()
               << "points-to: resolved indirect call ";
      };
      if (!call.site.calleePtr) {
        report() << "has no callee cell";
        continue;
      }
      const Node *callee = call.site.calleePtr->getNode();
      if (hasAny(callee->getFlags(), NodeFlags::UnknownObj |
                                         NodeFlags::External |
                                         NodeFlags::Incomplete)) {
        report() << "has a callee node that is "
                 << toString(callee->getFlags() &
                             (NodeFlags::UnknownObj | NodeFlags::External |
                              NodeFlags::Incomplete));
        continue;
      }
      for (const Origin &origin : callee->getOrigins()) {
        auto sym = dyn_cast_if_present<FlatSymbolRefAttr>(origin.sym);
        if (origin.kind != Origin::Kind::FuncSym || !sym) {
          report() << "has a callee node with an origin that is not a "
                      "function";
          break;
        }
        if (!llvm::is_contained(call.targets, sym)) {
          report() << "has a callee node with the function " << sym
                   << ", which is not among its targets";
          break;
        }
      }
    }

  // clobbersGlobals has been applied.
  if (anyClobber) {
    constexpr NodeFlags clobber =
        NodeFlags::External | NodeFlags::Mod | NodeFlags::Ref;
    SymbolTableCollection symbols;
    for (auto [sym, node] : analysis.getSymbolNodes()) {
      auto global = symbols.lookupNearestSymbolFrom<cir::GlobalOp>(module, sym);
      if (!global || global.getConstant())
        continue;
      if (cir::isLocalLinkage(global.getLinkage()) &&
          !hasAny(node->getFlags(),
                  NodeFlags::External | NodeFlags::Incomplete))
        continue;
      if (!node->hasFlags(clobber)) {
        os << "global " << sym << " (" << nodeName(node)
           << ") is not clobbered by the unknown calls\n";
        ok = false;
      }
    }
  }

  if (!errors.empty())
    module.emitError() << "points-to analysis failed to verify:\n" << errors;
  return success(ok);
}
