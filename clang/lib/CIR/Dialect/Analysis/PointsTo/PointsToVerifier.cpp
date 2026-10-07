//===- PointsToVerifier.cpp - Self-check of points-to graphs --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToVerifier.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/LocalBuilder.h"
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
  for (const CallSiteRecord &site : fg.callSites)
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
