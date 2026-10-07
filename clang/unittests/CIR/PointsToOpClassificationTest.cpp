//===- PointsToOpClassificationTest.cpp - Points-to op table + self-check -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The points-to classification table must name every op the CIR dialect
// registers, exactly once, and nothing else. The second half checks that
// PointsToVerifier catches a graph that misses a value or a call.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/LocalBuilder.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/OpClassification.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToVerifier.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <gtest/gtest.h>

using namespace mlir;
using namespace cir::pointsto;

namespace {

llvm::StringSet<> getRegisteredCIROps(MLIRContext &context) {
  context.getOrLoadDialect<cir::CIRDialect>();
  llvm::StringSet<> names;
  for (RegisteredOperationName name : context.getRegisteredOperations())
    if (name.getDialectNamespace() == cir::CIRDialect::getDialectNamespace())
      names.insert(name.getStringRef());
  return names;
}

TEST(PointsToOpClassificationTest, EveryRegisteredOpIsClassified) {
  MLIRContext context;
  llvm::StringSet<> registered = getRegisteredCIROps(context);
  ASSERT_FALSE(registered.empty());

  std::vector<std::string> missing;
  for (const auto &entry : registered)
    if (!lookupOpClass(entry.getKey()))
      missing.push_back(entry.getKey().str());
  llvm::sort(missing);
  std::string list;
  for (const std::string &name : missing)
    list += "\n  " + name;
  EXPECT_TRUE(missing.empty())
      << missing.size()
      << " CIR op(s) are not in the points-to classification table:" << list
      << "\nClassify each of them as Handled, NoPointerSemantics or "
         "Pessimistic in clang/lib/CIR/Dialect/Analysis/PointsTo/"
         "OpClassification.cpp (until then the analysis treats them as "
         "Pessimistic).";
}

TEST(PointsToOpClassificationTest, NoStaleOrDuplicateEntries) {
  MLIRContext context;
  llvm::StringSet<> registered = getRegisteredCIROps(context);

  llvm::StringMap<unsigned> seen;
  std::string stale, duplicates;
  for (const OpClassEntry &entry : getOpClassificationTable()) {
    if (!registered.contains(entry.name))
      stale += "\n  " + entry.name.str();
    if (++seen[entry.name] == 2)
      duplicates += "\n  " + entry.name.str();
  }
  EXPECT_TRUE(stale.empty())
      << "the points-to classification table names ops the CIR dialect no "
         "longer registers; remove them:"
      << stale;
  EXPECT_TRUE(duplicates.empty())
      << "ops classified more than once:" << duplicates;
  EXPECT_EQ(seen.size(), registered.size());
}

//===----------------------------------------------------------------------===//
// PointsToVerifier
//===----------------------------------------------------------------------===//

class PointsToVerifierTest : public ::testing::Test {
protected:
  PointsToVerifierTest() { context.loadDialect<cir::CIRDialect>(); }

  /// Parse \p ir, build the graph of its first function with a body.
  std::unique_ptr<FunctionGraph> build(StringRef ir) {
    module = parseSourceString<ModuleOp>(ir, &context);
    EXPECT_TRUE(module) << "failed to parse IR";
    cir::FuncOp func;
    module->walk([&](cir::FuncOp f) {
      if (f.isDeclaration())
        return WalkResult::advance();
      func = f;
      return WalkResult::interrupt();
    });
    return LocalBuilder(func).build();
  }

  /// Verify \p fg and return the error messages it emitted.
  std::vector<std::string> verify(const FunctionGraph &fg, bool &ok) {
    std::vector<std::string> messages;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diag) {
      messages.push_back(diag.str());
      return success();
    });
    ok = succeeded(verifyFunctionGraph(fg));
    return messages;
  }

  static bool contains(const std::vector<std::string> &messages,
                       StringRef needle) {
    return llvm::any_of(messages, [&](const std::string &message) {
      return StringRef(message).contains(needle);
    });
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
};

constexpr StringLiteral kIR = R"CIR(
!s32i = !cir.int<s, 32>
!rec_S = !cir.struct<"S" {data !cir.ptr<!s32i>, data !s32i}>
module {
  cir.func private @g(!cir.ptr<!s32i>) -> !rec_S
  cir.func @f(%p: !cir.ptr<!s32i>) {
    %a = cir.alloca "a" align(8) : !cir.ptr<!rec_S>
    %v = cir.call @g(%p) : (!cir.ptr<!s32i>) -> !rec_S
    cir.store %v, %a : !rec_S, !cir.ptr<!rec_S>
    cir.return
  }
}
)CIR";

TEST_F(PointsToVerifierTest, CleanGraphVerifies) {
  std::unique_ptr<FunctionGraph> fg = build(kIR);
  bool ok = false;
  std::vector<std::string> messages = verify(*fg, ok);
  EXPECT_TRUE(ok);
  EXPECT_TRUE(messages.empty());
}

TEST_F(PointsToVerifierTest, MissingCell) {
  std::unique_ptr<FunctionGraph> fg = build(kIR);
  Value alloca;
  module->walk([&](cir::AllocaOp op) { alloca = op.getResult(); });
  ASSERT_TRUE(fg->scalarMap.erase(alloca));
  bool ok = true;
  std::vector<std::string> messages = verify(*fg, ok);
  EXPECT_FALSE(ok);
  EXPECT_TRUE(contains(messages, "pointer value has no cell"));
}

TEST_F(PointsToVerifierTest, MissingSlots) {
  std::unique_ptr<FunctionGraph> fg = build(kIR);
  Value call;
  module->walk([&](cir::CallOp op) { call = op.getResult(); });
  ASSERT_TRUE(fg->lookupSlots(call));
  fg->aggregateMap.erase(call);
  bool ok = true;
  std::vector<std::string> messages = verify(*fg, ok);
  EXPECT_FALSE(ok);
  EXPECT_TRUE(contains(messages, "has no slot record"));
}

TEST_F(PointsToVerifierTest, CallSiteRecordCount) {
  std::unique_ptr<FunctionGraph> fg = build(kIR);
  ASSERT_EQ(fg->callSites.size(), 1u);
  fg->callSites.push_back(fg->callSites.front());
  bool ok = true;
  std::vector<std::string> messages = verify(*fg, ok);
  EXPECT_FALSE(ok);
  EXPECT_TRUE(contains(messages, "call has 2 call-site records"));

  fg->callSites.clear();
  messages = verify(*fg, ok);
  EXPECT_FALSE(ok);
  EXPECT_TRUE(contains(messages, "call has 0 call-site records"));
}

} // namespace
