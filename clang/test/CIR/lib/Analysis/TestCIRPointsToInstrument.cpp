//===- TestCIRPointsToInstrument.cpp - Check points-to facts at run time --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// -test-cir-points-to-instrument turns the NoAlias answers of the points-to
// analysis into run-time checks (dynamic points-to validation, after Mock,
// Das, Chambers and Eggers, "Dynamic points-to sets: a comparison with static
// analyses", PASTE 2001). For every function with a body it:
//
//  1. collects the address operands of its memory accesses, each with the
//     number of bytes it touches: cir.load, cir.store, cir.copy (dst, src),
//     cir.libc.memcpy and cir.libc.memmove (dst, src), cir.libc.memset and
//     the atomics (fetch, xchg, cmpxchg, test_and_set, clear). An access
//     whose size is neither a constant nor an SSA length is left out;
//  2. pairs every two accesses (p at X, q at Y) with p != q, picking the
//     later one as the check point: Y if X dominates it, X if Y dominates it,
//     else whichever one the other address (and its length) dominates. A
//     pair with no such point is not a candidate. Candidates get ids in
//     module order (function, then the accesses in walk order), which depend
//     on the IR alone, not on the analysis;
//  3. asks PointsToAnalysis::aliasWithRule(p, q, sizeA, sizeB), and keeps the
//     NoAlias candidates. If a function has more than max-pairs-per-func
//     (default 64, 0 = all), it keeps the first ones by a hash of the seed
//     and the id;
//  4. inserts before the check point
//       cir.call @__cir_pt_check(p, sizeA, q, sizeB, id)
//     with the pointers cast to !cir.ptr<!void> and the sizes and id as
//     !u64i, and declares __cir_pt_check once in the module;
//  5. with must=1 (the default), does the same for the MustAlias answers,
//     sampled the same way, with
//       cir.call @__cir_pt_check_must(p, q, id)
//     but only for a pair whose later access is the check point and whose
//     earlier one dominates it (or one op with both addresses), both of a
//     constant non-zero size: on every path through the check both accesses
//     run, so the claim is about these two values.
//
//  6. with values=1, for every cir.load of an integer, bool, floating-point
//     or pointer value for which PointsToAnalysis::isConstantAt gives a
//     constant c (StoredValues.h), inserts after the load
//       cir.call @__cir_pt_check_value(loaded, c, id)
//     with both values as their bits in a !u64i (floating-point values
//     bitcast, pointers converted). Value ids are their own sequence, in
//     module order; the table lines read "value <id> @<func> <c> load(<def>)
//     at <loc>".
//
// The runtime (clang/test/CIR/Analysis/points-to/Inputs/pt_check_runtime.c)
// aborts with "PT-VIOLATION <id>" if the two byte ranges of a NoAlias pair
// overlap, and with "PT-MUST-VIOLATION <id>" if the two addresses of a
// MustAlias pair differ, and with "PT-VALUE-VIOLATION <id>" if a load
// claimed constant reads another value. A violation is an unsound answer, and the side
// table names the pair; its MustAlias lines read "<id> @<func> must <rule>
// ...".
//
// Options:
//   closed-tu=1               PointsToAnalysisOptions::assumeClosedTU
//   mode=bu                   the bottom-up mode (PointsToMode::BU)
//   td=0                      bu: answer from each function's own graph
//                             (PointsToAnalysisOptions::topDownQueries)
//   max-query-contexts=<n>    bu: PointsToAnalysisOptions::maxQueryContexts
//   budget-scale=<f>          PointsToAnalysisOptions::budgetScale (the
//                             stress run sets every budget tiny with 0.001)
//   slice=1                   bu: PointsToAnalysisOptions::relevanceSlice
//   max-pairs-per-func=<n>    as above
//   seed=<s>                  the sampling seed
//   table=<path>              write the side table: one line per check,
//                               <id> @<func> <rule> <a> | <b> at <loc> | why
//                             where <a> and <b> are "<access>(<def>) <size>"
//                             and why is explainAlias; then the totals
//   print-candidates=1        also write every candidate that is not
//                             checked, prefixed with "cand", with its answer
//   must=0                    do not check the MustAlias answers
//   values=1                  also check the constants of the loads
//   inject-wrong-value=<id>   test only: check value <id> against its
//                             constant plus one (a negative control: the
//                             run must abort)
//   inject-false-noalias=<id> test only: treat candidate <id> as NoAlias
//                             (and always check it), whatever the analysis
//                             says. A negative control: when the pair
//                             really aliases, the run must abort.
//
// Caveat: the static answer is about the objects p and q point to while both
// are live. A pointer to a freed object compared with a new allocation that
// reuses its memory is a false violation; reduce any violation before
// blaming the analysis.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "clang/CIR/Dialect/Analysis/CIRPointerDecomposition.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToAnalysis.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToModels.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/StoredValues.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <optional>

using namespace mlir;
using namespace cir::pointsto;

namespace {

constexpr llvm::StringLiteral kCheckName = "__cir_pt_check";
constexpr llvm::StringLiteral kMustCheckName = "__cir_pt_check_must";
constexpr llvm::StringLiteral kValueCheckName = "__cir_pt_check_value";
constexpr uint64_t kNoInjection = std::numeric_limits<uint64_t>::max();

/// One address operand of a memory access.
struct Access {
  Operation *op;
  Value addr;
  /// What the access is: "load", "store", "copy.dst", ...
  StringRef role;
  /// The size for the static query: a constant, or kUnknownSize.
  uint64_t querySize;
  /// The size the check passes: a constant (if querySize is one) or the
  /// length operand of the op.
  std::optional<uint64_t> constSize;
  Value dynSize;
};

/// A pair of accesses checked before \p point.
struct Candidate {
  uint64_t id;
  unsigned a, b;
  Operation *point;
  AliasAnswer answer;
  bool injected = false;
  /// A MustAlias check (p == q) rather than an overlap check.
  bool must = false;
};

/// splitmix64: a hash that is the same in every run (llvm::hash_combine is
/// seeded per process in builds with assertions).
uint64_t mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

/// "file:line:col" of the first file location in \p loc, or "?".
std::string formatLoc(Location loc) {
  std::string out = "?";
  loc->walk([&](Location inner) {
    if (auto file = dyn_cast<FileLineColLoc>(inner)) {
      out = (llvm::sys::path::filename(file.getFilename().getValue()) + ":" +
             Twine(file.getLine()) + ":" + Twine(file.getColumn()))
                .str();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return out;
}

/// The op that defines \p value and where, or the block argument.
std::string describeValue(Value value) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    Operation *parent = arg.getOwner()->getParentOp();
    bool entry = arg.getOwner()->isEntryBlock() && isa<cir::FuncOp>(parent);
    return ((entry ? "formal#" : "blockarg#") + Twine(arg.getArgNumber()))
        .str();
  }
  Operation *def = value.getDefiningOp();
  std::string name = def->getName().getStringRef().str();
  if (auto alloca = dyn_cast<cir::AllocaOp>(def))
    name += " '" + alloca.getName().str() + "'";
  else if (auto global = dyn_cast<cir::GetGlobalOp>(def))
    name += " @" + global.getName().str();
  else if (auto call = dyn_cast<cir::CallOp>(def); call && call.getCallee())
    name += " @" + call.getCallee()->str();
  return name + "@" + formatLoc(def->getLoc());
}

struct TestCIRPointsToInstrumentPass
    : public PassWrapper<TestCIRPointsToInstrumentPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCIRPointsToInstrumentPass)

  TestCIRPointsToInstrumentPass() = default;
  TestCIRPointsToInstrumentPass(const TestCIRPointsToInstrumentPass &pass)
      : PassWrapper(pass) {}

  StringRef getArgument() const final {
    return "test-cir-points-to-instrument";
  }
  StringRef getDescription() const final {
    return "Insert run-time overlap checks for points-to NoAlias facts.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cir::CIRDialect>();
  }

  Option<bool> closedTU{
      *this, "closed-tu",
      llvm::cl::desc("Points-to: assume every caller is in the module"),
      llvm::cl::init(false)};
  Option<std::string> ptMode{
      *this, "mode",
      llvm::cl::desc("Points-to: ci (context-insensitive) or bu (bottom-up)"),
      llvm::cl::init("ci")};
  Option<bool> subobjects{
      *this, "subobjects",
      llvm::cl::desc("Points-to: give class, union and array members their "
                     "own nodes"),
      llvm::cl::init(false)};
  Option<bool> topDown{
      *this, "td",
      llvm::cl::desc("Points-to, bu: answer from the calling contexts"),
      llvm::cl::init(true)};
  Option<unsigned> maxQueryContexts{
      *this, "max-query-contexts",
      llvm::cl::desc("Points-to, bu: the most calling contexts per query"),
      llvm::cl::init(64)};
  Option<double> budgetScale{
      *this, "budget-scale",
      llvm::cl::desc("Points-to: multiply every budget by this factor"),
      llvm::cl::init(1.0)};
  Option<bool> slice{
      *this, "slice",
      llvm::cl::desc("Points-to, bu: copy only the callees in the relevance "
                     "slice"),
      llvm::cl::init(false)};
  Option<unsigned> maxPairs{
      *this, "max-pairs-per-func",
      llvm::cl::desc("The most pairs checked per function (0 = all)"),
      llvm::cl::init(64)};
  Option<unsigned> seed{*this, "seed", llvm::cl::desc("The sampling seed"),
                        llvm::cl::init(1)};
  Option<std::string> tablePath{
      *this, "table", llvm::cl::desc("Write the side table to this file"),
      llvm::cl::init("")};
  Option<bool> checkMust{
      *this, "must",
      llvm::cl::desc("Also check the MustAlias answers (p == q)"),
      llvm::cl::init(true)};
  Option<bool> printCandidates{
      *this, "print-candidates",
      llvm::cl::desc("Also write the candidates that are not checked"),
      llvm::cl::init(false)};
  Option<bool> checkValues{
      *this, "values",
      llvm::cl::desc("Also check the constants of the loads (isConstantAt)"),
      llvm::cl::init(false)};
  Option<uint64_t> injectWrongValue{
      *this, "inject-wrong-value",
      llvm::cl::desc("Test only: check this value against a wrong constant"),
      llvm::cl::init(kNoInjection)};
  Option<uint64_t> injectFalseNoAlias{
      *this, "inject-false-noalias",
      llvm::cl::desc("Test only: check this candidate as if it were NoAlias"),
      llvm::cl::init(kNoInjection)};

  void runOnOperation() override;
};

/// The accesses of \p func, in walk order.
SmallVector<Access> collectAccesses(cir::FuncOp func,
                                    const DataLayout &dataLayout) {
  SmallVector<Access> accesses;
  auto add = [&](Operation *op, Value addr, StringRef role,
                 std::optional<uint64_t> size, Value dynSize = {}) {
    auto ptrTy = dyn_cast<cir::PointerType>(addr.getType());
    // Host code only: a pointer in another address space is left out.
    if (!ptrTy || ptrTy.getAddrSpace())
      return;
    if (!size) {
      std::optional<int64_t> bytes =
          cir::getTypeSizeInBytes(ptrTy.getPointee(), dataLayout);
      if (bytes && *bytes >= 0)
        size = static_cast<uint64_t>(*bytes);
    }
    Access access{op, addr, role, PointsToAnalysis::kUnknownSize, size, {}};
    if (size)
      access.querySize = *size;
    else if (dynSize && isa<cir::IntType>(dynSize.getType()))
      access.dynSize = dynSize;
    else
      return;
    accesses.push_back(access);
  };
  // A memory op's length: a constant, or the operand itself.
  auto length = [](Value len) -> std::pair<std::optional<uint64_t>, Value> {
    std::optional<int64_t> constant = cir::getConstantIndex(len);
    if (constant && *constant >= 0)
      return {static_cast<uint64_t>(*constant), Value()};
    return {std::nullopt, len};
  };
  func.walk([&](Operation *op) {
    llvm::TypeSwitch<Operation *>(op)
        .Case<cir::LoadOp>([&](cir::LoadOp load) {
          add(op, load.getAddr(), "load", std::nullopt);
        })
        .Case<cir::StoreOp>([&](cir::StoreOp store) {
          add(op, store.getAddr(), "store", std::nullopt);
        })
        .Case<cir::CopyOp>([&](cir::CopyOp copy) {
          uint64_t size = copy.getCopySizeInBytes(dataLayout);
          add(op, copy.getDst(), "copy.dst", size);
          add(op, copy.getSrc(), "copy.src", size);
        })
        .Case<cir::MemCpyOp, cir::MemMoveOp>([&](auto mem) {
          auto [size, dyn] = length(mem.getLen());
          bool isCopy = isa<cir::MemCpyOp>(op);
          add(op, mem.getDst(), isCopy ? "memcpy.dst" : "memmove.dst", size,
              dyn);
          add(op, mem.getSrc(), isCopy ? "memcpy.src" : "memmove.src", size,
              dyn);
        })
        .Case<cir::MemSetOp>([&](cir::MemSetOp mem) {
          auto [size, dyn] = length(mem.getLen());
          add(op, mem.getDst(), "memset.dst", size, dyn);
        })
        .Case<cir::AtomicFetchOp, cir::AtomicXchgOp, cir::AtomicCmpXchgOp,
              cir::AtomicTestAndSetOp, cir::AtomicClearOp>([&](auto atomic) {
          add(op, atomic.getPtr(), "atomic", std::nullopt);
        });
  });
  return accesses;
}

void TestCIRPointsToInstrumentPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = &getContext();

  PointsToModelRegistry registry;
  registerGenericModels(registry);
  PointsToAnalysisOptions options;
  options.assumeClosedTU = closedTU;
  if (ptMode == "bu")
    options.mode = PointsToMode::BU;
  options.subobjects = subobjects;
  options.topDownQueries = topDown;
  options.maxQueryContexts = maxQueryContexts;
  options.budgetScale = budgetScale;
  options.relevanceSlice = slice;
  PointsToAnalysis pt(module, &registry, options);
  DataLayout dataLayout(module);

  std::unique_ptr<llvm::raw_fd_ostream> table;
  if (!tablePath.empty()) {
    std::error_code error;
    table = std::make_unique<llvm::raw_fd_ostream>(tablePath, error,
                                                   llvm::sys::fs::OF_Text);
    if (error) {
      module.emitError() << "cannot open " << tablePath << ": "
                         << error.message();
      return signalPassFailure();
    }
    *table << "# test-cir-points-to-instrument closed-tu=" << closedTU
           << " max-pairs-per-func=" << maxPairs << " seed=" << seed << "\n";
  }

  // All the queries run before the IR changes.
  struct FunctionPlan {
    cir::FuncOp func;
    SmallVector<Access> accesses;
    SmallVector<Candidate> checks;
  };
  SmallVector<FunctionPlan> plans;
  uint64_t nextId = 0, numCandidates = 0, numNoAlias = 0, numMust = 0;
  llvm::StringMap<unsigned> checkedRules;

  auto describeAccess = [](const Access &access) {
    std::string size = access.constSize ? std::to_string(*access.constSize)
                                        : std::string("dyn");
    return (access.role + "(" + describeValue(access.addr) + ") " + size)
        .str();
  };

  for (cir::FuncOp func : module.getOps<cir::FuncOp>()) {
    if (func.isDeclaration())
      continue;
    FunctionPlan plan{func, collectAccesses(func, dataLayout), {}};
    DominanceInfo dom(func);
    const SmallVector<Access> &accesses = plan.accesses;
    // A cir.goto names its label; it has no successor until the goto
    // solver runs. So the CFG DominanceInfo sees lacks the edges into the
    // labels, and a block or region a goto enters looks dominated by code
    // the goto bypasses (`goto L; p = &g; { L: *p; }`). In a function with a
    // goto, trust the order of \p point after a value defined in
    // \p defBlock only along structured nesting (each block above \p point
    // the entry of its region, up to \p defBlock), and only if no label is
    // between them: none in the ops of \p defBlock from the definition up to
    // the one holding \p point, nor in the nested blocks before \p point.
    bool hasGoto = false;
    func.walk([&](cir::GotoOp) { hasGoto = true; });
    // For each op: whether it holds a label (is one, or has one in a region),
    // and how many ops of its block before it do.
    llvm::DenseMap<Operation *, bool> holdsLabel;
    llvm::DenseMap<Operation *, unsigned> labelsBefore;
    if (hasGoto) {
      func.walk<WalkOrder::PostOrder>([&](Operation *op) {
        bool held = isa<cir::LabelOp>(op);
        for (Region &region : op->getRegions())
          for (Block &block : region)
            for (Operation &inner : block)
              held |= holdsLabel.lookup(&inner);
        holdsLabel[op] = held;
      });
      func.walk([&](Block *block) {
        unsigned count = 0;
        for (Operation &op : *block) {
          labelsBefore[&op] = count;
          count += holdsLabel.lookup(&op);
        }
      });
    }
    auto hasLabel = [&](Operation *op) { return holdsLabel.lookup(op); };
    // Whether an op of \p op's block before \p op (after \p from, or from
    // the start) holds a label.
    auto labelBefore = [&](Operation *op, Operation *from) {
      unsigned upTo = labelsBefore.lookup(op);
      unsigned skip = from ? labelsBefore.lookup(from) + hasLabel(from) : 0;
      return upTo > skip;
    };
    // Whether \p point runs after \p def (null: the start of \p defBlock)
    // on every path, given that DominanceInfo says so.
    auto reliable = [&](Block *defBlock, Operation *def, Operation *point) {
      if (!hasGoto)
        return true;
      for (Operation *op = point; op && op != func.getOperation();
           op = op->getParentOp()) {
        Block *block = op->getBlock();
        if (block == defBlock)
          return !labelBefore(op, def) && (op == point || !hasLabel(op));
        if (!block->isEntryBlock() || labelBefore(op, nullptr))
          return false;
      }
      return false;
    };
    auto opDominates = [&](Operation *a, Operation *b) {
      return dom.properlyDominates(a, b) && reliable(a->getBlock(), a, b);
    };
    auto valueDominates = [&](Value value, Operation *point) {
      if (!dom.properlyDominates(value, point))
        return false;
      if (Operation *def = value.getDefiningOp())
        return reliable(def->getBlock(), def, point);
      return reliable(cast<BlockArgument>(value).getOwner(), nullptr, point);
    };
    // Whether the operands a check of \p access needs are available at
    // \p point.
    auto available = [&](const Access &access, Operation *point) {
      if (access.op == point)
        return true;
      if (!valueDominates(access.addr, point))
        return false;
      return !access.dynSize || valueDominates(access.dynSize, point);
    };
    SmallVector<Candidate> noAlias, mustAlias;
    for (unsigned j = 0; j < accesses.size(); ++j)
      for (unsigned i = 0; i < j; ++i) {
        const Access &a = accesses[i], &b = accesses[j];
        if (a.addr == b.addr)
          continue;
        Operation *point = nullptr;
        if (a.op == b.op || opDominates(a.op, b.op))
          point = b.op;
        else if (opDominates(b.op, a.op))
          point = a.op;
        else if (available(a, b.op))
          point = b.op;
        else if (available(b, a.op))
          point = a.op;
        if (!point || !available(a, point) || !available(b, point))
          continue;
        Candidate candidate{nextId++, i, j, point,
                            pt.aliasWithRule(a.addr, b.addr, a.querySize,
                                             b.querySize)};
        ++numCandidates;
        if (candidate.id == injectFalseNoAlias)
          candidate.injected = true;
        if (candidate.answer.result.isNo())
          ++numNoAlias;
        // A MustAlias pair is checked only where both accesses run: the
        // check point is the later access, which the earlier one dominates.
        bool ordered = a.op == b.op ||
                       (point == b.op && opDominates(a.op, b.op)) ||
                       (point == a.op && opDominates(b.op, a.op));
        bool sized = a.constSize && *a.constSize && b.constSize &&
                     *b.constSize;
        if (candidate.answer.result.isMust())
          ++numMust;
        if (candidate.answer.result.isNo() || candidate.injected) {
          noAlias.push_back(candidate);
        } else if (checkMust && candidate.answer.result.isMust() && ordered &&
                   sized) {
          candidate.must = true;
          mustAlias.push_back(candidate);
        } else if (table && printCandidates) {
          *table << "cand " << candidate.id << " @" << func.getSymName()
                 << " " << candidate.answer.result << " "
                 << getAliasRuleName(candidate.answer.rule) << " "
                 << describeAccess(a) << " | " << describeAccess(b) << "\n";
        }
      }

    // Sample: the injected pair, then the first ones by hash.
    auto sample = [&](SmallVector<Candidate> &list) {
      if (!maxPairs || list.size() <= maxPairs)
        return;
      auto key = [&](const Candidate &c) {
        return c.injected ? 0 : mix(mix(seed) ^ c.id) | 1;
      };
      llvm::stable_sort(list, [&](const Candidate &x, const Candidate &y) {
        return key(x) < key(y);
      });
      if (table && printCandidates)
        for (const Candidate &c : llvm::ArrayRef(list).drop_front(maxPairs))
          *table << "cand " << c.id << " @" << func.getSymName() << " "
                 << (c.must ? "MustAlias" : "NoAlias") << "(not sampled) "
                 << getAliasRuleName(c.answer.rule) << " "
                 << describeAccess(accesses[c.a]) << " | "
                 << describeAccess(accesses[c.b]) << "\n";
      list.truncate(maxPairs);
    };
    sample(noAlias);
    sample(mustAlias);
    noAlias.append(mustAlias.begin(), mustAlias.end());
    llvm::sort(noAlias, [](const Candidate &x, const Candidate &y) {
      return x.id < y.id;
    });
    for (const Candidate &c : noAlias) {
      const Access &a = accesses[c.a], &b = accesses[c.b];
      StringRef rule =
          c.injected ? StringRef("injected") : getAliasRuleName(c.answer.rule);
      ++checkedRules[c.must ? ("must " + rule).str() : rule.str()];
      if (!table)
        continue;
      *table << c.id << " @" << func.getSymName() << " "
             << (c.must ? "must " : "") << rule;
      if (c.injected)
        *table << "(really " << c.answer.result << " "
               << getAliasRuleName(c.answer.rule) << ")";
      *table << " " << describeAccess(a) << " | " << describeAccess(b)
             << " at " << c.point->getName() << "@"
             << formatLoc(c.point->getLoc()) << " | "
             << pt.explainAlias(a.addr, b.addr, a.querySize, b.querySize)
             << "\n";
    }
    plan.checks = std::move(noAlias);
    plans.push_back(std::move(plan));
  }

  // The constants of the loads (values=1), asked before the IR changes.
  struct ValueCheck {
    cir::LoadOp load;
    TypedAttr constant;
    uint64_t id;
  };
  SmallVector<ValueCheck> valueChecks;
  if (checkValues) {
    auto checkable = [&](Type type) {
      if (isa<cir::BoolType, cir::PointerType>(type))
        return true;
      if (auto intTy = dyn_cast<cir::IntType>(type))
        return intTy.getWidth() <= 64;
      if (auto fpTy = dyn_cast<cir::FPTypeInterface>(type)) {
        unsigned width = llvm::APFloat::getSizeInBits(fpTy.getFloatSemantics());
        return width == 32 || width == 64;
      }
      return false;
    };
    uint64_t nextValueId = 0;
    for (FunctionPlan &plan : plans)
      plan.func.walk([&](cir::LoadOp load) {
        auto ptrTy = dyn_cast<cir::PointerType>(load.getAddr().getType());
        if (!ptrTy || ptrTy.getAddrSpace() || !checkable(load.getType()))
          return;
        std::optional<TypedAttr> c =
            pt.isConstantAt(load.getAddr(), std::nullopt, load.getType());
        if (!c)
          return;
        // A pointer constant: only null is checkable as bits.
        if (auto ptr = dyn_cast<cir::ConstPtrAttr>(*c); ptr && !ptr.isNullValue())
          return;
        if (!isa<cir::IntAttr, cir::BoolAttr, cir::FPAttr, cir::ConstPtrAttr>(
                *c))
          return;
        ValueCheck check{load, *c, nextValueId++};
        if (table)
          *table << "value " << check.id << " @"
                 << plan.func.getSymName() << " " << check.constant << " load("
                 << describeValue(load.getAddr()) << ") at "
                 << formatLoc(load.getLoc()) << "\n";
        valueChecks.push_back(check);
      });
  }

  // Instrument.
  cir::CIRBaseBuilderTy builder(*ctx);
  cir::PointerType voidPtrTy = builder.getVoidPtrTy();
  cir::IntType u64Ty = builder.getUIntNTy(64);
  uint64_t numChecks = 0, numMustChecks = 0;
  for (const FunctionPlan &plan : plans)
    for (const Candidate &c : plan.checks)
      ++(c.must ? numMustChecks : numChecks);
  // The runtime function \p name of type (\p params) -> void, declared once.
  auto declare = [&](StringRef name, ArrayRef<Type> params) {
    auto check = module.lookupSymbol<cir::FuncOp>(name);
    if (check)
      return check;
    builder.setInsertionPointToStart(module.getBody());
    auto fnTy = cir::FuncType::get(params, cir::VoidType::get(ctx));
    check = cir::FuncOp::create(builder, module.getLoc(), name, fnTy);
    check.setLinkageAttr(cir::GlobalLinkageKindAttr::get(
        ctx, cir::GlobalLinkageKind::ExternalLinkage));
    SymbolTable::setSymbolVisibility(check, SymbolTable::Visibility::Private);
    return check;
  };
  if (numChecks || numMustChecks) {
    cir::FuncOp check, mustCheck;
    if (numChecks)
      check =
          declare(kCheckName, {voidPtrTy, u64Ty, voidPtrTy, u64Ty, u64Ty});
    if (numMustChecks)
      mustCheck = declare(kMustCheckName, {voidPtrTy, voidPtrTy, u64Ty});
    for (const FunctionPlan &plan : plans)
      for (const Candidate &c : plan.checks) {
        builder.setInsertionPoint(c.point);
        Location loc = c.point->getLoc();
        auto pointer = [&](const Access &access) -> Value {
          if (access.addr.getType() == voidPtrTy)
            return access.addr;
          return builder.createBitcast(loc, access.addr, voidPtrTy);
        };
        auto size = [&](const Access &access) -> Value {
          if (access.constSize)
            return builder.getConstantInt(loc, u64Ty, *access.constSize);
          if (access.dynSize.getType() == u64Ty)
            return access.dynSize;
          return builder.createIntCast(access.dynSize, u64Ty);
        };
        const Access &a = plan.accesses[c.a], &b = plan.accesses[c.b];
        cir::CallOp call;
        if (c.must) {
          SmallVector<Value, 3> args{pointer(a), pointer(b),
                                     builder.getConstantInt(loc, u64Ty, c.id)};
          call = builder.createCallOp(loc, mustCheck, args);
        } else {
          SmallVector<Value, 5> args{pointer(a), size(a), pointer(b), size(b),
                                     builder.getConstantInt(loc, u64Ty, c.id)};
          call = builder.createCallOp(loc, check, args);
        }
        call.setNothrow(true);
      }
  }

  if (!valueChecks.empty()) {
    cir::FuncOp check = declare(kValueCheckName, {u64Ty, u64Ty, u64Ty});
    // The bits of \p value, of the load's type, in a !u64i.
    auto bits = [&](Location loc, Value value) -> Value {
      Type type = value.getType();
      if (isa<cir::BoolType>(type))
        return builder.createCast(loc, cir::CastKind::bool_to_int, value,
                                  u64Ty);
      if (isa<cir::PointerType>(type))
        return builder.createPtrToInt(value, u64Ty);
      if (auto fpTy = dyn_cast<cir::FPTypeInterface>(type)) {
        unsigned width = llvm::APFloat::getSizeInBits(fpTy.getFloatSemantics());
        Value raw = builder.createCast(loc, cir::CastKind::bitcast, value,
                                       builder.getUIntNTy(width));
        return width == 64 ? raw : builder.createIntCast(raw, u64Ty);
      }
      if (value.getType() == u64Ty)
        return value;
      return builder.createIntCast(value, u64Ty);
    };
    for (const ValueCheck &v : valueChecks) {
      cir::LoadOp load = v.load;
      Location loc = load.getLoc();
      builder.setInsertionPointAfter(load);
      TypedAttr expected = v.constant;
      if (v.id == injectWrongValue) {
        // The negative control: a constant that is not the one claimed.
        if (auto i = dyn_cast<cir::IntAttr>(expected))
          expected = cir::IntAttr::get(i.getType(), i.getValue() + 1);
        else if (auto b = dyn_cast<cir::BoolAttr>(expected))
          expected = cir::BoolAttr::get(ctx, !b.getValue());
      }
      Value constant = cir::ConstantOp::create(builder, loc, expected);
      SmallVector<Value, 3> args{bits(loc, load.getResult()),
                                 bits(loc, constant),
                                 builder.getConstantInt(loc, u64Ty, v.id)};
      builder.createCallOp(loc, check, args).setNothrow(true);
    }
  }

  if (table) {
    *table << "# functions " << plans.size() << " candidates " << numCandidates
           << " noalias " << numNoAlias << " checks " << numChecks << " must "
           << numMust << " must-checks " << numMustChecks << " values "
           << valueChecks.size() << "\n";
    SmallVector<std::pair<StringRef, unsigned>> rules;
    for (const auto &entry : checkedRules)
      rules.emplace_back(entry.getKey(), entry.getValue());
    llvm::sort(rules);
    for (auto [rule, count] : rules)
      *table << "# rule " << rule << " " << count << "\n";
    *table << "# budget trips " << pt.formatBudgetTrips() << "\n";
  }
}

} // namespace

namespace cir {
namespace test {
void registerTestCIRPointsToInstrumentPass() {
  PassRegistration<TestCIRPointsToInstrumentPass>();
}
} // namespace test
} // namespace cir
