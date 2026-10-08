//===- StoredValues.h - Stored-value queries over points-to cells -*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the stored-value queries of the CIR points-to analysis:
// what values a memory cell of the points-to graph may hold, over every write
// the module makes to it. It generalizes the MemoryReader of the legacy
// KernelBindingTable (valueAt, uniqueInitializer), which works on an inlined
// copy of the launch path, to the interprocedural graph, with no inlining.
//
//  - StoredValueIndex: every write of the module, indexed by the node it
//    writes (and by each ancestor of that node in its tree of subobjects), and
//    the inverse of the bottom-up copies (copy node -> source node), so that a
//    cell of a caller's graph finds the writes of the callees whose graphs
//    were copied into it.
//  - StoredValueQuery::valuesAt: the union of the writes that may reach a
//    cell (see StoredValues.cpp for the rules and their soundness).
//
// Queries are flow-insensitive: the `at` operation is reserved for strong
// updates (T24) and ignored.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_STOREDVALUES_H
#define CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_STOREDVALUES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToGraph.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

namespace cir::pointsto {

class PointsToAnalysis;
struct TUClone;
struct TUFunction;

/// What a cell may hold.
struct StoredValues {
  /// SSA values whose run-time value the cell may hold (each at some
  /// execution of its op), when no constant is known for them.
  llvm::SmallVector<mlir::Value, 2> defs;
  /// Constants the cell may hold, without duplicates.
  llvm::SmallVector<mlir::TypedAttr, 2> consts;
  /// Whether defs and consts are all the cell may hold.
  bool complete = true;
  /// Why the set is incomplete: the first reason found.
  const Reason *why = nullptr;
  /// Why the first def that is a load was not resolved to what it reads,
  /// if it was asked (an explanation only).
  const Reason *defWhy = nullptr;

  void addConst(mlir::TypedAttr value);
  void addDef(mlir::Value value);
  void setIncomplete(const Reason *reason);
  /// Add everything in \p other.
  void merge(const StoredValues &other);
};

/// A write to memory: what the module may store into a range of a node.
struct WriteRecord {
  enum class Kind : uint8_t {
    /// cir.store of `value` (a scalar, or an aggregate the range covers).
    Store,
    /// A copy of `size` bytes from `copySrc` (cir.copy, cir.libc.memcpy,
    /// cir.libc.memmove, a model's MemTransferRecord).
    Copy,
    /// cir.libc.memset with the byte `pattern` (std::nullopt: not a
    /// constant).
    Memset,
    /// The initial value `init` of a global (null: zero-initialized).
    GlobalInit,
    /// A write of an unknown value (a model's Mod effect, an atomic, a
    /// bitfield store, an op with an unknown memory effect).
    Unknown,
  };
  Kind kind;
  /// The writing op (the cir.global for GlobalInit).
  mlir::Operation *op = nullptr;
  /// Where it writes, a cell of the graph of `owner` (the TU graph).
  Cell dst;
  /// The bytes written; kUnknownSize for an unknown length (to the end).
  uint64_t size = 0;
  /// Unknown: anywhere in the node of `dst` (and, with `reachable`, in every
  /// node reachable from it).
  bool anywhere = false;
  bool reachable = false;
  mlir::Value value;
  Cell copySrc;
  mlir::Attribute init;
  std::optional<uint8_t> pattern;
  /// The function whose graph holds `dst` (null for a global's initializer).
  const TUFunction *owner = nullptr;
};

/// Where an access is, in one node: the bytes [lo, hi) if `exact`;
/// somewhere within [lo, hi) otherwise (hi may be kUnbounded). On an array
/// node the offsets are element-relative (one element of `elemSize`).
struct WriteLocation {
  static constexpr int64_t kUnbounded = INT64_MAX;
  int64_t lo = 0;
  int64_t hi = kUnbounded;
  bool exact = false;
};

/// Every write of a module and the inverse of the bottom-up copies.
class StoredValueIndex {
public:
  explicit StoredValueIndex(const PointsToAnalysis &analysis);

  /// A write indexed at a node, located in that node.
  struct Entry {
    unsigned write;
    WriteLocation loc;
  };
  /// A node of a caller's graph that is the copy of \p source (resolved) of
  /// the callee's graph made by \p clone: (copy, o) is (source, o + shift).
  struct Inverse {
    const TUClone *clone;
    const Node *source;
    int64_t shift;
  };

  llvm::ArrayRef<WriteRecord> getWrites() const { return writes; }
  llvm::ArrayRef<Entry> getEntries(const Node *node) const;
  llvm::ArrayRef<Inverse> getInverses(const Node *node) const;
  /// BU: whether a caller whose graph copied this node's function may write
  /// \p node: it is reachable from the formals or the return of a function
  /// with a copied caller, and not shared.
  const Reason *getCallerWriter(const Node *node) const;
  /// BU: whether \p node is shared (reachable from a global) but was copied
  /// by some clone before it became shared, or is such a copy.
  bool isCopiedShared(const Node *node) const {
    return copiedShared.contains(node);
  }
  /// Why every query is incomplete (an op whose address has no cell, a model
  /// write outside any function's graph), or null.
  const Reason *getGlobalUnknown() const { return globalUnknown; }

  /// The locations of an access of \p size bytes at \p cell (resolved) in
  /// its node and in each of its ancestors (the node first); with
  /// \p anywhere, of an access anywhere in the node.
  void
  locate(Cell cell, uint64_t size,
         llvm::SmallVectorImpl<std::pair<const Node *, WriteLocation>> &out,
         bool anywhere = false) const;
  /// The location of [offset, offset + size) of \p node (normalized).
  static WriteLocation makeLocation(const Node *node, int64_t offset,
                                    uint64_t size);

  size_t getNumEntries() const { return numEntries; }
  /// The ops (other than calls) indexed as writes of unknown value, by name.
  const llvm::StringMap<unsigned> &getUnknownWriteOps() const {
    return unknownWriteOps;
  }

private:
  void addWrite(WriteRecord write);
  void indexWrites();
  void indexInverses();
  void indexCallerWriters();

  const PointsToAnalysis &analysis;
  std::vector<WriteRecord> writes;
  llvm::DenseMap<const Node *, llvm::SmallVector<Entry, 2>> entries;
  llvm::DenseMap<const Node *, llvm::SmallVector<Inverse, 1>> inverses;
  llvm::DenseMap<const Node *, const Reason *> callerWriters;
  llvm::DenseSet<const Node *> copiedShared;
  const Reason *globalUnknown = nullptr;
  size_t numEntries = 0;
  llvm::StringMap<unsigned> unknownWriteOps;
};

/// The stored-value queries (see the file comment and StoredValues.cpp).
class StoredValueQuery {
public:
  explicit StoredValueQuery(const PointsToAnalysis &analysis,
                            unsigned maxQueryVisits = 4096);
  ~StoredValueQuery();

  /// What the \p size bytes at \p cell (a cell of the TU graph) may hold,
  /// read as \p type (null: constants keep the type they were written
  /// with, and a zero write gives no constant). \p at is reserved for flow
  /// sensitivity (T24) and ignored: the answer holds at every point.
  StoredValues valuesAt(Cell cell, uint64_t size, mlir::Operation *at,
                        mlir::Type type = {});

  /// valuesAt for \p cell, a cell of the graph of the callee of the last
  /// copy of \p context (copies from an outer graph down, outermost first),
  /// in that calling context: the cell is mapped out through the copies as
  /// far as its images are faithful, and the formals the writes store name
  /// the operands of those calls.
  StoredValues valuesInContext(Cell cell, uint64_t size, mlir::Type type,
                               llvm::ArrayRef<const TUClone *> context);

  const StoredValueIndex &getIndex() const { return *index; }
  /// The most cells one query may visit; past it, Reason Budget.
  unsigned getMaxQueryVisits() const { return maxQueryVisits; }
  void setMaxQueryVisits(unsigned max) { maxQueryVisits = max; }

  struct Stats {
    size_t numQueries = 0;
    size_t numVisits = 0;
    size_t numBudgetHits = 0;
  };
  const Stats &getStats() const { return stats; }

private:
  /// valuesAt without resetting the budget, for a cell of the innermost
  /// graph of \p chain (the copies from the read's graph down to the cell's;
  /// see StoredValues.cpp): memoized per cell, size and type for an empty
  /// chain.
  StoredValues cellValues(Cell cell, uint64_t size, mlir::Type type,
                          llvm::ArrayRef<const TUClone *> chain);
  /// Give \p out what \p write contributes to a read it relates to by
  /// \p relation (StoredValues.cpp), \p chain the copies the walk came
  /// through from the read's graph down to the write's.
  void applyWrite(const WriteRecord &write, int relation, int64_t delta,
                  uint64_t size, mlir::Type type,
                  llvm::ArrayRef<const TUClone *> chain, StoredValues &out);
  void resolveValue(mlir::Value value, uint64_t size, mlir::Type type,
                    llvm::ArrayRef<const TUClone *> chain, StoredValues &out);
  /// Map \p cell, a cell of the innermost graph of \p chain, out through the
  /// copies towards the outermost graph, as far as each step has a faithful
  /// image; return the length of the chain prefix left above the cells.
  size_t mapOut(Cell cell, llvm::ArrayRef<const TUClone *> chain,
                llvm::SmallVectorImpl<Cell> &out) const;
  /// Read \p cell of the innermost graph of \p chain as far out as it maps.
  void readOut(Cell cell, uint64_t size, mlir::Type type,
               llvm::ArrayRef<const TUClone *> chain, StoredValues &out);
  const Reason *reason(Reason::Kind kind, mlir::Operation *op,
                       llvm::StringRef detail);

  using Key =
      std::tuple<const Node *, int64_t, uint64_t, const void *, uint64_t>;
  const PointsToAnalysis &analysis;
  std::unique_ptr<StoredValueIndex> index;
  unsigned maxQueryVisits;
  unsigned visitsLeft = 0;
  bool overBudget = false;
  llvm::DenseMap<Key, StoredValues> memo;
  llvm::DenseSet<Key> inProgress;
  Stats stats;
};

} // namespace cir::pointsto

#endif // CLANG_CIR_DIALECT_ANALYSIS_POINTSTO_STOREDVALUES_H
