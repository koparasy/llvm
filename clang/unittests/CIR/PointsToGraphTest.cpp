//===- PointsToGraphTest.cpp - Unit tests for the points-to graph ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToGraph.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToAnalysis.h"
#include "clang/CIR/Dialect/Analysis/PointsTo/PointsToVerifier.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/Support/raw_ostream.h"

#include <gtest/gtest.h>

using namespace cir::pointsto;

namespace {

/// Assert that the graph invariants hold, printing the problems otherwise.
#define EXPECT_VERIFIED(g)                                                     \
  do {                                                                         \
    std::string errors;                                                        \
    llvm::raw_string_ostream os(errors);                                       \
    EXPECT_TRUE((g).verify(&os)) << errors;                                    \
  } while (false)

static Origin stackOrigin(uint32_t ctx = 0) {
  return Origin{Origin::Kind::Alloca, nullptr, {}, ctx};
}

static bool sameCell(Cell a, Cell b) { return a.resolve() == b.resolve(); }

static std::string printed(const Graph &g) {
  std::string out;
  llvm::raw_string_ostream os(out);
  g.print(os);
  return out;
}

//===----------------------------------------------------------------------===//
// 1. Unification shifts links and forwards cells.
//===----------------------------------------------------------------------===//

TEST(PointsToGraphTest, UnifyAtZero) {
  Graph g;
  Node *a = g.makeNode(stackOrigin(), NodeFlags::Stack);
  Node *b = g.makeNode(NodeFlags::Heap);
  Node *t1 = g.makeNode();
  Node *t2 = g.makeNode();
  g.setLink(Cell(a, 0), Cell(t1));
  g.setLink(Cell(b, 8), Cell(t2));
  EXPECT_VERIFIED(g);

  g.unify(Cell(a, 0), Cell(b, 0));
  EXPECT_VERIFIED(g);

  // Equal offsets: the older node survives.
  EXPECT_FALSE(a->isForwarded());
  EXPECT_TRUE(b->isForwarded());
  EXPECT_EQ(Cell(b, 8).getNode(), a);
  EXPECT_EQ(Cell(b, 8).getOffset(), 8);
  EXPECT_EQ(a->getFlags(), NodeFlags::Stack | NodeFlags::Heap);
  auto links = a->getLinks();
  ASSERT_EQ(links.size(), 2u);
  EXPECT_EQ(links[0].first, 0);
  EXPECT_EQ(links[0].second, Cell(t1, 0));
  EXPECT_EQ(links[1].first, 8);
  EXPECT_EQ(links[1].second, Cell(t2, 0));
  EXPECT_EQ(g.getNumLiveNodes(), 3u);
}

TEST(PointsToGraphTest, UnifyAtDelta) {
  Graph g;
  Node *a = g.makeNode();
  Node *b = g.makeNode();
  Node *ta = g.makeNode();
  Node *tb = g.makeNode();
  g.setLink(Cell(a, 0), Cell(ta));
  g.setLink(Cell(b, 0), Cell(tb, 4));
  g.growSize(b, 16);

  // (a, 16) == (b, 0): b is shifted by 16 into a.
  g.unify(Cell(a, 16), Cell(b, 0));
  EXPECT_VERIFIED(g);
  EXPECT_TRUE(b->isForwarded());
  EXPECT_TRUE(sameCell(Cell(b, 8), Cell(a, 24)));
  EXPECT_EQ(a->getSize(), 32u);
  auto links = a->getLinks();
  ASSERT_EQ(links.size(), 2u);
  EXPECT_EQ(links[1].first, 16);
  EXPECT_EQ(links[1].second, Cell(tb, 4));

  // The other way around: the node at the larger offset survives, so the
  // shift stays non-negative.
  Graph h;
  Node *c = h.makeNode();
  Node *d = h.makeNode();
  h.unify(Cell(c, 0), Cell(d, 16));
  EXPECT_VERIFIED(h);
  EXPECT_TRUE(c->isForwarded());
  EXPECT_EQ(Cell(c, 8).getNode(), d);
  EXPECT_EQ(Cell(c, 8).getOffset(), 24);

  // Unifying a cell with itself, or twice, is a no-op.
  h.unify(Cell(c, 0), Cell(d, 16));
  h.unify(Cell(d, 16), Cell(d, 16));
  EXPECT_VERIFIED(h);
  EXPECT_FALSE(d->isCollapsed());
}

TEST(PointsToGraphTest, GetLinkCreatesAndLookupReads) {
  Graph g;
  Node *a = g.makeNode();
  EXPECT_FALSE(g.lookupLink(Cell(a, 8)).has_value());
  Cell t = g.getLink(Cell(a, 8));
  EXPECT_VERIFIED(g);
  ASSERT_TRUE(t);
  EXPECT_EQ(t.getNode()->getFlags(), NodeFlags::None);
  EXPECT_TRUE(t.getNode()->getOrigins().empty());
  EXPECT_EQ(g.getLink(Cell(a, 8)), t);
  ASSERT_TRUE(g.lookupLink(Cell(a, 8)).has_value());
  EXPECT_EQ(*g.lookupLink(Cell(a, 8)), t);

  // setLink on an existing slot unifies the targets.
  Node *u = g.makeNode();
  g.setLink(Cell(a, 8), Cell(u, 0));
  EXPECT_VERIFIED(g);
  EXPECT_TRUE(sameCell(t, Cell(u, 0)));
}

//===----------------------------------------------------------------------===//
// 2. Deep graphs: no recursion.
//===----------------------------------------------------------------------===//

TEST(PointsToGraphTest, DeepCollidingChains) {
  constexpr unsigned depth = 100000;
  Graph g;
  std::vector<Node *> as, bs;
  for (unsigned i = 0; i <= depth; ++i) {
    as.push_back(g.makeNode());
    bs.push_back(g.makeNode());
  }
  for (unsigned i = 0; i < depth; ++i) {
    g.setLink(Cell(as[i], 0), Cell(as[i + 1]));
    g.setLink(Cell(bs[i], 0), Cell(bs[i + 1]));
  }
  // Unifying the heads makes their links collide, which unifies the next
  // level, and so on down the chain: a recursive implementation would need
  // `depth` frames.
  g.unify(Cell(as[0]), Cell(bs[0]));
  EXPECT_VERIFIED(g);
  for (unsigned i = 0; i <= depth; ++i)
    ASSERT_TRUE(sameCell(Cell(as[i]), Cell(bs[i]))) << "level " << i;
  EXPECT_EQ(g.getNumLiveNodes(), depth + 1);
}

TEST(PointsToGraphTest, LongForwardingChain) {
  // Each unify makes the new node the survivor, building a forwarding chain
  // as long as the loop; resolve() must walk it iteratively.
  constexpr unsigned depth = 100000;
  Graph g;
  Node *first = g.makeNode();
  Node *cur = first;
  for (unsigned i = 0; i < depth; ++i) {
    Node *next = g.makeNode();
    g.unify(Cell(cur, 0), Cell(next, 1));
    cur = next;
  }
  EXPECT_EQ(Cell(first, 0).getNode(), cur);
  EXPECT_EQ(Cell(first, 0).getOffset(), int64_t(depth));
  // After path compression every node forwards straight to the survivor.
  EXPECT_VERIFIED(g);
  EXPECT_EQ(Cell(first, 5).getOffset(), int64_t(depth) + 5);
  EXPECT_EQ(g.getNumLiveNodes(), 1u);
}

//===----------------------------------------------------------------------===//
// 3. and 4. Collapse on misalignment and on self-unification.
//===----------------------------------------------------------------------===//

TEST(PointsToGraphTest, MisalignedLinksCollapse) {
  Graph g;
  ReasonTable &rt = g.getReasonTable();
  const Reason *why = rt.make(Reason::Kind::CollapsedAt, nullptr, "test");
  Node *n = g.makeNode();
  Node *t1 = g.makeNode();
  Node *t2 = g.makeNode();
  g.setLink(Cell(n, 0), Cell(t1));
  EXPECT_FALSE(n->isCollapsed());
  g.setLink(Cell(n, 4), Cell(t2), why);
  EXPECT_VERIFIED(g);
  EXPECT_TRUE(n->isCollapsed());
  EXPECT_EQ(n->getReason(NodeFlags::Collapsed), why);
  ASSERT_EQ(n->getNumLinks(), 1u);
  EXPECT_TRUE(sameCell(Cell(t1), Cell(t2)));
  EXPECT_EQ(Cell(n, 12).getOffset(), 0);

  // Links exactly one pointer apart are fine, also with a smaller pointer.
  Graph h(/*pointerSize=*/4);
  Node *m = h.makeNode();
  h.getLink(Cell(m, 0));
  h.getLink(Cell(m, 4));
  EXPECT_VERIFIED(h);
  EXPECT_FALSE(m->isCollapsed());
  h.getLink(Cell(m, 6));
  EXPECT_VERIFIED(h);
  EXPECT_TRUE(m->isCollapsed());
}

TEST(PointsToGraphTest, MisalignedMergeCollapses) {
  Graph g;
  Node *a = g.makeNode();
  Node *b = g.makeNode();
  g.getLink(Cell(a, 0));
  g.getLink(Cell(b, 0));
  // b's link lands at offset 4 of a, overlapping a's link at 0.
  g.unify(Cell(a, 4), Cell(b, 0));
  EXPECT_VERIFIED(g);
  EXPECT_TRUE(Cell(a).getNode()->isCollapsed());
  EXPECT_EQ(Cell(a).getNode()->getNumLinks(), 1u);
}

TEST(PointsToGraphTest, SelfUnifyAtDeltaCollapses) {
  // `p` and `p+8` are the same cell: the object contains itself at a shift,
  // a recursive or array structure, so the node collapses.
  Graph g;
  Node *n = g.makeNode();
  g.getLink(Cell(n, 0));
  g.getLink(Cell(n, 8));
  g.unify(Cell(n, 0), Cell(n, 8));
  EXPECT_VERIFIED(g);
  EXPECT_TRUE(n->isCollapsed());
  EXPECT_EQ(n->getNumLinks(), 1u);

  // The same happens through a cycle: n.0 -> m+0, then m+0 == n+8.
  Graph h;
  Node *p = h.makeNode();
  Node *q = h.makeNode();
  h.setLink(Cell(p, 0), Cell(q, 0));
  h.unify(Cell(q, 0), Cell(p, 8));
  EXPECT_VERIFIED(h);
  EXPECT_FALSE(p->isCollapsed()); // p+8 == q+0 is just a shift so far.
  h.unify(Cell(p, 0), Cell(q, 0));
  EXPECT_VERIFIED(h);
  EXPECT_TRUE(Cell(p).getNode()->isCollapsed());
}

//===----------------------------------------------------------------------===//
// 5. Explicit collapse.
//===----------------------------------------------------------------------===//

TEST(PointsToGraphTest, CollapseMergesAllTargets) {
  Graph g;
  Node *n = g.makeNode();
  Node *t0 = g.makeNode(NodeFlags::Stack);
  Node *t1 = g.makeNode(NodeFlags::Heap);
  Node *t2 = g.makeNode(NodeFlags::Global);
  g.setLink(Cell(n, 0), Cell(t0));
  g.setLink(Cell(n, 8), Cell(t1));
  g.setLink(Cell(n, 16), Cell(t2, 8));
  g.markArray(n, 32);
  EXPECT_VERIFIED(g);

  const Reason *why = g.makeReason(Reason::Kind::CollapsedAt);
  g.collapse(n, why);
  EXPECT_VERIFIED(g);
  EXPECT_TRUE(n->isCollapsed());
  EXPECT_FALSE(n->isArray());
  EXPECT_EQ(n->getReason(NodeFlags::Collapsed), why);
  ASSERT_EQ(n->getNumLinks(), 1u);
  // t0+0, t1+0 and t2+8 are one cell now. t2 is at the larger offset and
  // survives the first merge; the shifted t0/t1 land on t2+8.
  Cell target = n->getLinks()[0].second;
  EXPECT_TRUE(sameCell(target, Cell(t0)));
  EXPECT_TRUE(sameCell(target, Cell(t1)));
  EXPECT_TRUE(sameCell(target, Cell(t2, 8)));
  EXPECT_EQ(target.getNode()->getFlags(),
            NodeFlags::Stack | NodeFlags::Heap | NodeFlags::Global);
  EXPECT_EQ(Cell(n, 24).getOffset(), 0);

  // Collapse is idempotent and keeps the first reason.
  g.collapse(n, g.makeReason(Reason::Kind::Budget));
  EXPECT_EQ(n->getReason(NodeFlags::Collapsed), why);
  EXPECT_VERIFIED(g);

  // addFlags(Collapsed) is a collapse too.
  Node *m = g.makeNode();
  g.getLink(Cell(m, 0));
  g.getLink(Cell(m, 8));
  g.addFlags(Cell(m), NodeFlags::Collapsed);
  EXPECT_VERIFIED(g);
  EXPECT_EQ(m->getNumLinks(), 1u);
}

TEST(PointsToGraphTest, CollapsedAbsorbsMergedNode) {
  Graph g;
  Node *c = g.makeNode(NodeFlags::Collapsed);
  Node *s = g.makeNode();
  g.getLink(Cell(s, 0));
  g.getLink(Cell(s, 8));
  g.unify(Cell(c, 0), Cell(s, 8));
  EXPECT_VERIFIED(g);
  Node *live = Cell(c).getNode();
  EXPECT_TRUE(live->isCollapsed());
  EXPECT_EQ(live->getNumLinks(), 1u);
}

//===----------------------------------------------------------------------===//
// 6. Arrays.
//===----------------------------------------------------------------------===//

TEST(PointsToGraphTest, ArrayNormalization) {
  Graph g;
  Node *n = g.makeNode();
  Node *t0 = g.makeNode();
  Node *t1 = g.makeNode();
  Node *t2 = g.makeNode();
  g.setLink(Cell(n, 0), Cell(t0));
  g.setLink(Cell(n, 16), Cell(t1));
  g.setLink(Cell(n, 24), Cell(t2));
  g.growSize(n, 32);
  g.markArray(n, 16);
  EXPECT_VERIFIED(g);
  EXPECT_FALSE(n->isCollapsed());
  EXPECT_EQ(n->getArrayElemSize(), std::optional<uint64_t>(16));
  EXPECT_EQ(n->getSize(), 32u); // The whole region, not the element.
  auto links = n->getLinks();
  ASSERT_EQ(links.size(), 2u);
  EXPECT_EQ(links[0].first, 0);
  EXPECT_EQ(links[1].first, 8);
  EXPECT_TRUE(sameCell(Cell(t0), Cell(t1)));
  EXPECT_TRUE(sameCell(Cell(t2), links[1].second));
  EXPECT_EQ(Cell(n, 40).getOffset(), 8);
  EXPECT_EQ(Cell(n, -8).getOffset(), 8);

  // A matching-stride self-unification is a no-op on an array.
  g.unify(Cell(n, 0), Cell(n, 16));
  EXPECT_VERIFIED(g);
  EXPECT_FALSE(n->isCollapsed());

  // markArray with the same size is a no-op, with another size collapses.
  g.markArray(n, 16);
  EXPECT_FALSE(n->isCollapsed());
  g.markArray(n, 24);
  EXPECT_VERIFIED(g);
  EXPECT_TRUE(n->isCollapsed());
  EXPECT_FALSE(n->isArray());
}

TEST(PointsToGraphTest, ArrayStrideMismatchCollapses) {
  Graph g;
  Node *a = g.makeNode();
  Node *b = g.makeNode();
  g.markArray(a, 16);
  g.markArray(b, 24);
  g.unify(Cell(a), Cell(b));
  EXPECT_VERIFIED(g);
  EXPECT_TRUE(Cell(a).getNode()->isCollapsed());
  EXPECT_EQ(Cell(a).getNode(), Cell(b).getNode());
}

TEST(PointsToGraphTest, ArrayWithNonArray) {
  // A non-array merged into an array is taken modulo the element size: the
  // links at 0 and 16 become one slot.
  Graph g;
  Node *arr = g.makeNode();
  g.markArray(arr, 16);
  Node *s = g.makeNode();
  Node *t0 = g.makeNode();
  Node *t1 = g.makeNode();
  g.setLink(Cell(s, 0), Cell(t0));
  g.setLink(Cell(s, 16), Cell(t1));
  g.unify(Cell(arr, 0), Cell(s, 0));
  EXPECT_VERIFIED(g);
  Node *live = Cell(arr).getNode();
  EXPECT_FALSE(live->isCollapsed());
  EXPECT_EQ(live->getArrayElemSize(), std::optional<uint64_t>(16));
  EXPECT_EQ(live->getNumLinks(), 1u);
  EXPECT_TRUE(sameCell(Cell(t0), Cell(t1)));

  // A non-array whose slots don't fit the stride collapses: a link at 12
  // runs into the next element's slot at 0 (16 + 0 - 12 < 8).
  Node *u = g.makeNode();
  g.getLink(Cell(u, 12));
  g.unify(Cell(arr, 0), Cell(u, 0));
  EXPECT_VERIFIED(g);
  EXPECT_TRUE(Cell(arr).getNode()->isCollapsed());

  // An element smaller than a pointer can't hold one.
  Node *bytes = g.makeNode();
  g.markArray(bytes, 4);
  EXPECT_FALSE(bytes->isCollapsed());
  g.getLink(Cell(bytes, 0));
  EXPECT_VERIFIED(g);
  EXPECT_TRUE(bytes->isCollapsed());

  // Element size 0 means unknown and collapses.
  Node *z = g.makeNode();
  g.markArray(z, 0);
  EXPECT_TRUE(z->isCollapsed());
  EXPECT_VERIFIED(g);
}

//===----------------------------------------------------------------------===//
// 7. Flags, origins, reasons.
//===----------------------------------------------------------------------===//

TEST(PointsToGraphTest, FlagsOriginsReasons) {
  Graph g;
  const Reason *rA = g.makeReason(Reason::Kind::EscapedTo, nullptr, "a");
  const Reason *rB = g.makeReason(Reason::Kind::EscapedTo, nullptr, "b");
  const Reason *rC = g.makeReason(Reason::Kind::UnknownCaller, nullptr, "c");

  Node *a = g.makeNode(stackOrigin(), NodeFlags::Stack);
  Node *b = g.makeNode(stackOrigin(), NodeFlags::Heap);
  g.addOrigin(Cell(b), stackOrigin(/*ctx=*/1));
  g.addOrigin(Cell(b), stackOrigin(/*ctx=*/1));
  EXPECT_EQ(b->getOrigins().size(), 2u);

  g.addFlags(Cell(a), NodeFlags::External, rA);
  g.addFlags(Cell(b), NodeFlags::External, rB);
  g.addFlags(Cell(b), NodeFlags::Incomplete, rC);
  // The first reason wins.
  g.addFlags(Cell(a), NodeFlags::External, rB);
  EXPECT_EQ(a->getReason(NodeFlags::External), rA);
  EXPECT_VERIFIED(g);

  g.unify(Cell(a), Cell(b));
  EXPECT_VERIFIED(g);
  EXPECT_EQ(a->getFlags(), NodeFlags::Stack | NodeFlags::Heap |
                               NodeFlags::External | NodeFlags::Incomplete);
  // Origins union without duplicates, survivor's first.
  ASSERT_EQ(a->getOrigins().size(), 2u);
  EXPECT_EQ(a->getOrigins()[0], stackOrigin());
  EXPECT_EQ(a->getOrigins()[1], stackOrigin(1));
  // The survivor's reason wins; the merged-in node fills in missing bits.
  EXPECT_EQ(a->getReason(NodeFlags::External), rA);
  EXPECT_EQ(a->getReason(NodeFlags::Incomplete), rC);
  EXPECT_EQ(a->getReason(NodeFlags::Summary), nullptr);

  // When the shift makes the younger node the survivor, its reason wins.
  Node *x = g.makeNode();
  Node *y = g.makeNode();
  g.addFlags(Cell(x), NodeFlags::Mod, rA);
  g.addFlags(Cell(y), NodeFlags::Mod, rB);
  g.unify(Cell(x, 0), Cell(y, 8));
  EXPECT_VERIFIED(g);
  EXPECT_EQ(Cell(x).getNode(), y);
  EXPECT_EQ(y->getReason(NodeFlags::Mod), rB);

  // Reasons own their detail string and chain to their cause.
  std::string detail = "temporary";
  const Reason *chained =
      g.makeReason(Reason::Kind::MergedAt, nullptr, detail, rA);
  detail = "overwritten";
  EXPECT_EQ(chained->detail, "temporary");
  EXPECT_EQ(chained->cause, rA);

  EXPECT_EQ(toString(NodeFlags::None), "");
  EXPECT_EQ(toString(NodeFlags::Heap | NodeFlags::Summary | NodeFlags::HasUSM),
            "Heap|Summary|HasUSM");
}

//===----------------------------------------------------------------------===//
// 8. Cloning.
//===----------------------------------------------------------------------===//

TEST(PointsToGraphTest, CloneReachableWithCycles) {
  Graph src;
  const Reason *cause = src.makeReason(Reason::Kind::UnknownCaller);
  const Reason *why =
      src.makeReason(Reason::Kind::EscapedTo, nullptr, "callee", cause);
  Node *r = src.makeNode(stackOrigin(3), NodeFlags::Stack);
  Node *a = src.makeNode(NodeFlags::Heap);
  Node *b = src.makeNode(NodeFlags::Heap);
  Node *unrelated = src.makeNode(NodeFlags::Global);
  src.setLink(Cell(r, 0), Cell(a, 8));
  src.setLink(Cell(a, 0), Cell(b));
  src.setLink(Cell(b, 8), Cell(a, 0)); // cycle a -> b -> a
  src.getLink(Cell(unrelated, 0));
  src.addFlags(Cell(a), NodeFlags::External | NodeFlags::Mod, why);
  src.markArray(b, 16);
  EXPECT_VERIFIED(src);

  Graph dst;
  Node *existing = dst.makeNode();
  (void)existing;
  CloneMap map =
      src.cloneInto(dst, {Cell(r)}, NodeFlags::All & ~NodeFlags::Mod);
  EXPECT_VERIFIED(dst);
  EXPECT_EQ(map.size(), 3u);
  EXPECT_EQ(dst.getNumLiveNodes(), 4u);
  EXPECT_FALSE(map.lookup(unrelated));

  Cell r2 = map.lookup(r);
  Cell a2 = map.lookup(a);
  Cell b2 = map.lookup(b);
  ASSERT_TRUE(r2 && a2 && b2);
  EXPECT_NE(a2.getNode(), b2.getNode());
  // Links and the cycle are reproduced.
  EXPECT_TRUE(
      sameCell(*dst.lookupLink(Cell(r2.getNode(), 0)), Cell(a2.getNode(), 8)));
  EXPECT_TRUE(sameCell(*dst.lookupLink(Cell(a2.getNode(), 0)), b2));
  EXPECT_TRUE(sameCell(*dst.lookupLink(Cell(b2.getNode(), 8)), a2));
  // Cells map with their offset; arrays keep their element size.
  EXPECT_TRUE(sameCell(map.map(Cell(a, 8)), Cell(a2.getNode(), 8)));
  EXPECT_EQ(map.map(Cell(b, 24)).getOffset(), 8);
  EXPECT_EQ(b2.getNode()->getArrayElemSize(), std::optional<uint64_t>(16));
  // Origins keep their ctx; flags are masked.
  ASSERT_EQ(r2.getNode()->getOrigins().size(), 1u);
  EXPECT_EQ(r2.getNode()->getOrigins()[0].ctx, 3u);
  EXPECT_EQ(a2.getNode()->getFlags(), NodeFlags::Heap | NodeFlags::External);
  EXPECT_EQ(a2.getNode()->getReason(NodeFlags::Mod), nullptr);
  // Reasons were copied into dst's own table, with their cause chain.
  const Reason *why2 = a2.getNode()->getReason(NodeFlags::External);
  ASSERT_NE(why2, nullptr);
  EXPECT_NE(why2, why);
  EXPECT_EQ(why2->kind, Reason::Kind::EscapedTo);
  EXPECT_EQ(why2->detail, "callee");
  ASSERT_NE(why2->cause, nullptr);
  EXPECT_NE(why2->cause, cause);
  EXPECT_EQ(why2->cause->kind, Reason::Kind::UnknownCaller);

  // With a shared table the reason pointers are reused.
  Graph shared(8, &src.getReasonTable());
  CloneMap map2 = src.cloneInto(shared, {Cell(a)});
  EXPECT_VERIFIED(shared);
  EXPECT_EQ(map2.size(), 2u);
  EXPECT_EQ(map2.lookup(a).getNode()->getReason(NodeFlags::External), why);
  EXPECT_EQ(map2.lookup(a).getNode()->getFlags(),
            NodeFlags::Heap | NodeFlags::External | NodeFlags::Mod);

  // Cloned nodes are independent of the source.
  dst.unify(a2, b2);
  EXPECT_VERIFIED(dst);
  EXPECT_NE(Cell(a).getNode(), Cell(b).getNode());
  EXPECT_VERIFIED(src);
}

//===----------------------------------------------------------------------===//
// markReachable
//===----------------------------------------------------------------------===//

TEST(PointsToGraphTest, MarkReachableFollowsLinksAndCycles) {
  Graph g;
  Node *root = g.makeNode(stackOrigin(), NodeFlags::Stack);
  Node *mid = g.makeNode(NodeFlags::Heap);
  Node *leaf = g.makeNode(NodeFlags::Heap);
  Node *other = g.makeNode(NodeFlags::Heap);
  // root+8 -> mid, mid -> leaf, leaf -> root (a cycle); other is unreachable
  // but points into the cycle.
  g.setLink(Cell(root, 8), Cell(mid));
  g.setLink(Cell(mid, 0), Cell(leaf));
  g.setLink(Cell(leaf, 0), Cell(root));
  g.setLink(Cell(other, 0), Cell(mid));

  const Reason *why = g.makeReason(Reason::Kind::EscapedTo, nullptr, "@f");
  // From an interior cell: the whole node's links count.
  EXPECT_EQ(g.markReachable(Cell(root, 16), NodeFlags::External, why), 3u);
  EXPECT_VERIFIED(g);
  for (Node *n : {root, mid, leaf})
    EXPECT_TRUE(n->hasFlags(NodeFlags::External));
  EXPECT_FALSE(other->hasFlags(NodeFlags::External));

  EXPECT_EQ(root->getReason(NodeFlags::External), why);
  const Reason *derived = mid->getReason(NodeFlags::External);
  ASSERT_NE(derived, nullptr);
  EXPECT_EQ(derived->kind, Reason::Kind::EscapedTo);
  EXPECT_EQ(derived->cause, why);
  EXPECT_EQ(leaf->getReason(NodeFlags::External), derived);

  // Marking again keeps the first reasons.
  const Reason *later = g.makeReason(Reason::Kind::EscapedTo, nullptr, "@g");
  g.markReachable(Cell(leaf), NodeFlags::External | NodeFlags::Mod, later);
  EXPECT_EQ(root->getReason(NodeFlags::External), why);
  EXPECT_TRUE(root->hasFlags(NodeFlags::Mod));
  EXPECT_EQ(g.markReachable(Cell(), NodeFlags::External, why), 0u);
}

TEST(PointsToGraphTest, MarkReachableStopsAtFlaggedNodes) {
  Graph g;
  Node *a = g.makeNode(NodeFlags::Heap);
  Node *b = g.makeNode(NodeFlags::Heap);
  Node *c = g.makeNode(NodeFlags::Heap);
  Node *d = g.makeNode(NodeFlags::Heap);
  // a -> b -> c -> d; b already Incomplete.
  g.setLink(Cell(a), Cell(b));
  g.setLink(Cell(b), Cell(c));
  g.setLink(Cell(c), Cell(d));
  const Reason *first = g.makeReason(Reason::Kind::UnknownCaller, nullptr, "b");
  g.addFlags(Cell(b), NodeFlags::Incomplete, first);
  const Reason *why = g.makeReason(Reason::Kind::UnknownCaller, nullptr, "a");
  g.addFlags(Cell(a), NodeFlags::Incomplete, why);

  // From a: the walk stops at b, which has the flag already.
  g.markReachable(Cell(a), NodeFlags::Incomplete, why, /*stopAtFlagged=*/true);
  EXPECT_FALSE(c->hasFlags(NodeFlags::Incomplete));
  EXPECT_EQ(b->getReason(NodeFlags::Incomplete), first);
  // From b: its own call reaches c and d, with a reason derived from b's.
  g.markReachable(Cell(b), NodeFlags::Incomplete,
                  b->getReason(NodeFlags::Incomplete), /*stopAtFlagged=*/true);
  for (Node *n : {c, d})
    EXPECT_TRUE(n->hasFlags(NodeFlags::Incomplete));
  const Reason *derived = d->getReason(NodeFlags::Incomplete);
  ASSERT_NE(derived, nullptr);
  EXPECT_EQ(derived->cause, first);
  EXPECT_EQ(derived->detail, "reachable from an incomplete node");
}

//===----------------------------------------------------------------------===//
// 9. Printing.
//===----------------------------------------------------------------------===//

TEST(PointsToGraphTest, PrintFormat) {
  mlir::MLIRContext context;
  context.loadDialect<cir::CIRDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    !s32i = !cir.int<s, 32>
    module {
      cir.func @f() {
        %x = cir.alloca "x" align(4) : !cir.ptr<!s32i> loc("some/dir/t.cpp":3:5)
        cir.return
      }
    }
  )mlir",
                                              &context);
  ASSERT_TRUE(module);
  mlir::Operation *alloca = nullptr;
  module->walk([&](cir::AllocaOp op) { alloca = op; });
  ASSERT_NE(alloca, nullptr);
  auto sym = mlir::FlatSymbolRefAttr::get(&context, "g");

  Graph g;
  Node *x =
      g.makeNode(Origin{Origin::Kind::Alloca, alloca, {}, 0}, NodeFlags::Stack);
  g.growSize(x, 16);
  Node *h = g.makeNode(Origin{Origin::Kind::HeapCall, nullptr, {}, 2},
                       NodeFlags::Heap | NodeFlags::Summary);
  Node *gl = g.makeNode(Origin{Origin::Kind::GlobalSym, nullptr, sym, 0},
                        NodeFlags::Global);
  g.growSize(gl, 32);
  g.setLink(Cell(x, 0), Cell(h, 0));
  g.setLink(Cell(x, 8), Cell(gl, 16));
  g.markArray(gl, 8);
  const Reason *cause = g.makeReason(Reason::Kind::UnknownCaller);
  g.addFlags(Cell(gl), NodeFlags::External,
             g.makeReason(Reason::Kind::EscapedTo, alloca, "a \"b\"", cause));
  // A merged-away node does not print; its origin moves to the survivor.
  Node *m = g.makeNode(Origin{Origin::Kind::FuncSym, nullptr, sym, 0},
                       NodeFlags::Function);
  g.unify(Cell(h), Cell(m));
  EXPECT_VERIFIED(g);

  EXPECT_EQ(printed(g),
            "n0 [Stack] size=16 origins={stack(cir.alloca 'x' t.cpp:3:5)} "
            "links={0->n1+0, 8->n2+0}\n"
            "n1 [Heap|Function|Summary] size=0 origins={heap(?)#2, "
            "func(@g)} links={}\n"
            "n2 [Global|External] size=32 arr=8 origins={global(@g)} "
            "links={} reasons={External: EscapedTo(cir.alloca 'x' "
            "t.cpp:3:5, \"a \\22b\\22\") <- UnknownCaller(?)}\n");

  std::string cell;
  llvm::raw_string_ostream os(cell);
  Graph::printCell(os, Cell(m, 4));
  os << ' ';
  Graph::printCell(os, Cell());
  EXPECT_EQ(cell, "n1+4 null");
}

TEST(PointsToGraphTest, PrintNegativeOffset) {
  Graph g;
  Node *a = g.makeNode();
  Node *t = g.makeNode();
  g.setLink(Cell(a, -8), Cell(t, -16));
  EXPECT_VERIFIED(g);
  EXPECT_EQ(printed(g), "n0 [] size=0 origins={} links={-8->n1-16}\n"
                        "n1 [] size=0 origins={} links={}\n");
}

//===----------------------------------------------------------------------===//
// 10. Explanations.
//===----------------------------------------------------------------------===//

TEST(PointsToGraphTest, ExplainReasons) {
  mlir::MLIRContext context;
  context.loadDialect<cir::CIRDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    !s32i = !cir.int<s, 32>
    module {
      cir.func private @foo(!cir.ptr<!s32i>)
      cir.func @f(%p: !cir.ptr<!s32i>) {
        %x = cir.alloca "x" align(4) : !cir.ptr<!s32i> loc("dir/t.cpp":3:5)
        cir.call @foo(%x) : (!cir.ptr<!s32i>) -> () loc("dir/t.cpp":12:3)
        %y = cir.alloca "y" align(4) : !cir.ptr<!s32i> loc(unknown)
        cir.return
      } loc("dir/t.cpp":1:1)
    }
  )mlir",
                                              &context);
  ASSERT_TRUE(module);
  mlir::Operation *call = nullptr, *func = nullptr, *noLoc = nullptr;
  module->walk([&](cir::CallOp op) { call = op; });
  module->walk([&](cir::FuncOp op) {
    if (!op.isDeclaration())
      func = op;
  });
  module->walk([&](cir::AllocaOp op) {
    if (op.getName() == "y")
      noLoc = op;
  });
  ASSERT_TRUE(call && func && noLoc);

  Graph g;
  using K = Reason::Kind;
  auto text = [&](K kind, mlir::Operation *op, llvm::StringRef detail) {
    return explain(g.makeReason(kind, op, detail));
  };
  EXPECT_EQ(explain(nullptr), "no reason was recorded");

  // EscapedTo, for each detail the analysis makes.
  EXPECT_EQ(text(K::EscapedTo, call, "@foo"),
            "it is an escaped pointer passed to @foo (t.cpp:12)");
  EXPECT_EQ(text(K::EscapedTo, call, "@foo not analysed"),
            "it is an escaped pointer passed to @foo, whose body is not "
            "analysed (t.cpp:12)");
  EXPECT_EQ(text(K::EscapedTo, call, "@foo may access the global"),
            "it is an escaped pointer stored in a global that @foo may access "
            "(t.cpp:12)");
  EXPECT_EQ(
      text(K::EscapedTo, call, "an indirect callee may access the global"),
      "it is an escaped pointer stored in a global that an unresolved "
      "indirect callee may access (t.cpp:12)");
  EXPECT_EQ(text(K::EscapedTo, call, "global visible to other TUs"),
            "it is an escaped pointer stored in a global visible to other "
            "translation units (t.cpp:12)");
  EXPECT_EQ(text(K::EscapedTo, call, "variadic argument"),
            "it is an escaped pointer passed as a variadic argument "
            "(t.cpp:12)");
  EXPECT_EQ(text(K::EscapedTo, call, "address escaped as an integer"),
            "it is an escaped pointer converted to an integer (t.cpp:12)");
  EXPECT_EQ(text(K::EscapedTo, call, "result of strdup"),
            "it is an escaped pointer returned by strdup (t.cpp:12)");
  EXPECT_EQ(text(K::EscapedTo, call, "something new"),
            "it is an escaped pointer (\"something new\") (t.cpp:12)");

  // UnknownCaller names the function.
  EXPECT_EQ(text(K::UnknownCaller, func, "formal #0"),
            "it is parameter #0 of @f, which may have callers that are not "
            "analysed (t.cpp:1)");
  EXPECT_EQ(text(K::UnknownCaller, func, "returned to an unknown caller"),
            "it is the return value of @f, which may have callers that are "
            "not analysed (t.cpp:1)");

  // The other kinds.
  EXPECT_EQ(text(K::IndirectUnresolved, call, "callee node is External"),
            "it is a pointer passed to or returned from an indirect call "
            "whose targets are not known (\"callee node is External\") "
            "(t.cpp:12)");
  EXPECT_EQ(text(K::Budget, call, "indirect-call rounds exhausted"),
            "it is a pointer the analysis stopped tracking when it ran out of "
            "budget (\"indirect-call rounds exhausted\") (t.cpp:12)");
  EXPECT_EQ(text(K::UnmodelledRole, call, "submit"),
            "it is a pointer passed to a runtime call that has no model "
            "(\"submit\") (t.cpp:12)");
  EXPECT_EQ(text(K::UnhandledOp, call, "result without a rule"),
            "it is a pointer through `cir.call`, which the analysis does not "
            "model (\"result without a rule\") (t.cpp:12)");
  EXPECT_EQ(text(K::UnhandledOp, nullptr, "value not analysed"),
            "it is a value the analysis did not see (\"value not analysed\")");
  EXPECT_EQ(text(K::CollapsedAt, call, ""),
            "it is an object made field-insensitive at `cir.call` (t.cpp:12)");
  EXPECT_EQ(text(K::MergedAt, call, "@foo"),
            "it is an object merged with others by binding a call to @foo "
            "(t.cpp:12)");
  EXPECT_EQ(text(K::SummaryObject, call, "loop"),
            "it is an object allocated in a loop (t.cpp:12)");
  EXPECT_EQ(text(K::SummaryObject, call, "recursion"),
            "it is an object of a recursive function (t.cpp:12)");
  EXPECT_EQ(text(K::SummaryObject, call, "several callers"),
            "it is an object allocated in a function that may run several "
            "times (t.cpp:12)");
  EXPECT_EQ(text(K::SummaryObject, call, "merged origins"),
            "it is an object with several origins (t.cpp:12)");

  // An op without a file location is placed by its function.
  EXPECT_EQ(text(K::EscapedTo, noLoc, "@foo"),
            "it is an escaped pointer passed to @foo (in @f)");

  // A derived reason reads "reachable from" its root, through any number of
  // derivations.
  Node *a = g.makeNode(NodeFlags::Stack);
  Node *b = g.makeNode(NodeFlags::Stack);
  Node *c = g.makeNode(NodeFlags::Stack);
  g.setLink(Cell(a), Cell(b));
  g.setLink(Cell(b), Cell(c));
  const Reason *root = g.makeReason(K::EscapedTo, call, "@foo");
  g.markReachable(Cell(a), NodeFlags::External, root);
  const Reason *derived = b->getReason(NodeFlags::External);
  EXPECT_TRUE(isDerivedReason(derived));
  EXPECT_FALSE(isDerivedReason(root));
  EXPECT_EQ(getRootReason(derived), root);
  const Reason *twice = g.makeReason(
      K::EscapedTo, call, "reachable from an escaped pointer", derived);
  EXPECT_EQ(getRootReason(twice), root);
  std::string expected = "it is reachable from an escaped pointer passed to "
                         "@foo (t.cpp:12)";
  EXPECT_EQ(explain(derived), expected);
  EXPECT_EQ(explain(twice), expected);

  // The chain, one entry per reason.
  llvm::SmallVector<std::string, 4> chain = explainChain(derived);
  ASSERT_EQ(chain.size(), 2u);
  EXPECT_EQ(chain[0], "EscapedTo(cir.call t.cpp:12:3, \"reachable from an "
                      "escaped pointer\")");
  EXPECT_EQ(chain[1], "EscapedTo(cir.call t.cpp:12:3, \"@foo\")");
  EXPECT_TRUE(explainChain(nullptr).empty());
}

//===----------------------------------------------------------------------===//
// 11. The TU self-check catches a broken invariant.
//===----------------------------------------------------------------------===//

TEST(PointsToGraphTest, VerifyAnalysisInvariants) {
  mlir::MLIRContext context;
  context.loadDialect<cir::CIRDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    !s32i = !cir.int<s, 32>
    module {
      cir.func @f() {
        %x = cir.alloca "x" align(4) : !cir.ptr<!s32i>
        %pp = cir.alloca "pp" align(8) : !cir.ptr<!cir.ptr<!s32i>>
        cir.store %x, %pp : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
        cir.return
      }
    }
  )mlir",
                                              &context);
  ASSERT_TRUE(module);
  mlir::Value x, pp;
  module->walk([&](cir::AllocaOp op) {
    (op.getName() == "x" ? x : pp) = op.getResult();
  });
  std::string messages;
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diag) {
    messages += diag.str() + "\n";
    return mlir::success();
  });

  {
    PointsToAnalysis analysis(*module);
    EXPECT_TRUE(mlir::succeeded(verifyPointsToAnalysis(analysis)));
    EXPECT_TRUE(analysis.pointsTo(x).complete);
    // What pp's slot holds is x's object.
    ObjectSet held = analysis.objectsAt(analysis.cellOf(pp));
    ASSERT_EQ(held.cells.size(), 1u);
    EXPECT_EQ(held.cells[0], analysis.cellOf(x).resolve());
    EXPECT_TRUE(held.complete);
    EXPECT_EQ(explain(held), "pointer refers to known memory");
  }
  EXPECT_EQ(messages, "");

  // External without propagation to the link target.
  {
    PointsToAnalysis analysis(*module);
    Graph &g = analysis.getGraph();
    g.addFlags(analysis.cellOf(pp), NodeFlags::External,
               g.makeReason(Reason::Kind::EscapedTo, nullptr, "@foo"));
    EXPECT_FALSE(analysis.pointsTo(pp).complete);
    EXPECT_TRUE(analysis.pointsTo(x).complete);
    EXPECT_TRUE(mlir::failed(verifyPointsToAnalysis(analysis)));
    EXPECT_NE(messages.find("is External but"), std::string::npos) << messages;
  }
  messages.clear();

  // Summary without a SummaryObject reason.
  {
    PointsToAnalysis analysis(*module);
    analysis.getGraph().addFlags(analysis.cellOf(x), NodeFlags::Summary);
    EXPECT_TRUE(mlir::failed(verifyPointsToAnalysis(analysis)));
    EXPECT_NE(messages.find("Summary without a SummaryObject reason"),
              std::string::npos)
        << messages;
  }
}

//===----------------------------------------------------------------------===//
// 12. Named ghost links.
//===----------------------------------------------------------------------===//

TEST(PointsToGraphTest, GhostFieldsAreInterned) {
  GhostField a = GhostField::get("$test.a");
  GhostField b = GhostField::get("$test.b");
  EXPECT_TRUE(a);
  EXPECT_FALSE(GhostField());
  EXPECT_NE(a, b);
  EXPECT_EQ(a, GhostField::get("$test.a"));
  EXPECT_EQ(a.getName(), "$test.a");
  EXPECT_EQ(GhostField().getName(), "");
}

TEST(PointsToGraphTest, GhostLinksMergeOnUnify) {
  Graph g;
  GhostField storage = GhostField::get("$storage");
  GhostField host = GhostField::get("$host");
  Node *a = g.makeNode(stackOrigin(), NodeFlags::Stack);
  Node *b = g.makeNode(stackOrigin(1), NodeFlags::Stack);
  Node *s1 = g.makeNode(NodeFlags::Ghost);
  Node *s2 = g.makeNode(NodeFlags::Ghost);
  Node *h = g.makeNode();
  g.setGhostLink(Cell(a, 16), storage, Cell(s1));
  g.setGhostLink(Cell(b, 0), storage, Cell(s2));
  g.setGhostLink(Cell(b, 0), host, Cell(h));
  // Different offsets and fields stay apart.
  EXPECT_FALSE(g.lookupGhostLink(Cell(a, 0), storage));
  EXPECT_FALSE(g.lookupGhostLink(Cell(a, 16), host));
  EXPECT_VERIFIED(g);

  // b's object sits at a+16: its $storage collides with a's and unifies.
  g.unify(Cell(a, 16), Cell(b, 0));
  EXPECT_TRUE(sameCell(Cell(s1), Cell(s2)));
  EXPECT_TRUE(sameCell(*g.lookupGhostLink(Cell(a, 16), host), Cell(h)));
  EXPECT_EQ(Cell(a).getNode()->getNumGhostLinks(), 2u);
  EXPECT_EQ(Cell(a).getNode()->getNumLinks(), 0u);
  EXPECT_VERIFIED(g);

  // getGhostLink creates on demand, then reads the same target.
  Cell made = g.getGhostLink(Cell(a, 32), storage);
  EXPECT_TRUE(sameCell(made, g.getGhostLink(Cell(a, 32), storage)));
  EXPECT_FALSE(sameCell(made, Cell(s1)));
  // setGhostLink on an existing ghost unifies the targets.
  Node *s3 = g.makeNode(NodeFlags::Ghost);
  g.setGhostLink(Cell(a, 32), storage, Cell(s3));
  EXPECT_TRUE(sameCell(made, Cell(s3)));
  EXPECT_VERIFIED(g);
}

TEST(PointsToGraphTest, GhostLinksIndependentOfLinkShape) {
  Graph g;
  GhostField storage = GhostField::get("$storage");
  GhostField host = GhostField::get("$host");

  // A ghost at a pointer slot's offset neither collides with the link nor
  // makes the node misaligned.
  Node *n = g.makeNode(stackOrigin(), NodeFlags::Stack);
  Node *t = g.makeNode();
  Node *s = g.makeNode(NodeFlags::Ghost);
  g.setLink(Cell(n, 0), Cell(t));
  g.setGhostLink(Cell(n, 0), storage, Cell(s));
  g.setGhostLink(Cell(n, 4), host, Cell(g.makeNode()));
  EXPECT_FALSE(Cell(n).getNode()->isCollapsed());
  EXPECT_TRUE(sameCell(*g.lookupLink(Cell(n, 0)), Cell(t)));
  EXPECT_FALSE(sameCell(Cell(t), Cell(s)));
  EXPECT_VERIFIED(g);

  // An array of handles: ghosts are taken modulo the element size, so the
  // elements share one ghost per field, apart from the real links.
  Node *arr = g.makeNode(stackOrigin(1), NodeFlags::Stack);
  Node *e0 = g.makeNode(NodeFlags::Ghost);
  Node *e1 = g.makeNode(NodeFlags::Ghost);
  Node *p = g.makeNode();
  g.setGhostLink(Cell(arr, 0), storage, Cell(e0));
  g.setGhostLink(Cell(arr, 16), storage, Cell(e1));
  g.setLink(Cell(arr, 8), Cell(p));
  EXPECT_FALSE(sameCell(Cell(e0), Cell(e1)));
  g.markArray(arr, 16);
  EXPECT_TRUE(sameCell(Cell(e0), Cell(e1)));
  EXPECT_FALSE(sameCell(Cell(e0), Cell(p)));
  EXPECT_TRUE(sameCell(*g.lookupGhostLink(Cell(arr, 32), storage), Cell(e0)));
  EXPECT_FALSE(Cell(arr).getNode()->isCollapsed());
  EXPECT_VERIFIED(g);

  // Collapse moves every ghost to offset 0: one per field, real links apart.
  Node *c = g.makeNode(stackOrigin(2), NodeFlags::Stack);
  Node *c0 = g.makeNode(NodeFlags::Ghost);
  Node *c1 = g.makeNode(NodeFlags::Ghost);
  Node *ch = g.makeNode();
  Node *cl = g.makeNode();
  g.setGhostLink(Cell(c, 0), storage, Cell(c0));
  g.setGhostLink(Cell(c, 24), storage, Cell(c1));
  g.setGhostLink(Cell(c, 24), host, Cell(ch));
  g.setLink(Cell(c, 8), Cell(cl));
  g.collapse(c);
  EXPECT_TRUE(sameCell(Cell(c0), Cell(c1)));
  EXPECT_FALSE(sameCell(Cell(c0), Cell(ch)));
  EXPECT_FALSE(sameCell(Cell(c0), Cell(cl)));
  EXPECT_FALSE(sameCell(Cell(ch), Cell(cl)));
  EXPECT_EQ(Cell(c).getNode()->getNumGhostLinks(), 2u);
  EXPECT_TRUE(sameCell(*g.lookupGhostLink(Cell(c, 40), host), Cell(ch)));
  EXPECT_VERIFIED(g);
}

TEST(PointsToGraphTest, GhostLinksCloneAndPropagate) {
  Graph g;
  GhostField storage = GhostField::get("$storage");
  GhostField host = GhostField::get("$host");
  Node *buf = g.makeNode(stackOrigin(), NodeFlags::Stack);
  Node *st = g.makeNode(Origin{Origin::Kind::GhostSite}, NodeFlags::Ghost);
  Node *mem = g.makeNode(stackOrigin(1), NodeFlags::Stack);
  g.setGhostLink(Cell(buf, 8), storage, Cell(st));
  g.setGhostLink(Cell(st), host, Cell(mem));
  // A cycle through a ghost link.
  g.setLink(Cell(mem), Cell(buf));

  Graph dst;
  CloneMap map = g.cloneInto(dst, {Cell(buf)});
  EXPECT_EQ(map.size(), 3u);
  Cell cbuf = map.lookup(buf);
  std::optional<Cell> cst =
      dst.lookupGhostLink(Cell(cbuf.getNode(), 8), storage);
  ASSERT_TRUE(cst);
  EXPECT_TRUE(sameCell(*cst, map.lookup(st)));
  std::optional<Cell> cmem = dst.lookupGhostLink(*cst, host);
  ASSERT_TRUE(cmem);
  EXPECT_TRUE(sameCell(*cmem, map.lookup(mem)));
  EXPECT_TRUE(sameCell(*dst.lookupLink(*cmem), cbuf));
  EXPECT_VERIFIED(dst);

  // Escape follows ghost links: what a ghost refers to is unknown once the
  // handle escapes.
  EXPECT_EQ(g.markReachable(Cell(buf), NodeFlags::External), 3u);
  EXPECT_TRUE(Cell(st).getNode()->hasFlags(NodeFlags::External));
  EXPECT_TRUE(Cell(mem).getNode()->hasFlags(NodeFlags::External));

  // Printing.
  Graph p;
  Node *h = p.makeNode(NodeFlags::Stack);
  Node *s = p.makeNode(NodeFlags::Ghost);
  p.setGhostLink(Cell(h, 16), storage, Cell(s));
  EXPECT_EQ(printed(p), "n0 [Stack] size=0 origins={} links={} "
                        "ghost={16:$storage->n1+0}\n"
                        "n1 [Ghost] size=0 origins={} links={}\n");
}

} // namespace
