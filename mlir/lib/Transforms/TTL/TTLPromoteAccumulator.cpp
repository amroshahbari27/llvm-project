//===- TTLPromoteAccumulator.cpp - Reduction variable promotion ------------===//
//
// Promotes loop-carried reduction store-load chains to iter_args.
//
// Problem statement
// -----------------
// After affine loop tiling, the innermost reduction loop retains a
// load-reduce-store pattern on the accumulation buffer:
//
//   affine.for %k = ...                         <- reduction loop
//     %a = affine.load %A[%i, %k]
//     %b = affine.load %B[%k, %j]
//     %p = arith.muli %a, %b
//     %c = affine.load %C[%i, %j]               <- loop-invariant address
//     %s = arith.addi %c, %p
//     affine.store %s, %C[%i, %j]               <- loop-invariant address
//
// Each iteration reads the value written by the previous iteration.
// This is a reduction that can be promoted to a scalar (iter_arg):
//
//   %init = affine.load %C[%i, %j]
//   %res  = affine.for %k = ... iter_args(%acc = %init) -> (i32) {
//     %a = affine.load %A[%i, %k]
//     %b = affine.load %B[%k, %j]
//     %p = arith.muli %a, %b
//     %s = arith.addi %acc, %p
//     affine.yield %s
//   }
//   affine.store %res, %C[%i, %j]
//
// Soundness
// ---------
// The transformation is sound when:
//   (a) The load/store address does not depend on the loop IV.
//   (b) The stored value feeds back through a commutative reduction op
//       (addi, addf, muli, mulf) from the loaded value.
//   (c) No other store in the loop body targets the SAME memref.
//       This conservative check prevents aliasing hazards. A more precise
//       analysis would use affine dependence, but the conservative rule
//       covers the standard reduction patterns (matmul, convolution, scan).
//
// Pipeline position
// -----------------
// After ttl-tile and before ttl-copy-generate.  At this point the loop
// nest uses the original global memrefs with clean affine access maps.
// After promotion, the copy-generate pass will DMA the output buffer at
// the outer loop level, and the inner compute loop will accumulate into
// a scalar register instead of through repeated memory traffic.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::affine;

namespace {

// ─── Analysis helpers ─────────────────────────────────────────────────────

/// True if any operand of `op` is the induction variable of `forOp`.
static bool usesInductionVar(AffineForOp forOp, Operation *op) {
  Value iv = forOp.getInductionVar();
  return llvm::any_of(op->getOperands(), [iv](Value v) { return v == iv; });
}

/// True if `load` and `store` access the same memref via identical affine
/// maps with identical map operands.
static bool sameAccess(AffineReadOpInterface load,
                       AffineWriteOpInterface store) {
  if (load.getMemRef() != store.getMemRef())
    return false;
  if (load.getAffineMap() != store.getAffineMap())
    return false;
  auto li = load.getMapOperands();
  auto si = store.getMapOperands();
  if (li.size() != si.size())
    return false;
  for (unsigned i = 0; i < li.size(); ++i)
    if (li[i] != si[i])
      return false;
  return true;
}

/// True if `op` is a supported commutative reduction operation.
static bool isSupportedReduceOp(Operation *op) {
  return isa<arith::AddIOp>(*op) || isa<arith::AddFOp>(*op) ||
         isa<arith::MulIOp>(*op) || isa<arith::MulFOp>(*op);
}

struct ReductionCandidate {
  AffineReadOpInterface load;   ///< The loop-invariant load (accumulator read)
  AffineWriteOpInterface store; ///< The loop-invariant store (accumulator write)
};

/// Scan the DIRECT children of `forOp` for reduction candidates.
///
/// Returns at most one candidate per (memref, address) pair.  All three
/// soundness conditions (a)–(c) from the file header are checked here.
static SmallVector<ReductionCandidate>
findReductionCandidates(AffineForOp forOp) {
  // Collect all stores that are direct children of the loop body.
  SmallVector<AffineWriteOpInterface> allStores;
  for (auto &op : *forOp.getBody())
    if (auto s = dyn_cast<AffineWriteOpInterface>(&op))
      allStores.push_back(s);

  SmallVector<ReductionCandidate> results;

  for (auto store : allStores) {
    // ── Condition (a): store address must be loop-invariant. ──
    if (usesInductionVar(forOp, store))
      continue;

    // ── Condition (c): no other store to the same memref. ──
    Value memref = store.getMemRef();
    bool hasAliasConflict = llvm::any_of(allStores, [&](auto other) {
      return other.getOperation() != store.getOperation() &&
             other.getMemRef() == memref;
    });
    if (hasAliasConflict)
      continue;

    // ── Condition (b): stored value comes from a reduce op fed by
    //    a matching load. ──
    Value storedVal = store.getValueToStore();
    auto *reduceOp = storedVal.getDefiningOp();
    if (!reduceOp || reduceOp->getNumOperands() < 2)
      continue;
    if (!isSupportedReduceOp(reduceOp))
      continue;

    for (unsigned i = 0; i < 2; ++i) {
      auto *defOp = reduceOp->getOperand(i).getDefiningOp();
      if (!defOp)
        continue;
      auto load = dyn_cast<AffineReadOpInterface>(defOp);
      if (!load)
        continue;
      if (sameAccess(load, store) && !usesInductionVar(forOp, load)) {
        results.push_back({load, store});
        break;
      }
    }
  }
  return results;
}

// ─── Transformation ───────────────────────────────────────────────────────

/// Promote a single reduction candidate to an iter_arg.
static LogicalResult promoteReduction(AffineForOp forOp,
                                      ReductionCandidate cand) {
  Location loc = forOp.getLoc();
  OpBuilder builder(forOp);

  // ── Capture values from the old store before any IR mutation. ──
  Value storeMemRef = cand.store.getMemRef();
  AffineMap storeMap = cand.store.getAffineMap();
  SmallVector<Value> storeMapOps(cand.store.getMapOperands().begin(),
                                 cand.store.getMapOperands().end());

  // ── 1. Hoist: clone the reduction load before the loop. ──
  Operation *initLoadOp = builder.clone(*cand.load.getOperation());
  Value initVal = initLoadOp->getResult(0);

  // ── 2. Create new affine.for with one iter_arg. ──
  // Provide a body builder so the block gets a proper yield terminator.
  AffineForOp newFor;
  {
    // The body builder creates a placeholder yield that returns the
    // iter_arg unchanged.  We will replace it below.
    auto bodyBuilder = [](OpBuilder &b, Location loc, Value /*iv*/,
                          ValueRange iterArgs) {
      b.create<AffineYieldOp>(loc, iterArgs);
    };
    newFor = builder.create<AffineForOp>(
        loc, forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
        forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(),
        forOp.getStepAsInt(), /*iterArgs=*/ValueRange{initVal}, bodyBuilder);
  }

  // Preserve user-facing attributes (ttl.pipeline, ttl.reduction, etc.).
  for (auto &attr : forOp->getAttrs()) {
    if (attr.getName().getValue() == AffineForOp::getOperandSegmentSizeAttr())
      continue;
    newFor->setAttr(attr.getName(), attr.getValue());
  }

  // ── 3. Populate the new body. ──
  Block *newBody = newFor.getBody();
  // The body has: arg0 = IV, arg1 = iter_arg (accumulator).
  // It currently contains just the placeholder yield.
  Operation *placeholderYield = newBody->getTerminator();

  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), newBody->getArgument(0));
  // Pre-map: wherever the old code uses the reduction load result,
  // the new code uses the iter_arg instead.
  mapping.map(cand.load->getResult(0), newBody->getArgument(1));

  builder.setInsertionPoint(placeholderYield);

  Operation *origLoad = cand.load.getOperation();
  Operation *origStore = cand.store.getOperation();
  Value yieldVal;

  for (auto &op : forOp.getBody()->getOperations()) {
    if (isa<AffineYieldOp>(&op))
      continue;
    if (&op == origLoad)
      continue; // Replaced by iter_arg via the IRMapping.
    if (&op == origStore) {
      // Record the (mapped) value that was being stored — this is what
      // the new loop should yield.
      yieldVal = mapping.lookupOrDefault(cand.store.getValueToStore());
      continue; // Replaced by the yield + post-loop store.
    }
    builder.clone(op, mapping);
  }

  if (!yieldVal) {
    // Safety: this should never happen if findReductionCandidates is correct.
    newFor->erase();
    initLoadOp->erase();
    return failure();
  }

  // Replace the placeholder yield with one that carries the accumulator.
  builder.create<AffineYieldOp>(loc, ValueRange{yieldVal});
  placeholderYield->erase();

  // ── 4. Sink: store the final accumulated value after the loop. ──
  builder.setInsertionPointAfter(newFor);
  builder.create<AffineStoreOp>(loc, newFor.getResult(0), storeMemRef,
                                storeMap, storeMapOps);

  // ── 5. Remove the old loop. ──
  forOp.erase();
  return success();
}

// ─── Pass ─────────────────────────────────────────────────────────────────

struct TTLPromoteAccumulator
    : public PassWrapper<TTLPromoteAccumulator, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTLPromoteAccumulator)

  StringRef getArgument() const override { return "ttl-promote-accumulator"; }
  StringRef getDescription() const override {
    return "Promote loop-invariant reduction store-load pairs to iter_args";
  }

  void runOnOperation() override {
    // Walk innermost-first (post-order) so that promoting an inner loop
    // does not invalidate outer-loop candidates.
    SmallVector<AffineForOp> worklist;
    getOperation().walk([&](AffineForOp op) { worklist.push_back(op); });

    for (auto forOp : worklist) {
      auto candidates = findReductionCandidates(forOp);
      if (candidates.empty())
        continue;
      // Promote the first candidate.  After this, forOp is erased.
      // Additional reductions would need another pass invocation.
      if (failed(promoteReduction(forOp, candidates.front())))
        return signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createTTLPromoteAccumulatorPass() {
  return std::make_unique<TTLPromoteAccumulator>();
}
} // namespace mlir
