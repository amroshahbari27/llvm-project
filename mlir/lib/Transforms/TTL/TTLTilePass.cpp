//===- TTLTilePass.cpp - Pragma-to-upstream adapter: affine tiling --------===//
//
// Minimal pragma-to-upstream adapter for affine loop tiling.
//
// Maps ttl.tile attributes to tilePerfectlyNested (LoopUtils.h).
//
// NOTE: This pass is imperative (collect-then-process) rather than
// pattern-based because tilePerfectlyNested modifies the IR directly
// without going through a PatternRewriter.  The greedy pattern driver
// would crash on the stale worklist entries.
//
// Contract:
//   1. Find each affine.for with a ttl.tile attribute  (region identification)
//   2. Extract the perfectly nested band from that loop (structural check)
//   3. Normalize tile sizes: 0 → 1 (size 0 = "don't tile this dim")
//   4. Call tilePerfectlyNested(band, sizes)            (upstream utility)
//   5. Transfer ttl.copy, ttl.pipeline from original root to new outermost loop
//   6. Remove the ttl.tile attribute                    (consumed)
//
// Fixed implementation parameters: none (tile sizes come from pragma).
// Semantic legality: assumed for the restricted supported kernel class.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct TTLTilePass : public PassWrapper<TTLTilePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTLTilePass)

  StringRef getArgument() const override { return "ttl-tile"; }
  StringRef getDescription() const override {
    return "Pragma-to-upstream adapter: tile annotated affine loop regions "
           "via tilePerfectlyNested";
  }

  void runOnOperation() override {
    SmallVector<affine::AffineForOp> annotatedLoops;
    getOperation().walk([&](affine::AffineForOp forOp) {
      if (forOp->hasAttr("ttl.tile"))
        annotatedLoops.push_back(forOp);
    });

    for (auto root : annotatedLoops) {
      auto tileAttr = root->getAttrOfType<ArrayAttr>("ttl.tile");
      if (!tileAttr)
        continue;

      // Parse tile sizes (0 → 1, i.e. "don't tile this dim").
      SmallVector<unsigned> tileSizes;
      for (Attribute a : tileAttr.getValue()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(a)) {
          int64_t v = intAttr.getInt();
          tileSizes.push_back(v <= 0 ? 1 : static_cast<unsigned>(v));
        }
      }
      if (tileSizes.empty())
        continue;

      SmallVector<affine::AffineForOp, 6> band;
      affine::getPerfectlyNestedLoops(band, root);
      if (band.empty()) {
        root.emitError("ttl-tile: no valid loop band");
        return signalPassFailure();
      }

      // Pad or truncate to match band depth.
      unsigned depth = band.size();
      SmallVector<unsigned> sizes(depth, 1);
      for (unsigned i = 0; i < std::min<unsigned>(tileSizes.size(), depth); ++i)
        sizes[i] = tileSizes[i];

      // Save attributes that must survive tiling.
      ArrayAttr copyAttr = root->getAttrOfType<ArrayAttr>("ttl.copy");
      ArrayAttr reductionAttr = root->getAttrOfType<ArrayAttr>("ttl.reduction");
      UnitAttr pipelineAttr = root->getAttrOfType<UnitAttr>("ttl.pipeline");

      SmallVector<affine::AffineForOp> tiledNest;
      if (failed(affine::tilePerfectlyNested(band, sizes, &tiledNest))) {
        root.emitError("ttl-tile: tilePerfectlyNested failed — "
                       "loop band not structurally legal");
        return signalPassFailure();
      }

      // Transfer preserved attributes to new outermost loop.
      if (!tiledNest.empty()) {
        auto newRoot = tiledNest.front();
        if (copyAttr) newRoot->setAttr("ttl.copy", copyAttr);
        if (reductionAttr) newRoot->setAttr("ttl.reduction", reductionAttr);
        if (pipelineAttr) newRoot->setAttr("ttl.pipeline", pipelineAttr);
      }

      // Consume ttl.tile.
      for (auto op : tiledNest)
        op->removeAttr("ttl.tile");
      root->removeAttr("ttl.tile");
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createTTLTilePass() {
  return std::make_unique<TTLTilePass>();
}
} // namespace mlir
