#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"

using namespace mlir;

namespace {

struct TTLTilePass : public PassWrapper<TTLTilePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTLTilePass)

  TTLTilePass() = default;
  TTLTilePass(const TTLTilePass &other) : PassWrapper<TTLTilePass, OperationPass<ModuleOp>>(other) {}

  StringRef getArgument() const override { return "ttl-tile"; }
  StringRef getDescription() const override { return "TTL tiling pass"; }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Walk all affine.for ops and tile those with a ttl.tile attribute
    // Only process the outermost loop of a band to avoid processing tiled loops
    module.walk([&](affine::AffineForOp forOp) {
      // Skip if this is not the outermost loop of a band
      if (auto parent = forOp->getParentOfType<affine::AffineForOp>())
        return;

      if (auto tileAttr = forOp->getAttrOfType<ArrayAttr>("ttl.tile")) {
        SmallVector<unsigned, 4> sizes;
        for (Attribute size : tileAttr.getValue()) {
          if (auto intAttr = llvm::dyn_cast<IntegerAttr>(size))
            sizes.push_back(intAttr.getInt());
        }
        if (!sizes.empty()) {
          SmallVector<affine::AffineForOp, 4> band;
          getPerfectlyNestedLoops(band, forOp);
          if (!band.empty()) {
            SmallVector<affine::AffineForOp, 4> tiledNest;
            if (failed(tilePerfectlyNested(band, sizes, &tiledNest))) {
              forOp.emitError("Failed to tile loop with sizes: ");
              for (auto s : sizes) forOp.emitError() << s << " ";
            }
          }
        }
      }
    });
  }
};

} // end anonymous namespace

namespace mlir {
std::unique_ptr<Pass> createTTLTilePass() {
  return std::make_unique<TTLTilePass>();
}
} // end namespace mlir 