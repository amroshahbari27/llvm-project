#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"

using namespace mlir;

namespace {

struct TTLPipeline : public PassWrapper<TTLPipeline, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTLPipeline)

  // Default constructor
  TTLPipeline() = default;

  // Copy constructor - needed for pass cloning
  TTLPipeline(const TTLPipeline &other) : PassWrapper<TTLPipeline, OperationPass<ModuleOp>>(other) {}

  // Pass options
  ListOption<unsigned> tileSizes{
      *this, "tile-sizes",
      llvm::cl::desc("Tile sizes for each dimension"),
      llvm::cl::ZeroOrMore};

  StringRef getArgument() const override { return "ttl-pipeline"; }
  StringRef getDescription() const override { return "TTL pipeline"; }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    PassManager pm(module->getContext());
    
    // Add module-level passes
    pm.addPass(createTTLTilePass());
    pm.addPass(createTTLToEmitC());
    
    // Run the pipeline
    if (failed(pm.run(module))) {
      signalPassFailure();
    }
  }
};

// Register the pass
void registerTTLPipeline() {
  PassRegistration<TTLPipeline>();
}

} // end anonymous namespace

namespace mlir {
std::unique_ptr<Pass> createTTLPipelinePass() {
  return std::make_unique<TTLPipeline>();
}
} // end namespace mlir 