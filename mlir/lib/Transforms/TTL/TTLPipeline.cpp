//===- TTLPipeline.cpp - End-to-end TTL compilation pipeline --------------===//
//
// Orchestrates the pragma-guided compilation pipeline:
//
//   Upstream-guided affine optimization stages:
//     1. ttl-legality-check  — structural validation of annotations
//     2. ttl-tile            — affine tiling (tilePerfectlyNested)
//     3. ttl-copy-generate   — explicit copy promotion (DMA-based)
//     4. ttl-pipeline-data-transfer — double-buffer DMA ops (if ttl.pipeline)
//     5. lower-affine        — convert affine → scf (upstream)
//
//   Custom backend lowering:
//     6. ttl-lower-copies  — DMA ops → TTL ops (single + double buffer)
//     7. ttl-to-emitc      — TTL + remaining dialects → EmitC / OpenCL C
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

struct TTLPipeline : public PassWrapper<TTLPipeline, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTLPipeline)

  StringRef getArgument() const override { return "ttl-pipeline"; }
  StringRef getDescription() const override {
    return "End-to-end TTL compilation: "
           "legality → tile → copy → pipeline → lower-affine → "
           "lower-copies → EmitC";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    PassManager pm(module->getContext());

    // --- Upstream-guided affine optimization stages ---

    // Step 1: Validate structural preconditions for all annotations.
    pm.addPass(createTTLLegalityCheckPass());

    // Step 2: Tile annotated affine loop regions (upstream utility).
    pm.addPass(createTTLTilePass());

    // Step 2.5: Promote reduction store-load pairs to iter_args.
    // This hoists C[i,j] load/store out of the k-loop, replacing with
    // a scalar accumulator (iter_arg).  Must run after tiling and before
    // copy-generate so the promoted pattern carries through to DMA/TTL.
    pm.addPass(createTTLPromoteAccumulatorPass());

    // Step 3: Generate explicit copies for annotated tensors (upstream utility).
    pm.addPass(createTTLCopyGeneratePass());

    // Step 4: Pipeline data transfers (double-buffer DMA-promoted locals).
    // Only runs when ttl.pipeline is present. Produces double-buffered allocs
    // and skewed loop bodies that overlap data movement with compute.
    pm.addPass(createTTLPipelineDataTransferPass());

    // Step 5: Lower affine → SCF (upstream pass, no TTL involvement).
    pm.addPass(createLowerAffinePass());

    // --- Custom backend lowering ---

    // Step 6: Convert DMA ops to TTL dialect ops.
    // Handles both single-buffered (blocking import/export) and
    // double-buffered (async import/export with events).
    pm.addPass(createTTLLowerCopiesPass());

    // Step 7: Lower to EmitC / OpenCL C output.
    pm.addPass(createTTLToEmitC());

    if (failed(pm.run(module)))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createTTLPipelinePass() {
  return std::make_unique<TTLPipeline>();
}
} // namespace mlir
