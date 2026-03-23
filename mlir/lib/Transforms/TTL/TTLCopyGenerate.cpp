//===- TTLCopyGenerate.cpp - Pragma-to-upstream adapter: copy promotion ---===//
//
// Minimal pragma-to-upstream adapter for explicit copy promotion.
//
// Maps ttl.copy attributes to affineDataCopyGenerate (LoopUtils.h).
//
// NOTE: This pass is imperative (collect-then-process) rather than
// pattern-based because affineDataCopyGenerate modifies the IR directly
// without going through a PatternRewriter.  The greedy pattern driver
// would crash on the stale worklist entries.
//
// This is NOT a caching model. It translates region-scoped directives into
// invocations of the upstream utility, one per named tensor.
//
// Fixed implementation parameters (not user-controllable):
//   - fastMemorySpace = 3 (OpenCL __local address space)
//   - slowMemorySpace = 0 (OpenCL __global address space)
//   - generateDma = true  (produces memref.dma_start/dma_wait)
//   - tagMemorySpace = 0  (DMA tag buffer space, required by upstream API)
//   - fastMemCapacityBytes = 48KB (upper bound for promoted buffers)
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringMap.h"

using namespace mlir;

namespace {

static constexpr unsigned kLocalMemSpace = 3;

/// Find the innermost tile loop in a perfectly nested band.
/// After tilePerfectlyNested, tile loops have step > 1; point loops step = 1.
/// Calling affineDataCopyGenerate on the innermost tile loop gives the
/// tightest promoted access footprint (e.g. A→8×8 for matmul with tile 8,8,8).
static affine::AffineForOp findInnermostTileLoop(affine::AffineForOp root) {
  SmallVector<affine::AffineForOp> band;
  affine::getPerfectlyNestedLoops(band, root);
  affine::AffineForOp innermost = root;
  for (auto &loop : band) {
    if (loop.getStepAsInt() > 1)
      innermost = loop;
  }
  return innermost;
}

static LogicalResult processAnnotatedLoop(affine::AffineForOp loop) {
  auto copyAttr = loop->getAttrOfType<ArrayAttr>("ttl.copy");
  if (!copyAttr)
    return success();

  auto fn = loop->getParentOfType<func::FuncOp>();
  if (!fn)
    return loop.emitError("ttl-copy-generate: not inside a function"),
           failure();

  // Build name → arg index map from ttl.tensor metadata.
  llvm::StringMap<unsigned> nameToArgIdx;
  for (unsigned i = 0, e = fn.getNumArguments(); i < e; ++i) {
    if (auto dict = fn.getArgAttrOfType<DictionaryAttr>(i, "ttl.tensor")) {
      StringRef name = dict.getAs<StringAttr>("name").getValue();
      nameToArgIdx[name] = i;
    }
  }

  affine::AffineForOp targetLoop = findInnermostTileLoop(loop);

  affine::AffineCopyOptions copyOptions;
  copyOptions.generateDma = true;
  copyOptions.slowMemorySpace = 0;
  copyOptions.fastMemorySpace = kLocalMemSpace;
  copyOptions.tagMemorySpace = 0;
  copyOptions.fastMemCapacityBytes = 48 * 1024;

  for (auto attr : copyAttr) {
    auto dict = cast<DictionaryAttr>(attr);
    StringRef tensorName = dict.getAs<StringAttr>("name").getValue();

    auto it = nameToArgIdx.find(tensorName);
    if (it == nameToArgIdx.end()) {
      return loop.emitError("ttl-copy-generate: tensor '")
                 << tensorName << "' not found in function annotations",
             failure();
    }
    Value memrefArg = fn.getArgument(it->second);

    DenseSet<Operation *> copyNests;
    if (failed(affine::affineDataCopyGenerate(targetLoop, copyOptions,
                                              memrefArg, copyNests))) {
      return loop.emitError("ttl-copy-generate: failed for tensor '")
                 << tensorName << "'",
             failure();
    }

    for (Operation *nest : copyNests)
      nest->setAttr("ttl.generated_copy", UnitAttr::get(fn.getContext()));
  }

  loop->removeAttr("ttl.copy");
  return success();
}

struct TTLCopyGenerate
    : public PassWrapper<TTLCopyGenerate, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTLCopyGenerate)

  StringRef getArgument() const override { return "ttl-copy-generate"; }
  StringRef getDescription() const override {
    return "Pragma-to-upstream adapter: generate explicit copies for annotated "
           "affine loop regions via affineDataCopyGenerate";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    SmallVector<affine::AffineForOp> annotatedLoops;
    getOperation().walk([&](affine::AffineForOp forOp) {
      if (forOp->hasAttr("ttl.copy"))
        annotatedLoops.push_back(forOp);
    });

    for (auto loop : annotatedLoops)
      if (failed(processAnnotatedLoop(loop)))
        return signalPassFailure();
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createTTLCopyGeneratePass() {
  return std::make_unique<TTLCopyGenerate>();
}
} // namespace mlir
