//===- TTLPipelineDataTransfer.cpp - Pipeline imports via TTL scheme -------===//
//
// Replaces the upstream AffinePipelineDataTransfer with a direct TTL
// double-buffering scheme.  Instead of skewing the loop body and producing
// memref<2x...> double buffers (which are hard to lower), we emit TTL
// pipeline ops that map 1:1 to the TTL C API:
//
//   ttl.start_import_db  → TTL_start_import_double_buffering(...)
//   ttl.step_import_db   → TTL_step_buffering(&db, next_tile)
//   ttl.finish_import_db → TTL_finish_buffering(&db)
//
// The pass runs at the affine level (after copy-generate, before lower-affine).
// It identifies import DMA pairs in loops marked with ttl.pipeline and
// replaces them with the TTL pipeline scheme.  Only read-only imports are
// pipelined; read-write buffers (like C in matmul) are left for
// TTLLowerCopies to handle as single-buffered blocking transfers.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/TTL/TTLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::affine;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Build ttl.create_shape from static dims in TTL order (reversed).
static Value buildShape(OpBuilder &b, Location loc, ArrayRef<int64_t> dims) {
  SmallVector<Value> vals;
  for (int i = dims.size() - 1; i >= 0; --i)
    vals.push_back(b.create<arith::ConstantIndexOp>(loc, dims[i]));
  return b.create<ttl::CreateShapeOp>(loc, ttl::ShapeType::get(b.getContext()),
                                      vals);
}

/// Build ttl.create_layout from static shape.
static Value buildLayout(OpBuilder &b, Location loc,
                         ArrayRef<int64_t> shape) {
  SmallVector<Value> dims;
  if (shape.size() >= 1)
    dims.push_back(b.create<arith::ConstantIndexOp>(loc, shape.back()));
  if (shape.size() >= 3)
    dims.push_back(b.create<arith::ConstantIndexOp>(
        loc, shape[shape.size() - 1] * shape[shape.size() - 2]));
  return b.create<ttl::CreateLayoutOp>(loc, ttl::LayoutType::get(b.getContext()),
                                       dims);
}

/// Check if a function argument is marked readonly via ttl.tensor.
static bool isReadonlyArg(func::FuncOp fn, Value arg) {
  for (unsigned i = 0; i < fn.getNumArguments(); ++i) {
    if (fn.getArgument(i) == arg) {
      if (auto d = fn.getArgAttrOfType<DictionaryAttr>(i, "ttl.tensor"))
        if (auto a = d.getAs<StringAttr>("access"))
          return a.getValue() == "readonly";
      break;
    }
  }
  return false;
}

/// If val is an affine.for induction variable, return the owning for op.
static AffineForOp getOwnerForOp(Value val) {
  if (auto arg = dyn_cast<BlockArgument>(val))
    if (auto forOp = dyn_cast<AffineForOp>(arg.getOwner()->getParentOp()))
      if (forOp.getInductionVar() == val)
        return forOp;
  return nullptr;
}

/// Materialize affinemap indices of an AffineLoadOp as SSA values.
static SmallVector<Value> materializeLoadIndices(OpBuilder &b,
                                                 AffineLoadOp loadOp) {
  Location loc = loadOp.getLoc();
  AffineMap map = loadOp.getAffineMap();
  SmallVector<Value> indices;
  for (unsigned i = 0; i < map.getNumResults(); ++i) {
    auto dimMap = AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                                {map.getResult(i)}, b.getContext());
    indices.push_back(
        b.create<AffineApplyOp>(loc, dimMap, loadOp.getMapOperands()));
  }
  return indices;
}

//===----------------------------------------------------------------------===//
// Import DMA pair: start + matching wait (matched by tag memref).
//===----------------------------------------------------------------------===//

struct ImportDMAPair {
  AffineDmaStartOp start;
  AffineDmaWaitOp wait;
  Value localAlloc;  // memref<TxT x elem, 3>
  Value globalRef;   // memref<MxK x elem>
  Value tagAlloc;    // memref<1xi32>
};

/// Find eligible import DMA pairs inside a loop.
/// Eligible = dst is faster memory, src not also dst of an export in scope.
static void findEligibleImports(AffineForOp loop,
                                SmallVectorImpl<ImportDMAPair> &pairs) {
  // Collect outgoing DMAs (exports: src is faster memory).
  SmallVector<AffineDmaStartOp> exports;
  for (auto &op : *loop.getBody()) {
    if (auto dma = dyn_cast<AffineDmaStartOp>(op))
      if (dma.isSrcMemorySpaceFaster())
        exports.push_back(dma);
  }

  // Collect import DMAs (dst is faster memory) that don't conflict.
  SmallVector<AffineDmaStartOp> imports;
  for (auto &op : *loop.getBody()) {
    auto dma = dyn_cast<AffineDmaStartOp>(op);
    if (!dma || !dma.isDestMemorySpaceFaster())
      continue;
    // Skip if src memref is also the dst of an export (read-write dep).
    bool conflict = false;
    for (auto exp : exports) {
      if (exp.getDstMemRef() == dma.getSrcMemRef()) {
        conflict = true;
        break;
      }
    }
    if (!conflict)
      imports.push_back(dma);
  }

  // Match each import with its wait (by tag memref).
  for (auto dma : imports) {
    Value tag = dma.getTagMemRef();
    AffineDmaWaitOp matchingWait;
    for (auto &op : *loop.getBody()) {
      if (auto w = dyn_cast<AffineDmaWaitOp>(op)) {
        if (w.getTagMemRef() == tag) {
          matchingWait = w;
          break;
        }
      }
    }
    if (!matchingWait)
      continue;

    ImportDMAPair pair;
    pair.start = dma;
    pair.wait = matchingWait;
    pair.localAlloc = dma.getDstMemRef();
    pair.globalRef = dma.getSrcMemRef();
    pair.tagAlloc = tag;
    pairs.push_back(pair);
  }
}

/// Find the innermost nested AffineForOp that directly contains import DMAs.
static AffineForOp findInnermostDMALoop(AffineForOp outerLoop) {
  AffineForOp result;
  outerLoop.walk([&](AffineDmaStartOp dma) {
    if (!dma.isDestMemorySpaceFaster())
      return;
    auto parent = dma->getParentOfType<AffineForOp>();
    if (!parent)
      return;
    // Pick the deepest (innermost) loop.
    if (!result || result->isAncestor(parent))
      result = parent;
  });
  return result;
}

//===----------------------------------------------------------------------===//
// Core: pipeline one import DMA pair
//===----------------------------------------------------------------------===//

static LogicalResult pipelineImport(ImportDMAPair &pair,
                                    AffineForOp dmaLoop) {
  auto fn = dmaLoop->getParentOfType<func::FuncOp>();
  if (!fn)
    return failure();

  MLIRContext *ctx = fn.getContext();
  Location loc = pair.start.getLoc();

  auto localType = cast<MemRefType>(pair.localAlloc.getType());
  auto globalType = cast<MemRefType>(pair.globalRef.getType());
  auto tileShape = localType.getShape();       // e.g. [8, 8]
  unsigned rank = globalType.getRank();

  // Compute effective full shape, resolving dynamic dims via loop bounds.
  auto srcIndicesForShape = pair.start.getSrcIndices();
  SmallVector<int64_t> fullShape;
  for (unsigned i = 0; i < rank; ++i) {
    int64_t dimSize = globalType.getShape()[i];
    if (ShapedType::isDynamic(dimSize)) {
      AffineForOp owner = getOwnerForOp(srcIndicesForShape[i]);
      if (owner)
        dimSize = owner.getConstantUpperBound();
      else
        return pair.start.emitError(
            "cannot determine full tensor extent for dynamic dim"),
               failure();
    }
    fullShape.push_back(dimSize);
  }

  int64_t tileElems = 1;
  for (auto d : tileShape)
    tileElems *= d;

  // ---- Function-scope TTL infrastructure ----
  OpBuilder b(ctx);
  b.setInsertionPointToStart(&fn.getBody().front());

  auto flatType = MemRefType::get({tileElems}, localType.getElementType(),
                                  {}, 3);
  auto alloc0 = b.create<ttl::AllocLocalOp>(loc, flatType,
                                             b.getIndexAttr(tileElems));
  auto alloc1 = b.create<ttl::AllocLocalOp>(loc, flatType,
                                             b.getIndexAttr(tileElems));

  Value shapeFullVal = buildShape(b, loc, fullShape);
  Value shapeTileVal = buildShape(b, loc, tileShape);
  Value extLayout = buildLayout(b, loc, fullShape);
  Value tiler = b.create<ttl::CreateTilerOp>(
      loc, ttl::TilerType::get(ctx), shapeFullVal, shapeTileVal);

  bool isConst = isReadonlyArg(fn, pair.globalRef);
  Value extTensor = b.create<ttl::CreateExtTensorOp>(
      loc, ttl::ExtTensorType::get(ctx), pair.globalRef,
      shapeFullVal, extLayout, b.getBoolAttr(isConst),
      /*offset_dims=*/ValueRange{});

  Value event = b.create<ttl::CreateEventOp>(loc, ttl::EventType::get(ctx));

  // ---- Tile coordinate computation ----
  // Determine which DMA src index is the loop IV (the varying dimension).
  auto srcIndices = pair.start.getSrcIndices();
  int varyingMlirDim = -1;
  for (unsigned i = 0; i < rank; ++i) {
    if (srcIndices[i] == dmaLoop.getInductionVar()) {
      varyingMlirDim = i;
      break;
    }
  }
  if (varyingMlirDim < 0)
    return pair.start.emitError("cannot identify varying DMA dimension");

  // ---- Prologue: before the DMA loop ----
  b.setInsertionPoint(dmaLoop);

  // First tile: use DMA src indices at the loop's lower bound.
  // For the varying dim, substitute the loop lower bound; others are outer IVs.
  int64_t lb = dmaLoop.getConstantLowerBound();
  int64_t step = dmaLoop.getStepAsInt();

  SmallVector<Value> firstCoords;
  for (int i = rank - 1; i >= 0; --i) {
    if (i == varyingMlirDim) {
      // At loop start: tile coord = lb / tile_dim
      int64_t firstTileCoord = lb / tileShape[i];
      firstCoords.push_back(
          b.create<arith::ConstantIndexOp>(loc, firstTileCoord));
    } else {
      // Outer loop IV: compute tile coord from the IV.
      Value idx = srcIndices[i];
      AffineForOp owner = getOwnerForOp(idx);
      int64_t dimStep = owner ? owner.getStepAsInt() : tileShape[i];
      Value dimStepVal = b.create<arith::ConstantIndexOp>(loc, dimStep);
      Value coord = b.create<arith::DivUIOp>(loc, idx, dimStepVal);
      firstCoords.push_back(coord);
    }
  }

  // Pad to 3 coordinates (x, y, z) for create_tile.
  while (firstCoords.size() < 3)
    firstCoords.push_back(b.create<arith::ConstantIndexOp>(loc, 0));

  Value firstTile = b.create<ttl::CreateTileOp>(
      loc, ttl::TileType::get(ctx),
      firstCoords[0], firstCoords[1], firstCoords[2], tiler);

  Value db = b.create<ttl::StartImportDBOp>(
      loc, ttl::ImportDBType::get(ctx),
      alloc0.getResult(), alloc1.getResult(), extTensor, event, firstTile);

  // ---- Loop body: step_import_db ----
  // Insert at the beginning of the loop body (after the IV arg).
  b.setInsertionPointToStart(dmaLoop.getBody());

  Value loopIV = dmaLoop.getInductionVar();
  Value stepVal = b.create<arith::ConstantIndexOp>(loc, step);
  Value nextK = b.create<arith::AddIOp>(loc, loopIV, stepVal);

  // Check if next iteration is in bounds.
  int64_t ub = dmaLoop.getConstantUpperBound();
  Value ubVal = b.create<arith::ConstantIndexOp>(loc, ub);
  Value inBounds = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, nextK, ubVal);

  // Compute tile coords for the NEXT tile.
  SmallVector<Value> nextCoords;
  for (int i = rank - 1; i >= 0; --i) {
    if (i == varyingMlirDim) {
      Value tileDim = b.create<arith::ConstantIndexOp>(loc, tileShape[i]);
      Value nextCoord = b.create<arith::DivUIOp>(loc, nextK, tileDim);
      nextCoords.push_back(nextCoord);
    } else {
      Value idx = srcIndices[i];
      AffineForOp owner = getOwnerForOp(idx);
      int64_t dimStep = owner ? owner.getStepAsInt() : tileShape[i];
      Value dimStepVal = b.create<arith::ConstantIndexOp>(loc, dimStep);
      Value coord = b.create<arith::DivUIOp>(loc, idx, dimStepVal);
      nextCoords.push_back(coord);
    }
  }
  while (nextCoords.size() < 3)
    nextCoords.push_back(b.create<arith::ConstantIndexOp>(loc, 0));

  // scf.if for boundary: real tile vs empty tile.
  auto tileTy = ttl::TileType::get(ctx);
  auto ifOp = b.create<scf::IfOp>(
      loc, TypeRange{tileTy}, inBounds,
      /*addThenBlock=*/true, /*addElseBlock=*/true);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(&ifOp.getThenRegion().front());
    Value realTile = b.create<ttl::CreateTileOp>(
        loc, tileTy, nextCoords[0], nextCoords[1], nextCoords[2], tiler);
    b.create<scf::YieldOp>(loc, realTile);
  }
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(&ifOp.getElseRegion().front());
    Value emptyTile = b.create<ttl::CreateEmptyTileOp>(loc, tileTy);
    b.create<scf::YieldOp>(loc, emptyTile);
  }
  Value nextTile = ifOp.getResult(0);

  Value sub = b.create<ttl::StepImportDBOp>(
      loc, ttl::IntTensorType::get(ctx), db, nextTile);

  // ---- Replace affine.load on the local buffer with ttl.read_tensor ----
  SmallVector<AffineLoadOp> loadsToReplace;
  dmaLoop.walk([&](AffineLoadOp loadOp) {
    if (loadOp.getMemRef() == pair.localAlloc)
      loadsToReplace.push_back(loadOp);
  });
  for (auto loadOp : loadsToReplace) {
    b.setInsertionPoint(loadOp);
    auto indices = materializeLoadIndices(b, loadOp);
    // TTL x = last dim, y = second-to-last dim.
    Value x, y;
    if (indices.size() == 1) {
      x = indices[0];
      y = b.create<arith::ConstantIndexOp>(loadOp.getLoc(), 0);
    } else {
      x = indices.back();
      y = indices[indices.size() - 2];
    }
    Value rd = b.create<ttl::ReadTensorOp>(
        loadOp.getLoc(), loadOp.getResult().getType(), sub, x, y);
    loadOp.getResult().replaceAllUsesWith(rd);
    loadOp.erase();
  }

  // ---- Epilogue: after the DMA loop ----
  b.setInsertionPointAfter(dmaLoop);
  b.create<ttl::FinishImportDBOp>(loc, db);

  // ---- Erase old DMA ops, tag alloc, deallocs, local alloc ----
  pair.wait.erase();
  pair.start.erase();

  // Erase deallocs on the local alloc and tag alloc.
  auto eraseDeallocs = [&](Value memref) {
    SmallVector<memref::DeallocOp> deallocs;
    for (auto *user : memref.getUsers())
      if (auto d = dyn_cast<memref::DeallocOp>(user))
        deallocs.push_back(d);
    for (auto d : deallocs)
      d.erase();
  };
  eraseDeallocs(pair.localAlloc);
  eraseDeallocs(pair.tagAlloc);

  // Erase the tag alloc.
  if (auto tagAllocOp = pair.tagAlloc.getDefiningOp())
    if (tagAllocOp->use_empty())
      tagAllocOp->erase();

  // Erase the local alloc.
  if (auto localAllocOp = pair.localAlloc.getDefiningOp())
    if (localAllocOp->use_empty())
      localAllocOp->erase();

  return success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct TTLPipelineDataTransfer
    : public PassWrapper<TTLPipelineDataTransfer, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTLPipelineDataTransfer)

  StringRef getArgument() const override {
    return "ttl-pipeline-data-transfer";
  }
  StringRef getDescription() const override {
    return "Pipeline import DMA transfers using TTL double-buffering scheme";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, func::FuncDialect,
                    memref::MemRefDialect, arith::ArithDialect,
                    scf::SCFDialect, ttl::TTLDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Find loops with ttl.pipeline.
    SmallVector<AffineForOp> pipelineLoops;
    module.walk([&](AffineForOp op) {
      if (op->hasAttr("ttl.pipeline"))
        pipelineLoops.push_back(op);
    });

    if (pipelineLoops.empty())
      return;

    for (auto outerLoop : pipelineLoops) {
      // Find the innermost loop that contains import DMA pairs.
      AffineForOp dmaLoop = findInnermostDMALoop(outerLoop);
      if (!dmaLoop) {
        outerLoop->removeAttr("ttl.pipeline");
        continue;
      }

      SmallVector<ImportDMAPair> pairs;
      findEligibleImports(dmaLoop, pairs);

      if (pairs.empty()) {
        outerLoop->removeAttr("ttl.pipeline");
        continue;
      }

      for (auto &pair : pairs) {
        if (failed(pipelineImport(pair, dmaLoop)))
          return signalPassFailure();
      }

      outerLoop->removeAttr("ttl.pipeline");
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createTTLPipelineDataTransferPass() {
  return std::make_unique<TTLPipelineDataTransfer>();
}
} // namespace mlir
