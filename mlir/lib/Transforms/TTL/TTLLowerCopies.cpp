//===- TTLLowerCopies.cpp - Convert DMA copy-promotion to TTL ops ---------===//
//
// Replaces DMA ops produced by affineDataCopyGenerate (generateDma=true)
// with TTL dialect operations:
//   memref.alloc (space 3)   → ttl.alloc_local
//   memref.dma_start (in)    → ttl.import
//   memref.dma_start (out)   → ttl.export
//   memref.load  (local)     → ttl.read_tensor
//   memref.store (local)     → ttl.write_tensor
//
// Implemented as a single rewrite pattern on memref.alloc with space 3.
//
// Runs after lower-affine, before ttl-to-emitc.
//
// Dimension order convention:
//   MLIR memrefs are row-major: [dim0, dim1, ..., dimN-1]
//   TTL shapes/offsets are width-first: [dimN-1, ..., dim1, dim0]
//   TTL read_tensor/write_tensor take (x, y) = (col, row) for 2-D.
//   TTL layouts take (row_spacing) for 2-D, (row_spacing, plane_spacing) for 3-D.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/TTL/TTLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// TTL descriptor helpers
//===----------------------------------------------------------------------===//

/// Build ttl.create_shape from a static memref shape, in TTL order (reversed).
static Value buildShape(OpBuilder &b, Location loc, ArrayRef<int64_t> shape) {
  MLIRContext *ctx = b.getContext();
  SmallVector<Value> dims;
  for (int i = shape.size() - 1; i >= 0; --i)
    dims.push_back(b.create<arith::ConstantIndexOp>(loc, shape[i]));
  return b.create<ttl::CreateShapeOp>(loc, ttl::ShapeType::get(ctx), dims);
}

/// Build ttl.create_layout from a memref shape.
///   rank 1: layout(width)
///   rank 2: layout(row_spacing = dim[1])
///   rank 3: layout(row_spacing = dim[2], plane_spacing = dim[1] * dim[2])
static Value buildLayout(OpBuilder &b, Location loc,
                         ArrayRef<int64_t> shape) {
  MLIRContext *ctx = b.getContext();
  unsigned rank = shape.size();
  SmallVector<Value> dims;
  if (rank >= 1)
    dims.push_back(b.create<arith::ConstantIndexOp>(loc, shape.back()));
  if (rank >= 3)
    dims.push_back(b.create<arith::ConstantIndexOp>(
        loc, shape[rank - 1] * shape[rank - 2]));
  return b.create<ttl::CreateLayoutOp>(loc, ttl::LayoutType::get(ctx), dims);
}

/// Extract global-side offsets from a DMA op, in TTL order (reversed).
static SmallVector<Value> extractDmaOffsets(OpBuilder &b,
                                            memref::DmaStartOp dma) {
  bool srcIsGlobal = (dma.getSrcMemorySpace() == 0);
  auto indices = srcIsGlobal ? dma.getSrcIndices() : dma.getDstIndices();
  SmallVector<Value> offs(indices.begin(), indices.end());
  std::reverse(offs.begin(), offs.end());
  return offs;
}

/// Map memref indices to TTL (x, y) for read_tensor / write_tensor.
///   rank 1: x = idx[0], y = 0
///   rank 2: x = idx[1] (col), y = idx[0] (row)
///   rank 3: x = idx[2], y = idx[1]  (depth in idx[0] folded into y)
static std::pair<Value, Value> mapToXY(OpBuilder &b, Location loc,
                                       OperandRange indices) {
  unsigned n = indices.size();
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  if (n == 0) return {zero, zero};
  if (n == 1) return {indices[0], zero};
  return {indices[n - 1], indices[n - 2]};
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

//===----------------------------------------------------------------------===//
// Gather DMA info for one local buffer.
//===----------------------------------------------------------------------===//

struct LocalBufInfo {
  memref::AllocOp alloc;
  Value globalRef;
  memref::DmaStartOp dmaIn;
  memref::DmaWaitOp waitIn;
  memref::DmaStartOp dmaOut;
  memref::DmaWaitOp waitOut;
  SmallVector<memref::AllocOp> tagAllocs;
};

static LogicalResult gatherBufInfo(memref::AllocOp alloc, LocalBufInfo &info) {
  info.alloc = alloc;
  info.globalRef = nullptr;

  auto fn = alloc->getParentOfType<func::FuncOp>();
  if (!fn)
    return failure();

  fn.walk([&](memref::DmaStartOp dma) {
    bool srcLocal = (dma.getSrcMemRef() == alloc.getResult());
    bool dstLocal = (dma.getDstMemRef() == alloc.getResult());
    if (!srcLocal && !dstLocal) return;

    auto tagAlloc = dma.getTagMemRef().getDefiningOp<memref::AllocOp>();
    if (dstLocal) {
      info.dmaIn = dma;
      info.globalRef = dma.getSrcMemRef();
    } else {
      info.dmaOut = dma;
      if (!info.globalRef) info.globalRef = dma.getDstMemRef();
    }
    if (tagAlloc) info.tagAllocs.push_back(tagAlloc);
  });

  fn.walk([&](memref::DmaWaitOp w) {
    if (info.dmaIn && w.getTagMemRef() == info.dmaIn.getTagMemRef())
      info.waitIn = w;
    if (info.dmaOut && w.getTagMemRef() == info.dmaOut.getTagMemRef())
      info.waitOut = w;
  });

  return info.globalRef ? success() : failure();
}

//===----------------------------------------------------------------------===//
// Pattern: single-buffered alloc (space 3) → blocking TTL ops
//===----------------------------------------------------------------------===//

struct LowerLocalBuffer : OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp alloc,
                                PatternRewriter &rw) const override {
    if (alloc.getType().getMemorySpaceAsInt() != 3)
      return failure();

    LocalBufInfo info;
    if (failed(gatherBufInfo(alloc, info)))
      return alloc.emitError("ttl-lower-copies: no global memref found"),
             failure();

    auto localType = alloc.getType();
    auto localShape = localType.getShape();
    MLIRContext *ctx = getContext();
    Location loc = alloc.getLoc();

    // Shapes and layouts.
    Value shapeVal = buildShape(rw, loc, localShape);
    auto globalType = cast<MemRefType>(info.globalRef.getType());
    Value extLayout = buildLayout(rw, loc, globalType.getShape());
    Value intLayout = buildLayout(rw, loc, localShape);

    // ttl.alloc_local at function entry (OpenCL requires outermost scope).
    int64_t numElems = 1;
    for (auto d : localShape) numElems *= d;
    auto flatType = MemRefType::get({numElems}, localType.getElementType(),
                                    {}, 3);
    ttl::AllocLocalOp allocLocal;
    {
      auto fn = alloc->getParentOfType<func::FuncOp>();
      OpBuilder::InsertionGuard guard(rw);
      rw.setInsertionPointToStart(&fn.getBody().front());
      allocLocal = rw.create<ttl::AllocLocalOp>(loc, flatType,
                                                 rw.getIndexAttr(numElems));
    }

    // Internal tensor.
    auto intTensor = rw.create<ttl::CreateIntTensorOp>(
        loc, ttl::IntTensorType::get(ctx),
        allocLocal.getResult(), shapeVal, intLayout);

    auto fn = alloc->getParentOfType<func::FuncOp>();
    bool isConst = isReadonlyArg(fn, info.globalRef);

    // Copy-in DMA → ttl.import.
    if (info.dmaIn) {
      rw.setInsertionPoint(info.dmaIn);
      SmallVector<Value> offs = extractDmaOffsets(rw, info.dmaIn);
      auto ext = rw.create<ttl::CreateExtTensorOp>(
          loc, ttl::ExtTensorType::get(ctx), info.globalRef,
          shapeVal, extLayout, rw.getBoolAttr(isConst), offs);
      rw.create<ttl::ImportOp>(loc, intTensor, ext);
      if (info.waitIn) rw.eraseOp(info.waitIn);
      rw.eraseOp(info.dmaIn);
    }

    // Copy-out DMA → ttl.export.
    if (info.dmaOut) {
      rw.setInsertionPoint(info.dmaOut);
      SmallVector<Value> offs = extractDmaOffsets(rw, info.dmaOut);
      auto ext = rw.create<ttl::CreateExtTensorOp>(
          loc, ttl::ExtTensorType::get(ctx), info.globalRef,
          shapeVal, extLayout, rw.getBoolAttr(false), offs);
      rw.create<ttl::ExportOp>(loc, intTensor, ext);
      if (info.waitOut) rw.eraseOp(info.waitOut);
      rw.eraseOp(info.dmaOut);
    }

    // Replace local loads → ttl.read_tensor.
    SmallVector<memref::LoadOp> loads;
    fn.walk([&](memref::LoadOp ld) {
      if (ld.getMemRef() == alloc.getResult())
        loads.push_back(ld);
    });
    for (auto ld : loads) {
      rw.setInsertionPoint(ld);
      auto [x, y] = mapToXY(rw, ld.getLoc(), ld.getIndices());
      auto read = rw.create<ttl::ReadTensorOp>(
          ld.getLoc(), ld.getResult().getType(), intTensor, x, y);
      rw.replaceOp(ld, read.getResult());
    }

    // Replace local stores → ttl.write_tensor.
    SmallVector<memref::StoreOp> stores;
    fn.walk([&](memref::StoreOp st) {
      if (st.getMemRef() == alloc.getResult())
        stores.push_back(st);
    });
    for (auto st : stores) {
      rw.setInsertionPoint(st);
      auto [x, y] = mapToXY(rw, st.getLoc(), st.getIndices());
      rw.create<ttl::WriteTensorOp>(
          st.getLoc(), intTensor, st.getValueToStore(), x, y);
      rw.eraseOp(st);
    }

    // Erase deallocs on local buffer.
    SmallVector<memref::DeallocOp> deallocs;
    fn.walk([&](memref::DeallocOp d) {
      if (d.getMemref() == alloc.getResult())
        deallocs.push_back(d);
    });
    for (auto d : deallocs) rw.eraseOp(d);

    // Erase tag allocs and their deallocs.
    for (auto tag : info.tagAllocs) {
      SmallVector<memref::DeallocOp> td;
      fn.walk([&](memref::DeallocOp d) {
        if (d.getMemref() == tag.getResult()) td.push_back(d);
      });
      for (auto d : td) rw.eraseOp(d);
      if (tag->use_empty()) rw.eraseOp(tag);
    }

    rw.eraseOp(alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct TTLLowerCopies
    : public PassWrapper<TTLLowerCopies, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTLLowerCopies)

  StringRef getArgument() const override { return "ttl-lower-copies"; }
  StringRef getDescription() const override {
    return "Convert DMA copy-promotion ops to TTL dialect ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ttl::TTLDialect, arith::ArithDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerLocalBuffer>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();

    // Verify no space-3 allocs remain.
    bool remaining = false;
    getOperation().walk([&](memref::AllocOp op) {
      if (op.getType().getMemorySpaceAsInt() == 3) {
        op.emitError("ttl-lower-copies: local alloc was not converted");
        remaining = true;
      }
    });
    if (remaining)
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createTTLLowerCopiesPass() {
  return std::make_unique<TTLLowerCopies>();
}
} // namespace mlir
