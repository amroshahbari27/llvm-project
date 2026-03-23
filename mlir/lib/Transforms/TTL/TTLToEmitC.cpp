//===- TTLToEmitC.cpp - Lower TTL dialect to EmitC / OpenCL C output ------===//
//
// Custom backend: converts post-tiling, post-copy-promotion IR into EmitC
// dialect suitable for OpenCL C code generation via mlir-translate --mlir-to-cpp.
//
// Phases (in runOnOperation order):
//   0. replaceUndefs       — zero-init llvm.mlir.undef (Polygeist artifact)
//   1. convertKernelFuncs  — func.func → emitc.func with __kernel signature
//   2. lowerGlobalMemrefs  — patterns: scalar alloca, global load/store, dead cast
//   3. convertAllocToAlloca — patterns: dealloc erase, alloc → alloca
//   4. standard conversion — arith, scf, memref → emitc (upstream passes)
//   5. lowerTTLOps         — patterns: ttl.* → emitc.call_opaque (TTL C API)
//   6. reconcile casts     — clean up unrealized_conversion_cast
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/TTL/TTLDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static Value lookThroughCast(Value v) {
  if (auto cast = v.getDefiningOp<UnrealizedConversionCastOp>())
    if (cast.getNumOperands() == 1)
      return cast.getOperand(0);
  return v;
}

static emitc::OpaqueType opaque(MLIRContext *ctx, StringRef name) {
  return emitc::OpaqueType::get(ctx, name);
}

static Value toDim(OpBuilder &b, Location loc, Value v) {
  v = lookThroughCast(v);
  if (v.getType().isIndex() || isa<emitc::SizeTType>(v.getType()))
    return b.create<emitc::CastOp>(loc, opaque(b.getContext(), "TTL_dim_t"), v);
  return v;
}

static Value toUnsigned(OpBuilder &b, Location loc, Value v) {
  v = lookThroughCast(v);
  if (v.getType().isIndex() || isa<emitc::SizeTType>(v.getType()))
    return b.create<emitc::CastOp>(loc, opaque(b.getContext(), "unsigned"), v);
  return v;
}

/// Emit blocking TTL_import or TTL_export with event wait.
static void emitBlockingTransfer(OpBuilder &b, Location loc,
                                 StringRef fnName, Value intTensor,
                                 Value extTensor) {
  MLIRContext *ctx = b.getContext();
  auto eventTy = opaque(ctx, "TTL_event_t");
  Value ev = b.create<emitc::VariableOp>(
      loc, emitc::LValueType::get(eventTy),
      emitc::OpaqueAttr::get(ctx, "TTL_get_event()"));
  Value evPtr = b.create<emitc::ApplyOp>(
      loc, emitc::PointerType::get(eventTy), "&", ev);
  b.create<emitc::CallOpaqueOp>(
      loc, TypeRange{}, fnName,
      ArrayAttr{}, ArrayAttr{}, ValueRange{intTensor, extTensor, evPtr});
  Value one = b.create<emitc::ConstantOp>(
      loc, opaque(ctx, "int"), emitc::OpaqueAttr::get(ctx, "1"));
  b.create<emitc::CallOpaqueOp>(
      loc, TypeRange{}, "TTL_wait",
      ArrayAttr{}, ArrayAttr{}, ValueRange{one, evPtr});
}

/// Flat row-major index from N-D indices on a static memref.
static Value linearizeIndices(OpBuilder &b, Location loc,
                              MemRefType memrefType, OperandRange indices) {
  auto shape = memrefType.getShape();
  unsigned rank = shape.size();
  if (rank == 0 || indices.size() != rank)
    return nullptr;

  SmallVector<int64_t> strides(rank, 1);
  for (int i = (int)rank - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];

  Value flat;
  for (unsigned i = 0; i < rank; ++i) {
    Value idx = indices[i];
    if (!idx.getType().isInteger(32))
      idx = b.create<arith::IndexCastOp>(loc, b.getI32Type(), idx);
    if (strides[i] > 1) {
      Value s = b.create<arith::ConstantIntOp>(loc, strides[i], 32);
      idx = b.create<arith::MulIOp>(loc, idx, s);
    }
    flat = flat ? b.create<arith::AddIOp>(loc, flat, idx) : idx;
  }
  return flat;
}

//===----------------------------------------------------------------------===//
// Phase 0: Replace llvm.mlir.undef with zero constants (imperative).
//===----------------------------------------------------------------------===//

static void replaceUndefs(ModuleOp module) {
  SmallVector<Operation *> undefs;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "llvm.mlir.undef")
      undefs.push_back(op);
  });
  for (auto *op : undefs) {
    OpBuilder b(op);
    Location loc = op->getLoc();
    for (Value result : op->getResults()) {
      Type ty = result.getType();
      Value zero;
      if (ty.isInteger())
        zero = b.create<arith::ConstantIntOp>(loc, 0,
                                               ty.getIntOrFloatBitWidth());
      else if (ty.isF32())
        zero = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0.0f));
      else if (ty.isF64())
        zero = b.create<arith::ConstantOp>(loc, b.getF64FloatAttr(0.0));
      else
        continue;
      result.replaceAllUsesWith(zero);
    }
    op->erase();
  }
}

//===----------------------------------------------------------------------===//
// Phase 1: Convert func.func with ttl.kernel → emitc.func (imperative).
//
// Kept imperative: creates a new op type, clones body with IRMapping.
//===----------------------------------------------------------------------===//

static void insertIncludes(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  OpBuilder b(ctx);
  b.setInsertionPointToStart(module.getBody());

  bool hasTTL = false, hasStdint = false;
  for (auto inc : module.getOps<emitc::IncludeOp>()) {
    if (inc.getInclude() == "TTL/TTL.h") hasTTL = true;
    if (inc.getInclude() == "stdint.h") hasStdint = true;
  }
  if (!hasTTL)
    b.create<emitc::IncludeOp>(module.getLoc(), "TTL/TTL.h", false);
  if (!hasStdint)
    b.create<emitc::VerbatimOp>(module.getLoc(),
        "typedef int int32_t;\ntypedef long int64_t;\n"
        "typedef unsigned int uint32_t;\ntypedef unsigned long uint64_t;");

  // TTL pipeline-scheme type macros needed by generated double-buffering code.
  b.create<emitc::VerbatimOp>(module.getLoc(),
      "#undef TTL_IMPORT_DOUBLE_BUFFERING_TYPE\n"
      "#define TTL_IMPORT_DOUBLE_BUFFERING_TYPE "
        "__TTL_tensor_name(TTL_import_double_, const_, , "
        "TEST_TENSOR_TYPE, , _buffering_t)\n"
      "#undef TTL_EXPORT_DOUBLE_BUFFERING_TYPE\n"
      "#define TTL_EXPORT_DOUBLE_BUFFERING_TYPE "
        "__TTL_tensor_name(TTL_export_double_, const_, , "
        "TEST_TENSOR_TYPE, , _buffering_t)\n"
      "#undef TTL_INT_SUB_TENSOR_TYPE\n"
      "#define TTL_INT_SUB_TENSOR_TYPE "
        "__TTL_tensor_name(TTL_, , int_, TEST_TENSOR_TYPE, sub_, _t)");
}

static void convertKernelFuncs(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  SmallVector<func::FuncOp> kernels;
  module.walk([&](func::FuncOp fn) {
    if (fn->hasAttr("ttl.kernel"))
      kernels.push_back(fn);
  });

  for (func::FuncOp fn : kernels) {
    OpBuilder mb(ctx);
    mb.setInsertionPoint(fn);
    Location loc = fn.getLoc();

    SmallVector<Type> newArgTypes;
    for (unsigned i = 0, e = fn.getNumArguments(); i < e; ++i) {
      if (isa<MemRefType>(fn.getArgument(i).getType())) {
        bool readonly = false;
        if (auto dict = fn.getArgAttrOfType<DictionaryAttr>(i, "ttl.tensor"))
          if (auto acc = dict.getAs<StringAttr>("access"))
            readonly = (acc.getValue() == "readonly");
        std::string qt = readonly
            ? "__global const TEST_TENSOR_TYPE *restrict"
            : "__global TEST_TENSOR_TYPE *restrict";
        newArgTypes.push_back(emitc::OpaqueType::get(ctx, qt));
      } else {
        newArgTypes.push_back(fn.getArgument(i).getType());
      }
    }

    auto emitFn = mb.create<emitc::FuncOp>(
        loc, fn.getName(), FunctionType::get(ctx, newArgTypes, {}));
    emitFn->setAttr("specifiers",
                     ArrayAttr::get(ctx, {StringAttr::get(ctx, "__kernel")}));
    emitFn.addEntryBlock();

    Block *newEntry = &emitFn.getBody().front();
    Block *oldEntry = &fn.getBody().front();

    IRMapping mapping;
    OpBuilder ib = OpBuilder::atBlockBegin(newEntry);
    for (unsigned i = 0, e = fn.getNumArguments(); i < e; ++i) {
      Value oldArg = oldEntry->getArgument(i);
      Value newArg = newEntry->getArgument(i);
      if (isa<MemRefType>(oldArg.getType()) &&
          oldArg.getType() != newArg.getType()) {
        auto cast = ib.create<UnrealizedConversionCastOp>(
            loc, oldArg.getType(), newArg);
        mapping.map(oldArg, cast.getResult(0));
      } else {
        mapping.map(oldArg, newArg);
      }
    }

    for (Operation &op : oldEntry->getOperations()) {
      if (isa<func::ReturnOp>(op)) {
        ib.create<emitc::ReturnOp>(op.getLoc(), Value());
        continue;
      }
      ib.clone(op, mapping);
    }
    fn.erase();
  }
}

//===----------------------------------------------------------------------===//
// Phase 2 patterns: lower global memref access.
//===----------------------------------------------------------------------===//

/// Rank-0 memref.alloca (scalar accumulator) → emitc.variable.
struct LowerScalarAlloca : OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::AllocaOp op,
                                PatternRewriter &rw) const override {
    if (op.getType().getRank() != 0)
      return failure();

    Type elemTy = op.getType().getElementType();
    auto var = rw.create<emitc::VariableOp>(
        op.getLoc(), emitc::LValueType::get(elemTy),
        emitc::OpaqueAttr::get(getContext(), "0"));

    // Replace all loads and stores on this scalar alloca.
    SmallVector<memref::LoadOp> loads;
    SmallVector<memref::StoreOp> stores;
    for (auto *user : op.getResult().getUsers()) {
      if (auto ld = dyn_cast<memref::LoadOp>(user)) loads.push_back(ld);
      else if (auto st = dyn_cast<memref::StoreOp>(user)) stores.push_back(st);
    }
    for (auto ld : loads) {
      rw.setInsertionPoint(ld);
      Value v = rw.create<emitc::LoadOp>(ld.getLoc(), elemTy, var.getResult());
      rw.replaceOp(ld, v);
    }
    for (auto st : stores) {
      rw.setInsertionPoint(st);
      rw.create<emitc::AssignOp>(st.getLoc(), var.getResult(),
                                  st.getValueToStore());
      rw.eraseOp(st);
    }
    rw.eraseOp(op);
    return success();
  }
};

/// memref.load through unrealized_conversion_cast → linearized subscript.
struct LowerGlobalLoad : OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::LoadOp op,
                                PatternRewriter &rw) const override {
    auto castOp = op.getMemRef().getDefiningOp<UnrealizedConversionCastOp>();
    if (!castOp)
      return failure();

    Value ptr = castOp->getOperand(0);
    auto memrefType = cast<MemRefType>(op.getMemRef().getType());
    Value flat = linearizeIndices(rw, op.getLoc(), memrefType, op.getIndices());
    if (!flat)
      return op.emitError("cannot linearize memref indices"), failure();

    Type resTy = op.getResult().getType();
    Value slot = rw.create<emitc::SubscriptOp>(
        op.getLoc(), emitc::LValueType::get(resTy), ptr, ValueRange{flat});
    Value val = rw.create<emitc::LoadOp>(op.getLoc(), resTy, slot);
    rw.replaceOp(op, val);
    return success();
  }
};

/// memref.store through unrealized_conversion_cast → linearized subscript.
struct LowerGlobalStore : OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::StoreOp op,
                                PatternRewriter &rw) const override {
    auto castOp = op.getMemRef().getDefiningOp<UnrealizedConversionCastOp>();
    if (!castOp)
      return failure();

    Value ptr = castOp->getOperand(0);
    auto memrefType = cast<MemRefType>(op.getMemRef().getType());
    Value flat = linearizeIndices(rw, op.getLoc(), memrefType, op.getIndices());
    if (!flat)
      return op.emitError("cannot linearize memref indices"), failure();

    Type elemTy = op.getValueToStore().getType();
    Value slot = rw.create<emitc::SubscriptOp>(
        op.getLoc(), emitc::LValueType::get(elemTy), ptr, ValueRange{flat});
    rw.create<emitc::AssignOp>(op.getLoc(), slot, op.getValueToStore());
    rw.eraseOp(op);
    return success();
  }
};

/// Erase dead unrealized_conversion_cast with no uses.
struct EraseDeadCast : OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rw) const override {
    if (!op.getResult(0).use_empty())
      return failure();
    rw.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Phase 3 patterns: alloc → alloca, erase dealloc.
//===----------------------------------------------------------------------===//

struct EraseDealloc : OpRewritePattern<memref::DeallocOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::DeallocOp op,
                                PatternRewriter &rw) const override {
    rw.eraseOp(op);
    return success();
  }
};

/// memref.alloc (static shape) → memref.alloca without memory space.
struct AllocToAlloca : OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rw) const override {
    auto ty = op.getType();
    if (!ty.hasStaticShape())
      return failure();
    auto allocaTy = MemRefType::get(ty.getShape(), ty.getElementType());
    auto alloca = rw.create<memref::AllocaOp>(op.getLoc(), allocaTy);
    rw.replaceOp(op, alloca.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Phase 5 patterns: TTL ops → emitc.call_opaque (TTL C API).
//===----------------------------------------------------------------------===//

struct LowerCreateShape : OpRewritePattern<ttl::CreateShapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttl::CreateShapeOp op,
                                PatternRewriter &rw) const override {
    SmallVector<Value> dims;
    for (Value d : op.getDims())
      dims.push_back(toDim(rw, op.getLoc(), d));
    auto call = rw.create<emitc::CallOpaqueOp>(
        op.getLoc(), opaque(getContext(), "TTL_shape_t"), "TTL_create_shape",
        ArrayAttr{}, ArrayAttr{}, dims);
    rw.replaceOp(op, call.getResult(0));
    return success();
  }
};

struct LowerCreateLayout : OpRewritePattern<ttl::CreateLayoutOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttl::CreateLayoutOp op,
                                PatternRewriter &rw) const override {
    SmallVector<Value> args;
    for (Value d : op.getDims())
      args.push_back(toDim(rw, op.getLoc(), d));
    auto call = rw.create<emitc::CallOpaqueOp>(
        op.getLoc(), opaque(getContext(), "TTL_layout_t"), "TTL_create_layout",
        ArrayAttr{}, ArrayAttr{}, args);
    rw.replaceOp(op, call.getResult(0));
    return success();
  }
};

struct LowerCreateTiler : OpRewritePattern<ttl::CreateTilerOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttl::CreateTilerOp op,
                                PatternRewriter &rw) const override {
    auto call = rw.create<emitc::CallOpaqueOp>(
        op.getLoc(), opaque(getContext(), "TTL_tiler_t"), "TTL_create_tiler",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{op.getFullShape(), op.getTileShape()});
    rw.replaceOp(op, call.getResult(0));
    return success();
  }
};

struct LowerTileCount : OpRewritePattern<ttl::TileCountOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttl::TileCountOp op,
                                PatternRewriter &rw) const override {
    std::string fn = "TTL_tiles_in_" + op.getDim().str();
    auto call = rw.create<emitc::CallOpaqueOp>(
        op.getLoc(), opaque(getContext(), "int"), fn,
        ArrayAttr{}, ArrayAttr{}, ValueRange{op.getTiler()});
    rw.replaceOp(op, call.getResult(0));
    return success();
  }
};

struct LowerGetTile : OpRewritePattern<ttl::GetTileOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttl::GetTileOp op,
                                PatternRewriter &rw) const override {
    Value idx = toDim(rw, op.getLoc(), op.getIndex());
    auto call = rw.create<emitc::CallOpaqueOp>(
        op.getLoc(), opaque(getContext(), "TTL_tile_t"), "TTL_get_tile",
        ArrayAttr{}, ArrayAttr{}, ValueRange{idx, op.getTiler()});
    rw.replaceOp(op, call.getResult(0));
    return success();
  }
};

struct LowerTileEmpty : OpRewritePattern<ttl::TileEmptyOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttl::TileEmptyOp op,
                                PatternRewriter &rw) const override {
    auto call = rw.create<emitc::CallOpaqueOp>(
        op.getLoc(), IntegerType::get(getContext(), 1), "TTL_tile_empty",
        ArrayAttr{}, ArrayAttr{}, ValueRange{op.getTile()});
    rw.replaceOp(op, call.getResult(0));
    return success();
  }
};

struct LowerCreateExtTensor : OpRewritePattern<ttl::CreateExtTensorOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttl::CreateExtTensorOp op,
                                PatternRewriter &rw) const override {
    MLIRContext *ctx = getContext();
    Location loc = op.getLoc();
    Value base = lookThroughCast(op.getBase());
    bool isConst = op.getIsConst();

    std::string typeName = isConst
        ? "__TTL_tensor_name(TTL_, const_, ext_, TEST_TENSOR_TYPE, , _t)"
        : "__TTL_tensor_name(TTL_, , ext_, TEST_TENSOR_TYPE, , _t)";
    std::string fnName = isConst
        ? "TTL_create_const_ext_tensor" : "TTL_create_ext_tensor";

    SmallVector<Value> args = {base, op.getShape(), op.getLayout()};
    if (!op.getOffsetDims().empty()) {
      auto intT = opaque(ctx, "int");
      SmallVector<Value> offArgs;
      for (Value d : op.getOffsetDims())
        offArgs.push_back(rw.create<emitc::CastOp>(loc, intT,
                                                     lookThroughCast(d)));
      auto off = rw.create<emitc::CallOpaqueOp>(
          loc, opaque(ctx, "TTL_offset_t"), "TTL_create_offset",
          ArrayAttr{}, ArrayAttr{}, offArgs);
      args.push_back(off.getResult(0));
      args.push_back(rw.create<emitc::ConstantOp>(
          loc, opaque(ctx, "TTL_dim_t"),
          emitc::OpaqueAttr::get(ctx, "sizeof(TEST_TENSOR_TYPE)")));
    }
    auto call = rw.create<emitc::CallOpaqueOp>(
        loc, opaque(ctx, typeName), fnName, ArrayAttr{}, ArrayAttr{}, args);
    rw.replaceOp(op, call.getResult(0));
    return success();
  }
};

struct LowerCreateIntTensor : OpRewritePattern<ttl::CreateIntTensorOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttl::CreateIntTensorOp op,
                                PatternRewriter &rw) const override {
    MLIRContext *ctx = getContext();
    Value buf = lookThroughCast(op.getLocalBuf());
    auto call = rw.create<emitc::CallOpaqueOp>(
        op.getLoc(),
        opaque(ctx, "__TTL_tensor_name(TTL_, , int_, TEST_TENSOR_TYPE, , _t)"),
        "TTL_create_int_tensor", ArrayAttr{}, ArrayAttr{},
        ValueRange{buf, op.getShape(), op.getLayout()});
    rw.replaceOp(op, call.getResult(0));
    return success();
  }
};

struct LowerImport : OpRewritePattern<ttl::ImportOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttl::ImportOp op,
                                PatternRewriter &rw) const override {
    emitBlockingTransfer(rw, op.getLoc(), "TTL_import",
                         op.getIntTensor(), op.getExtTensor());
    rw.eraseOp(op);
    return success();
  }
};

struct LowerExport : OpRewritePattern<ttl::ExportOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttl::ExportOp op,
                                PatternRewriter &rw) const override {
    emitBlockingTransfer(rw, op.getLoc(), "TTL_export",
                         op.getIntTensor(), op.getExtTensor());
    rw.eraseOp(op);
    return success();
  }
};

struct LowerWaitImports : OpRewritePattern<ttl::WaitImportsOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttl::WaitImportsOp op,
                                PatternRewriter &rw) const override {
    rw.eraseOp(op);
    return success();
  }
};

struct LowerCreateEvent : OpRewritePattern<ttl::CreateEventOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttl::CreateEventOp op,
                                PatternRewriter &rw) const override {
    MLIRContext *ctx = getContext();
    auto eventTy = opaque(ctx, "TTL_event_t");
    auto var = rw.create<emitc::VariableOp>(
        op.getLoc(), emitc::LValueType::get(eventTy),
        emitc::OpaqueAttr::get(ctx, "TTL_get_event()"));
    rw.replaceOp(op, var.getResult());
    return success();
  }
};

struct LowerImportAsync : OpRewritePattern<ttl::ImportAsyncOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttl::ImportAsyncOp op,
                                PatternRewriter &rw) const override {
    MLIRContext *ctx = getContext();
    Location loc = op.getLoc();
    auto eventTy = opaque(ctx, "TTL_event_t");
    Value evPtr = rw.create<emitc::ApplyOp>(
        loc, emitc::PointerType::get(eventTy), "&",
        lookThroughCast(op.getEvent()));
    rw.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TTL_import",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{op.getIntTensor(), op.getExtTensor(), evPtr});
    rw.eraseOp(op);
    return success();
  }
};

struct LowerExportAsync : OpRewritePattern<ttl::ExportAsyncOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttl::ExportAsyncOp op,
                                PatternRewriter &rw) const override {
    MLIRContext *ctx = getContext();
    Location loc = op.getLoc();
    auto eventTy = opaque(ctx, "TTL_event_t");
    Value evPtr = rw.create<emitc::ApplyOp>(
        loc, emitc::PointerType::get(eventTy), "&",
        lookThroughCast(op.getEvent()));
    rw.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TTL_export",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{op.getIntTensor(), op.getExtTensor(), evPtr});
    rw.eraseOp(op);
    return success();
  }
};

struct LowerWait : OpRewritePattern<ttl::WaitOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttl::WaitOp op,
                                PatternRewriter &rw) const override {
    MLIRContext *ctx = getContext();
    Location loc = op.getLoc();
    auto eventTy = opaque(ctx, "TTL_event_t");
    Value evPtr = rw.create<emitc::ApplyOp>(
        loc, emitc::PointerType::get(eventTy), "&",
        lookThroughCast(op.getEvent()));
    Value one = rw.create<emitc::ConstantOp>(
        loc, opaque(ctx, "int"), emitc::OpaqueAttr::get(ctx, "1"));
    rw.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TTL_wait",
        ArrayAttr{}, ArrayAttr{}, ValueRange{one, evPtr});
    rw.eraseOp(op);
    return success();
  }
};

struct LowerReadTensor : OpRewritePattern<ttl::ReadTensorOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttl::ReadTensorOp op,
                                PatternRewriter &rw) const override {
    MLIRContext *ctx = getContext();
    Location loc = op.getLoc();
    Value x = toUnsigned(rw, loc, op.getX());
    Value y = toUnsigned(rw, loc, op.getY());
    auto call = rw.create<emitc::CallOpaqueOp>(
        loc, opaque(ctx, "TEST_TENSOR_TYPE"), "TTL_read_tensor",
        ArrayAttr{}, ArrayAttr{}, ValueRange{op.getTensor(), x, y});
    Value casted = rw.create<emitc::CastOp>(
        loc, op.getResult().getType(), call.getResult(0));
    rw.replaceOp(op, casted);
    return success();
  }
};

struct LowerWriteTensor : OpRewritePattern<ttl::WriteTensorOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttl::WriteTensorOp op,
                                PatternRewriter &rw) const override {
    MLIRContext *ctx = getContext();
    Location loc = op.getLoc();
    Value x = toUnsigned(rw, loc, op.getX());
    Value y = toUnsigned(rw, loc, op.getY());
    Value val = rw.create<emitc::CastOp>(
        loc, opaque(ctx, "TEST_TENSOR_TYPE"), op.getValue());
    rw.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TTL_write_tensor",
        ArrayAttr{}, ArrayAttr{}, ValueRange{op.getTensor(), val, x, y});
    rw.eraseOp(op);
    return success();
  }
};

struct LowerAllocLocal : OpRewritePattern<ttl::AllocLocalOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttl::AllocLocalOp op,
                                PatternRewriter &rw) const override {
    MLIRContext *ctx = getContext();
    auto arrTy = emitc::ArrayType::get(
        ctx, {op.getNumElements().getSExtValue()},
        opaque(ctx, "__local TEST_TENSOR_TYPE"));
    auto var = rw.create<emitc::VariableOp>(
        op.getLoc(), arrTy, emitc::OpaqueAttr::get(ctx, ""));
    rw.replaceOp(op, var.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pipeline scheme: tile construction + import double buffering
//===----------------------------------------------------------------------===//

struct LowerCreateTile : OpRewritePattern<ttl::CreateTileOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttl::CreateTileOp op,
                                PatternRewriter &rw) const override {
    MLIRContext *ctx = getContext();
    Location loc = op.getLoc();
    Value x = toDim(rw, loc, op.getX());
    Value y = toDim(rw, loc, op.getY());
    Value z = toDim(rw, loc, op.getZ());
    auto call = rw.create<emitc::CallOpaqueOp>(
        loc, opaque(ctx, "TTL_tile_t"), "TTL_create_tile",
        ArrayAttr{}, ArrayAttr{}, ValueRange{x, y, z, op.getTiler()});
    rw.replaceOp(op, call.getResult(0));
    return success();
  }
};

struct LowerCreateEmptyTile : OpRewritePattern<ttl::CreateEmptyTileOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttl::CreateEmptyTileOp op,
                                PatternRewriter &rw) const override {
    auto call = rw.create<emitc::CallOpaqueOp>(
        op.getLoc(), opaque(getContext(), "TTL_tile_t"),
        "TTL_create_empty_tile", ArrayAttr{}, ArrayAttr{}, ValueRange{});
    rw.replaceOp(op, call.getResult(0));
    return success();
  }
};

struct LowerStartImportDB : OpRewritePattern<ttl::StartImportDBOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttl::StartImportDBOp op,
                                PatternRewriter &rw) const override {
    MLIRContext *ctx = getContext();
    Location loc = op.getLoc();

    // The C API: TTL_start_import_double_buffering(buf0, buf1, ext, &event, tile)
    // Returns a struct. We create a variable and assign.
    auto dbTy = opaque(ctx, "TTL_IMPORT_DOUBLE_BUFFERING_TYPE");
    auto dbVar = rw.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(dbTy),
        emitc::OpaqueAttr::get(ctx, "{0}"));

    auto eventTy = opaque(ctx, "TTL_event_t");
    Value evPtr = rw.create<emitc::ApplyOp>(
        loc, emitc::PointerType::get(eventTy), "&",
        lookThroughCast(op.getEvent()));

    Value buf0 = lookThroughCast(op.getBuf0());
    Value buf1 = lookThroughCast(op.getBuf1());

    auto call = rw.create<emitc::CallOpaqueOp>(
        loc, dbTy, "TTL_start_import_double_buffering",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{buf0, buf1, op.getExt(), evPtr, op.getFirstTile()});
    rw.create<emitc::AssignOp>(loc, dbVar.getResult(), call.getResult(0));

    rw.replaceOp(op, dbVar.getResult());
    return success();
  }
};

struct LowerStepImportDB : OpRewritePattern<ttl::StepImportDBOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttl::StepImportDBOp op,
                                PatternRewriter &rw) const override {
    MLIRContext *ctx = getContext();
    Location loc = op.getLoc();

    // TTL_step_buffering(&db, next_tile) → sub tensor
    auto dbTy = opaque(ctx, "TTL_IMPORT_DOUBLE_BUFFERING_TYPE");
    Value dbPtr = rw.create<emitc::ApplyOp>(
        loc, emitc::PointerType::get(dbTy), "&",
        lookThroughCast(op.getDb()));

    auto subTy = opaque(ctx, "TTL_INT_SUB_TENSOR_TYPE");
    auto call = rw.create<emitc::CallOpaqueOp>(
        loc, subTy, "TTL_step_buffering",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dbPtr, op.getNextTile()});
    rw.replaceOp(op, call.getResult(0));
    return success();
  }
};

/// Fold emitc.cast(unrealized_conversion_cast(x)) → emitc.cast(x).
/// Eliminates the intermediate index roundtrip from the divui lowering.
struct FoldCastThroughUnrealizedCast : OpRewritePattern<emitc::CastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(emitc::CastOp op,
                                PatternRewriter &rw) const override {
    auto ucc = op.getSource().getDefiningOp<UnrealizedConversionCastOp>();
    if (!ucc || ucc.getNumOperands() != 1) return failure();
    rw.replaceOpWithNewOp<emitc::CastOp>(op, op.getType(), ucc.getOperand(0));
    return success();
  }
};

/// Lower arith.divui on index type → emitc.div on size_t.
/// The standard createConvertToEmitC converts arith on integer types but
/// leaves arith ops on index type when their operands come through casts.
struct LowerArithDivUIIndex : OpRewritePattern<arith::DivUIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::DivUIOp op,
                                PatternRewriter &rw) const override {
    if (!op.getType().isIndex()) return failure();
    auto stTy = emitc::SizeTType::get(getContext());
    Value lhs = lookThroughCast(op.getLhs());
    Value rhs = lookThroughCast(op.getRhs());
    if (!isa<emitc::SizeTType>(lhs.getType()))
      lhs = rw.create<emitc::CastOp>(op.getLoc(), stTy, lhs);
    if (!isa<emitc::SizeTType>(rhs.getType()))
      rhs = rw.create<emitc::CastOp>(op.getLoc(), stTy, rhs);
    auto div = rw.create<emitc::DivOp>(op.getLoc(), stTy, lhs, rhs);
    auto cast = rw.create<UnrealizedConversionCastOp>(
        op.getLoc(), op.getType(), div.getResult());
    rw.replaceOp(op, cast.getResult(0));
    return success();
  }
};

/// Lower scf.if with a single !ttl.tile or !emitc.opaque<"TTL_tile_t"> result
/// to emitc.conditional (C ternary).  The pattern fires after the body ops
/// have been lowered to EmitC calls, so each branch is a single call + yield.
/// We extract the call results and emit:  cond ? thenVal : elseVal
struct LowerScfIfToTernary : OpRewritePattern<scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rw) const override {
    if (ifOp.getNumResults() != 1) return failure();
    Type resTy = ifOp.getResultTypes().front();
    bool isTTLTile = isa<ttl::TileType>(resTy);
    bool isOpaqueTile = false;
    if (auto oTy = dyn_cast<emitc::OpaqueType>(resTy))
      isOpaqueTile = oTy.getValue().contains("TTL_tile_t");
    if (!isTTLTile && !isOpaqueTile) return failure();

    auto tileTy = opaque(getContext(), "TTL_tile_t");

    // Extract the yielded value from each branch.
    auto getYieldValue = [](Region &region) -> Value {
      auto yield = cast<scf::YieldOp>(region.front().getTerminator());
      return yield.getOperand(0);
    };

    Value thenVal = getYieldValue(ifOp.getThenRegion());
    Value elseVal = getYieldValue(ifOp.getElseRegion());

    // Move body ops (except yield) before the scf.if so they dominate.
    auto moveOpsBeforeIf = [&](Region &region) {
      Block &block = region.front();
      auto yield = cast<scf::YieldOp>(block.getTerminator());
      SmallVector<Operation *> ops;
      for (auto &op : block)
        if (&op != yield) ops.push_back(&op);
      for (auto *op : ops)
        op->moveBefore(ifOp);
    };
    moveOpsBeforeIf(ifOp.getThenRegion());
    moveOpsBeforeIf(ifOp.getElseRegion());

    // Create ternary: cond ? thenVal : elseVal
    rw.setInsertionPoint(ifOp);
    Value cond = ifOp.getCondition();
    auto ternary = rw.create<emitc::ConditionalOp>(
        ifOp.getLoc(), tileTy, cond, thenVal, elseVal);
    rw.replaceOp(ifOp, ternary.getResult());
    return success();
  }
};

struct LowerFinishImportDB : OpRewritePattern<ttl::FinishImportDBOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttl::FinishImportDBOp op,
                                PatternRewriter &rw) const override {
    MLIRContext *ctx = getContext();
    Location loc = op.getLoc();

    auto dbTy = opaque(ctx, "TTL_IMPORT_DOUBLE_BUFFERING_TYPE");
    Value dbPtr = rw.create<emitc::ApplyOp>(
        loc, emitc::PointerType::get(dbTy), "&",
        lookThroughCast(op.getDb()));
    rw.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TTL_finish_buffering",
        ArrayAttr{}, ArrayAttr{}, ValueRange{dbPtr});
    rw.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

static void populateGlobalMemrefPatterns(RewritePatternSet &patterns) {
  patterns.add<LowerScalarAlloca, LowerGlobalLoad, LowerGlobalStore,
               EraseDeadCast>(patterns.getContext());
}

static void populateAllocToAllocaPatterns(RewritePatternSet &patterns) {
  patterns.add<EraseDealloc, AllocToAlloca>(patterns.getContext());
}

static void populateTTLToEmitCPatterns(RewritePatternSet &patterns) {
  patterns.add<
      LowerCreateShape, LowerCreateLayout,
      LowerCreateTiler, LowerTileCount, LowerGetTile, LowerTileEmpty,
      LowerCreateExtTensor, LowerCreateIntTensor,
      LowerImport, LowerExport, LowerWaitImports,
      LowerCreateEvent, LowerImportAsync, LowerExportAsync, LowerWait,
      LowerReadTensor, LowerWriteTensor, LowerAllocLocal,
      LowerCreateTile, LowerCreateEmptyTile,
      LowerStartImportDB, LowerStepImportDB, LowerFinishImportDB,
      LowerArithDivUIIndex, LowerScfIfToTernary,
      FoldCastThroughUnrealizedCast, EraseDeadCast
  >(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct TTLToEmitC : public PassWrapper<TTLToEmitC, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTLToEmitC)

  StringRef getArgument() const override { return "ttl-to-emitc"; }
  StringRef getDescription() const override {
    return "Lower TTL dialect to EmitC for OpenCL C output";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<emitc::EmitCDialect, ttl::TTLDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Phase 0: Zero-init Polygeist undef artifacts.
    replaceUndefs(module);

    // Phase 1: Kernel func.func → emitc.func with __kernel signature.
    insertIncludes(module);
    convertKernelFuncs(module);

    // Phase 2: Linearize global memref access, lower scalar allocas.
    {
      RewritePatternSet patterns(module->getContext());
      populateGlobalMemrefPatterns(patterns);
      if (failed(applyPatternsGreedily(module, std::move(patterns))))
        return signalPassFailure();
    }

    // Phase 3: memref.alloc → memref.alloca for copy-gen buffers.
    {
      RewritePatternSet patterns(module->getContext());
      populateAllocToAllocaPatterns(patterns);
      if (failed(applyPatternsGreedily(module, std::move(patterns))))
        return signalPassFailure();
    }

    // Phase 4: Standard dialect-to-EmitC conversion.
    {
      PassManager pm(module->getContext());
      pm.addPass(createMem2Reg());
      pm.addPass(arith::createArithExpandOpsPass());
      pm.addPass(createConvertMathToEmitC());
      pm.addPass(createConvertToEmitC());
      pm.addPass(createReconcileUnrealizedCastsPass());
      if (failed(pm.run(module)))
        return signalPassFailure();
    }

    // Phase 5: TTL ops → emitc.call_opaque via rewrite patterns.
    {
      RewritePatternSet patterns(module->getContext());
      populateTTLToEmitCPatterns(patterns);
      if (failed(applyPatternsGreedily(module, std::move(patterns))))
        return signalPassFailure();
    }

    // Phase 6: Final cast cleanup.
    {
      PassManager pm(module->getContext());
      pm.addPass(createReconcileUnrealizedCastsPass());
      if (failed(pm.run(module)))
        return signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createTTLToEmitC() {
  return std::make_unique<TTLToEmitC>();
}
} // namespace mlir
