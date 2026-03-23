//===- TTLOps.cpp - TTL dialect ops implementation ------------------------===//

#include "mlir/Dialect/TTL/TTLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::ttl;

LogicalResult CreateShapeOp::verify() {
  size_t n = getDims().size();
  if (n < 1 || n > 3)
    return emitOpError("expected 1 to 3 dimension operands, got ") << n;
  return success();
}

LogicalResult TileCountOp::verify() {
  StringRef d = getDim();
  if (d != "width" && d != "height" && d != "depth")
    return emitOpError("dim must be \"width\", \"height\", or \"depth\"");
  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/TTL/TTLOps.cpp.inc"
