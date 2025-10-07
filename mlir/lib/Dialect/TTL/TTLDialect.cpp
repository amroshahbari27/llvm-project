#include "mlir/Dialect/TTL/TTLDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace mlir::ttl;

#include "mlir/Dialect/TTL/TTLOpsDialect.cpp.inc"

void TTLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/TTL/TTLOps.cpp.inc"
  >();
} 