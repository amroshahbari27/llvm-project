#ifndef MLIR_DIALECT_TTL_TTLDIALECT_H
#define MLIR_DIALECT_TTL_TTLDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

// Generated dialect declarations
#include "mlir/Dialect/TTL/TTLOpsDialect.h.inc"

// Generated op classes
#define GET_OP_CLASSES
#include "mlir/Dialect/TTL/TTLOps.h.inc"

#endif // MLIR_DIALECT_TTL_TTLDIALECT_H 