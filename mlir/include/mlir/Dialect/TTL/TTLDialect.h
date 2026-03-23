#ifndef MLIR_DIALECT_TTL_TTLDIALECT_H
#define MLIR_DIALECT_TTL_TTLDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

// Generated dialect declaration
#include "mlir/Dialect/TTL/TTLOpsDialect.h.inc"

// Generated type declarations
#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/TTL/TTLOpsTypes.h.inc"

// Generated op declarations
#define GET_OP_CLASSES
#include "mlir/Dialect/TTL/TTLOps.h.inc"

#endif // MLIR_DIALECT_TTL_TTLDIALECT_H
