//===- TTLOps.cpp - MLIR TTL dialect ops implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TTL/TTLDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace mlir {
namespace ttl {

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

LogicalResult CopyOp::verify() {
  if (!isa<MemRefType>(getSource().getType()) || !isa<MemRefType>(getDestination().getType())) {
    return emitOpError("both source and destination must be memref types");
  }
  MemRefType sourceType = cast<MemRefType>(getSource().getType());
  MemRefType destType = cast<MemRefType>(getDestination().getType());

  if (sourceType.getElementType() != destType.getElementType()) {
    return emitOpError("source and destination must have the same element type");
  }

  if (sourceType.getShape() != destType.getShape()) {
    return emitOpError("source and destination must have the same shape");
  }

  return success();
}

} // namespace ttl
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/TTL/TTLOps.cpp.inc" 