//===- TTLDialect.cpp - TTL dialect implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TTL/TTLDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ttl;

#include "mlir/Dialect/TTL/TTLOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/TTL/TTLOpsTypes.cpp.inc"

void TTLDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/TTL/TTLOpsTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/TTL/TTLOps.cpp.inc"
      >();
}
