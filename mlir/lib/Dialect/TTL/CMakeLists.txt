add_mlir_dialect_library(MLIRTTLDialect
  TTLDialect.cpp
  TTLOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/TTL

  DEPENDS
  MLIRTTLOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
) 