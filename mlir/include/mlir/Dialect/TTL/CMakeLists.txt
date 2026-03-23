set(LLVM_TARGET_DEFINITIONS TTLOps.td)
mlir_tablegen(TTLOps.h.inc -gen-op-decls)
mlir_tablegen(TTLOps.cpp.inc -gen-op-defs)
mlir_tablegen(TTLOpsDialect.h.inc -gen-dialect-decls -dialect ttl)
mlir_tablegen(TTLOpsDialect.cpp.inc -gen-dialect-defs -dialect ttl)
mlir_tablegen(TTLOpsTypes.h.inc -gen-typedef-decls -typedefs-dialect=ttl)
mlir_tablegen(TTLOpsTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=ttl)
add_public_tablegen_target(MLIRTTLOpsIncGen)
add_dependencies(mlir-headers MLIRTTLOpsIncGen)

# Alias for the type-gen target.
add_custom_target(MLIRTTLOpsTypesIncGen DEPENDS MLIRTTLOpsIncGen)

add_mlir_doc(TTLOps TTLOps Dialects/ -gen-dialect-doc -dialect ttl)
