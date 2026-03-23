#ifndef TTL_DIALECT
#define TTL_DIALECT

include "mlir/IR/OpBase.td"

def TTLDialect : Dialect {
  let name = "ttl";
  let cppNamespace = "::mlir::ttl";
  let summary = "Dialect for Tensor Tiling Library operations";
  let description = [{
    The TTL dialect represents structured tensor tiling and data movement
    operations corresponding to the TTL (Tensor Tiling Library) C API.
    It serves as the bridge between MLIR's affine/memref analysis world
    and the final OpenCL + TTL runtime code.
  }];

  let useDefaultTypePrinterParser = 1;
}

#endif // TTL_DIALECT
