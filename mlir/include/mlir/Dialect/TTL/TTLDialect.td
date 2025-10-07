#ifndef TTL_DIALECT
#define TTL_DIALECT

include "mlir/IR/OpBase.td"

def TTLDialect : Dialect {
  let name = "ttl";
  let cppNamespace = "::mlir::ttl";
  let description = [{
    The TTL dialect for tensor tiling operations.
  }];
}

#endif // TTL_DIALECT 