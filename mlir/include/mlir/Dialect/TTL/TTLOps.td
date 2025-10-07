#ifndef TTL_OPS
#define TTL_OPS

include "mlir/IR/OpBase.td"
include "mlir/Dialect/TTL/TTLDialect.td"

//===----------------------------------------------------------------------===//
// TTL Operations
//===----------------------------------------------------------------------===//

// Base class for TTL operations
class TTL_Op<string mnemonic, list<Trait> traits = []> :
    Op<TTLDialect, mnemonic, traits>;

// A simple no-op operation
def TTL_NoOp : TTL_Op<"noop"> {
  let summary = "A no-op operation that does nothing";
  let description = [{
    This operation does nothing and has no side effects.
    It can be used for testing or as a placeholder.
  }];
  let assemblyFormat = [{ attr-dict }];
}

// Copy operation for transferring data between memory spaces
def TTL_CopyOp : TTL_Op<"copy"> {
  let summary = "Copy memref data between different memory spaces";
  let description = [{
    This operation copies data from a source memref to a destination memref.
    It is typically used for transferring data between different memory spaces
    such as global to local memory, or between different memory hierarchies.
    
    The operation ensures that the entire memref is copied from source to destination.
  }];
  
  let arguments = (ins AnyMemRef:$source, AnyMemRef:$destination);
  let results = (outs);
  
  let assemblyFormat = [{ 
    $source `to` $destination attr-dict `:` type($source) `to` type($destination)
  }];
  
  let hasVerifier = 1;
}

#endif // TTL_OPS