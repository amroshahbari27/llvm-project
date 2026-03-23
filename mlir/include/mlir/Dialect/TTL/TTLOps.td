#ifndef TTL_OPS
#define TTL_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/TTL/TTLDialect.td"
include "mlir/Dialect/TTL/TTLTypes.td"

class TTL_Op<string mnemonic, list<Trait> traits = []> :
    Op<TTLDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Shape / layout / tiler construction
//===----------------------------------------------------------------------===//

def TTL_CreateShapeOp : TTL_Op<"create_shape", [Pure]> {
  let summary = "Create a TTL shape descriptor from dimensions";
  let arguments = (ins Variadic<Index>:$dims);
  let results = (outs TTL_ShapeType:$result);
  let assemblyFormat = [{ `(` $dims `)` attr-dict }];
  let hasVerifier = 1;
}

def TTL_CreateLayoutOp : TTL_Op<"create_layout", [Pure]> {
  let summary = "Create a TTL layout from stride dimensions";
  let description = [{
    1-arg form: TTL_create_layout(row_spacing)        — plane_spacing = 0
    2-arg form: TTL_create_layout(row_spacing, plane_spacing) — for 3-D tensors
  }];
  let arguments = (ins Variadic<Index>:$dims);
  let results = (outs TTL_LayoutType:$result);
  let assemblyFormat = [{ `(` $dims `)` attr-dict }];
}

def TTL_CreateTilerOp : TTL_Op<"create_tiler", [Pure]> {
  let summary = "Create a tiler that partitions a shape into tiles";
  let arguments = (ins TTL_ShapeType:$full_shape,
                       TTL_ShapeType:$tile_shape);
  let results = (outs TTL_TilerType:$result);
  let assemblyFormat = [{ `(` $full_shape `,` $tile_shape `)` attr-dict }];
}

//===----------------------------------------------------------------------===//
// Tile query operations
//===----------------------------------------------------------------------===//

def TTL_TileCountOp : TTL_Op<"tile_count", [Pure]> {
  let summary = "Query number of tiles in a given dimension";
  let arguments = (ins TTL_TilerType:$tiler, StrAttr:$dim);
  let results = (outs Index:$result);
  let assemblyFormat = [{ $dim `of` $tiler attr-dict }];
  let hasVerifier = 1;
}

def TTL_GetTileOp : TTL_Op<"get_tile", [Pure]> {
  let summary = "Retrieve a specific tile by linear index";
  let arguments = (ins Index:$index, TTL_TilerType:$tiler);
  let results = (outs TTL_TileType:$result);
  let assemblyFormat = [{ `(` $index `,` $tiler `)` attr-dict }];
}

def TTL_TileEmptyOp : TTL_Op<"tile_empty", [Pure]> {
  let summary = "Check if a tile is empty (boundary handling)";
  let arguments = (ins TTL_TileType:$tile);
  let results = (outs I1:$result);
  let assemblyFormat = [{ $tile attr-dict }];
}

//===----------------------------------------------------------------------===//
// Local memory allocation
//===----------------------------------------------------------------------===//

def TTL_AllocLocalOp : TTL_Op<"alloc_local"> {
  let summary = "Allocate a local-memory buffer for a tile";
  let arguments = (ins IndexAttr:$num_elements);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{ $num_elements attr-dict `:` type($result) }];
}

//===----------------------------------------------------------------------===//
// Tensor construction — external and internal
//===----------------------------------------------------------------------===//

def TTL_CreateExtTensorOp : TTL_Op<"create_ext_tensor", [Pure]> {
  let summary = "Create a TTL external tensor from a global memory reference";
  let description = [{
    Wraps a global memory reference with shape and layout metadata.
    When offset is provided, creates a sub-tensor view at that offset
    (5-arg form with elem_size for the TTL C API).
  }];
  let arguments = (ins AnyMemRef:$base,
                       TTL_ShapeType:$shape,
                       TTL_LayoutType:$layout,
                       DefaultValuedAttr<BoolAttr, "false">:$is_const,
                       Variadic<Index>:$offset_dims);
  let results = (outs TTL_ExtTensorType:$result);
  let assemblyFormat = [{
    `(` $base `,` $shape `,` $layout (`,` `offset` `(` $offset_dims^ `)`)? `)` attr-dict `:` type($base)
  }];
}

def TTL_CreateIntTensorOp : TTL_Op<"create_int_tensor", [Pure]> {
  let summary = "Create a TTL internal tensor from a local buffer";
  let arguments = (ins AnyMemRef:$local_buf,
                       TTL_ShapeType:$shape,
                       TTL_LayoutType:$layout);
  let results = (outs TTL_IntTensorType:$result);
  let assemblyFormat = [{
    `(` $local_buf `,` $shape `,` $layout `)` attr-dict `:` type($local_buf)
  }];
}

//===----------------------------------------------------------------------===//
// Data movement: import / export (blocking scheme)
//===----------------------------------------------------------------------===//

def TTL_ImportOp : TTL_Op<"import", []> {
  let summary = "Import a tile from global to local memory (blocking)";
  let arguments = (ins TTL_IntTensorType:$int_tensor,
                       TTL_ExtTensorType:$ext_tensor);
  let results = (outs);
  let assemblyFormat = [{
    $int_tensor `from` $ext_tensor attr-dict
  }];
}

def TTL_ExportOp : TTL_Op<"export", []> {
  let summary = "Export a tile from local to global memory (blocking)";
  let arguments = (ins TTL_IntTensorType:$int_tensor,
                       TTL_ExtTensorType:$ext_tensor);
  let results = (outs);
  let assemblyFormat = [{
    $int_tensor `to` $ext_tensor attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Data movement: async import / export (non-blocking, event-based)
//===----------------------------------------------------------------------===//

def TTL_CreateEventOp : TTL_Op<"create_event", []> {
  let summary = "Allocate a DMA event handle";
  let arguments = (ins);
  let results = (outs TTL_EventType:$result);
  let assemblyFormat = [{ attr-dict }];
}

def TTL_ImportAsyncOp : TTL_Op<"import_async", []> {
  let summary = "Start a non-blocking import from global to local memory";
  let arguments = (ins TTL_IntTensorType:$int_tensor,
                       TTL_ExtTensorType:$ext_tensor,
                       TTL_EventType:$event);
  let results = (outs);
  let assemblyFormat = [{
    $int_tensor `from` $ext_tensor `event` $event attr-dict
  }];
}

def TTL_ExportAsyncOp : TTL_Op<"export_async", []> {
  let summary = "Start a non-blocking export from local to global memory";
  let arguments = (ins TTL_IntTensorType:$int_tensor,
                       TTL_ExtTensorType:$ext_tensor,
                       TTL_EventType:$event);
  let results = (outs);
  let assemblyFormat = [{
    $int_tensor `to` $ext_tensor `event` $event attr-dict
  }];
}

def TTL_WaitOp : TTL_Op<"wait", []> {
  let summary = "Wait for an async DMA event to complete";
  let arguments = (ins TTL_EventType:$event);
  let results = (outs);
  let assemblyFormat = [{ $event attr-dict }];
}

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

def TTL_WaitImportsOp : TTL_Op<"wait_imports", []> {
  let summary = "Wait for all pending async imports to complete";
  let description = [{
    Barrier: blocks until every preceding ttl.import that was started
    asynchronously has finished transferring data.  When used with the
    blocking import scheme (default), this is a no-op.
  }];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = [{ attr-dict }];
}

//===----------------------------------------------------------------------===//
// Element access on internal tensors
//===----------------------------------------------------------------------===//

def TTL_ReadTensorOp : TTL_Op<"read_tensor", [Pure]> {
  let summary = "Read an element from a TTL internal tensor";
  let arguments = (ins TTL_IntTensorType:$tensor,
                       Index:$x, Index:$y);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $tensor `[` $x `,` $y `]` attr-dict `:` type($result)
  }];
}

def TTL_WriteTensorOp : TTL_Op<"write_tensor", []> {
  let summary = "Write an element to a TTL internal tensor";
  let arguments = (ins TTL_IntTensorType:$tensor,
                       AnyType:$value,
                       Index:$x, Index:$y);
  let results = (outs);
  let assemblyFormat = [{
    $tensor `[` $x `,` $y `]` `=` $value attr-dict `:` type($value)
  }];
}

//===----------------------------------------------------------------------===//
// Tile construction
//===----------------------------------------------------------------------===//

def TTL_CreateTileOp : TTL_Op<"create_tile", [Pure]> {
  let summary = "Create a tile at (x, y, z) coordinates within a tiler";
  let arguments = (ins Index:$x, Index:$y, Index:$z,
                       TTL_TilerType:$tiler);
  let results = (outs TTL_TileType:$result);
  let assemblyFormat = [{ `(` $x `,` $y `,` $z `,` $tiler `)` attr-dict }];
}

def TTL_CreateEmptyTileOp : TTL_Op<"create_empty_tile", [Pure]> {
  let summary = "Create an empty (no-op) tile for boundary handling";
  let arguments = (ins);
  let results = (outs TTL_TileType:$result);
  let assemblyFormat = [{ attr-dict }];
}

//===----------------------------------------------------------------------===//
// Pipeline scheme: import double buffering
//===----------------------------------------------------------------------===//

def TTL_StartImportDBOp : TTL_Op<"start_import_db", []> {
  let summary = "Start import double-buffering: allocate context and prefetch first tile";
  let arguments = (ins AnyMemRef:$buf0, AnyMemRef:$buf1,
                       TTL_ExtTensorType:$ext,
                       TTL_EventType:$event,
                       TTL_TileType:$first_tile);
  let results = (outs TTL_ImportDBType:$db);
  let assemblyFormat = [{
    `(` $buf0 `,` $buf1 `,` $ext `,` $event `,` $first_tile `)` attr-dict
    `:` type($buf0) `,` type($buf1)
  }];
}

def TTL_StepImportDBOp : TTL_Op<"step_import_db", []> {
  let summary = "Step: wait for current import, start next, return ready buffer";
  let arguments = (ins TTL_ImportDBType:$db,
                       TTL_TileType:$next_tile);
  let results = (outs TTL_IntTensorType:$ready);
  let assemblyFormat = [{ `(` $db `,` $next_tile `)` attr-dict }];
}

def TTL_FinishImportDBOp : TTL_Op<"finish_import_db", []> {
  let summary = "Finish import double-buffering: wait for last transfer";
  let arguments = (ins TTL_ImportDBType:$db);
  let results = (outs);
  let assemblyFormat = [{ $db attr-dict }];
}

#endif // TTL_OPS
