#ifndef TTL_TYPES
#define TTL_TYPES

include "mlir/IR/OpBase.td"
include "mlir/Dialect/TTL/TTLDialect.td"

//===----------------------------------------------------------------------===//
// TTL type definitions
//
// These types are opaque handles that correspond 1:1 to TTL C API struct
// types.  They carry no parameters — their internal structure is hidden
// until we lower to EmitC opaque types.
//===----------------------------------------------------------------------===//

class TTL_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TTLDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TTL_ShapeType : TTL_Type<"Shape", "shape"> {
  let summary = "TTL tensor shape descriptor";
  let description = [{
    Opaque handle representing a TTL_shape_t — tensor dimensions
    (width, height, and optionally depth).
  }];
}

def TTL_LayoutType : TTL_Type<"Layout", "layout"> {
  let summary = "TTL tensor layout descriptor";
  let description = [{
    Opaque handle representing a TTL_layout_t — stride information
    (row_spacing, plane_spacing).
  }];
}

def TTL_TilerType : TTL_Type<"Tiler", "tiler"> {
  let summary = "TTL tiler descriptor";
  let description = [{
    Opaque handle representing a TTL_tiler_t — partitions a full tensor
    shape into tile-sized sub-regions.
  }];
}

def TTL_TileType : TTL_Type<"Tile", "tile"> {
  let summary = "TTL tile descriptor";
  let description = [{
    Opaque handle representing a TTL_tile_t — a specific sub-region of a
    tensor, carrying shape and offset.
  }];
}

def TTL_EventType : TTL_Type<"Event", "event"> {
  let summary = "TTL DMA event handle";
  let description = [{
    Opaque handle representing a TTL_event_t — used to synchronize
    asynchronous import/export operations.
  }];
}

def TTL_ExtTensorType : TTL_Type<"ExtTensor", "ext_tensor"> {
  let summary = "TTL external tensor handle";
  let description = [{
    Opaque handle representing a TTL external tensor — data residing in
    global memory with shape, layout, and base pointer.
  }];
}

def TTL_IntTensorType : TTL_Type<"IntTensor", "int_tensor"> {
  let summary = "TTL internal tensor handle";
  let description = [{
    Opaque handle representing a TTL internal tensor — data residing in
    local (on-chip) memory, used for tile-level computation.
  }];
}

def TTL_ImportDBType : TTL_Type<"ImportDB", "import_db"> {
  let summary = "Import double-buffering context handle";
  let description = [{
    Opaque handle for an import double-buffering context
    (TTL_import_double_*_buffering_t).  Created by ttl.start_import_db,
    consumed by ttl.step_import_db and ttl.finish_import_db.
  }];
}

#endif // TTL_TYPES
