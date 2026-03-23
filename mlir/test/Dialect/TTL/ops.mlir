// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func.func @test_ttl_types
func.func @test_ttl_types(%n: index, %m: index) {
  // CHECK: ttl.create_shape
  %shape = ttl.create_shape(%n, %m)
  // CHECK: ttl.create_layout
  %layout = ttl.create_layout(%n)
  // CHECK: ttl.create_tiler
  %tiler = ttl.create_tiler(%shape, %shape)
  // CHECK: ttl.tile_count
  %tc = ttl.tile_count "width" of %tiler
  // CHECK: ttl.get_tile
  %c0 = arith.constant 0 : index
  %tile = ttl.get_tile(%c0, %tiler)
  // CHECK: ttl.tile_empty
  %empty = ttl.tile_empty %tile
  return
}
