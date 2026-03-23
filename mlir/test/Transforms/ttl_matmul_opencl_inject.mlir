// RUN: mlir-opt %s --ttl-pipeline=backend=opencl --verify-each=false | FileCheck %s
//
// This test checks the non-hardcoded, analysis-driven matmul injection:
// we expect to see TTL blocking import/export calls and local buffer usage
// in the lowered EmitC.

module {
  // A minimal, already-tiled loop nest that mimics the post-tiling shape we
  // expect from the ingress+tiler pipeline (tile sizes are read from loop steps).
  emitc.func @matmul_TTL_optimized(%A: !emitc.array<64x64xi32>, %B: !emitc.array<64x128xi32>, %C: !emitc.array<64x128xi32>) {
    %c0 = "emitc.constant"(){value = 0 : index} : () -> !emitc.size_t
    %c64 = "emitc.constant"(){value = 64 : index} : () -> !emitc.size_t
    %c8 = "emitc.constant"(){value = 8 : index} : () -> !emitc.size_t
    emitc.for %i = %c0 to %c64 step %c8 : !emitc.size_t {
      emitc.for %j = %c0 to %c64 step %c8 : !emitc.size_t {
      }
    }
    emitc.return
  }
}

// CHECK: emitc.include "TTL/TTL.h"
// CHECK: emitc.func @matmul_TTL_optimized({{.*}}) attributes {specifiers = ["__kernel"]}
// CHECK: "emitc.variable"
// CHECK: call_opaque "TTL_blocking_import"
// CHECK: call_opaque "TTL_blocking_export"
