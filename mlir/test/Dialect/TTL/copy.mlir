// RUN: mlir-opt %s -verify-diagnostics

// Test basic copy operation
func.func @test_copy(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) {
  // Copy from global to local memory
  %result = "ttl.copy"(%arg0) to (%arg1) : memref<10x10xf32> to memref<10x10xf32>
  
  // Use the copied data
  %val = memref.load %result[%c0, %c0] : memref<10x10xf32>
  
  return
}

// Test with different memory spaces (if supported)
func.func @test_copy_memory_spaces(%arg0: memref<5x5xi32>, %arg1: memref<5x5xi32>) {
  // Copy between different memory hierarchies
  %result = "ttl.copy"(%arg0) to (%arg1) : memref<5x5xi32> to memref<5x5xi32>
  
  return
}

// Test with dynamic shapes
func.func @test_copy_dynamic(%arg0: memref<?x?xf64>, %arg1: memref<?x?xf64>) {
  %result = "ttl.copy"(%arg0) to (%arg1) : memref<?x?xf64> to memref<?x?xf64>
  
  return
} 