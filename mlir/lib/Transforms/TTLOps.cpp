#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "TTLDataStructures.h"
#include <set>
#include <map>

using namespace mlir;

namespace {

// Custom comparator for Value
struct ValueComparator {
  bool operator()(const Value& lhs, const Value& rhs) const {
    return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
  }
};

// Helper to get field index from GEP
int getFieldIndex(LLVM::GEPOp gepOp) {
  auto indices = gepOp.getIndices();
  if (indices.size() <= 1) return -1;
  
  auto index = indices[1];
  if (auto intAttr = index.dyn_cast<IntegerAttr>()) {
    return intAttr.getInt();
  } else if (auto constOp = index.dyn_cast<Value>().getDefiningOp<LLVM::ConstantOp>()) {
    if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>())
      return intAttr.getInt();
  }
  return -1;
}

void analyzeLoopStruct(Value loopStruct, TTLAnalysis& analysis) {
  for (auto& block : loopStruct.getParentRegion()->getBlocks()) {
    for (auto& op : block) {
      if (auto storeOp = dyn_cast<LLVM::StoreOp>(op)) {
        if (auto gepOp = storeOp.getAddr().getDefiningOp<LLVM::GEPOp>()) {
          if (gepOp.getBase() != loopStruct) continue;
          
          int fieldIndex = getFieldIndex(gepOp);
          if (fieldIndex < 0) continue;
          
          // Handle loop variables (indices 1-12)
          if (fieldIndex >= 1 && fieldIndex <= 12) {
            int dimIndex = (fieldIndex - 1) / 4;
            int fieldOffset = (fieldIndex - 1) % 4;
            const char* dimName = dimIndex == 0 ? "x" : dimIndex == 1 ? "y" : "z";
            
            Value storedValue = storeOp.getValue();
            bool isDynamic = true;
            int value = 0;
            
            // Check if it's a constant
            if (auto constOp = storedValue.getDefiningOp<LLVM::ConstantOp>()) {
              if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
                value = intAttr.getInt();
                isDynamic = false;
              }
            } else {
              // If not a constant, it's dynamic (e.g., function argument)
              // For now, we'll use -1 to indicate dynamic value
              value = -1;
              isDynamic = true;
            }
            
            // Add to analysis based on field offset
            if (fieldOffset == 0) { // start
              analysis.addLoop(dimName, value, 0, 0, 0, isDynamic);
            } else if (fieldOffset == 1) { // end
              analysis.updateLoopEnd(dimName, value, isDynamic);
            } else if (fieldOffset == 2) { // step
              analysis.updateLoopStep(dimName, value, isDynamic);
            } else if (fieldOffset == 3) { // loop_dimension
              analysis.updateLoopCurrent(dimName, value, isDynamic);
            }
          }
        }
      }
    }
  }
}

void analyzeTensorAccess(Value tensorStruct, const char* tensorName, Value loopStruct, TTLAnalysis& analysis) {
  TensorAccess access;
  std::map<Value, const char*, ValueComparator> loopVarMap;

  // First analyze loop struct to build mapping of loop variables
  for (auto& block : loopStruct.getParentRegion()->getBlocks()) {
    for (auto& op : block) {
      if (auto gepOp = dyn_cast<LLVM::GEPOp>(op)) {
        if (gepOp.getBase() != loopStruct) continue;
        
        int fieldIndex = getFieldIndex(gepOp);
        if (fieldIndex < 0) continue;
        
        // Map loop variables (x=4, y=8, z=12)
        if (fieldIndex == 4) loopVarMap[gepOp] = "x";
        else if (fieldIndex == 8) loopVarMap[gepOp] = "y";
        else if (fieldIndex == 12) loopVarMap[gepOp] = "z";
      }
    }
  }

  // Now analyze tensor access
  for (auto& block : tensorStruct.getParentRegion()->getBlocks()) {
    for (auto& op : block) {
      if (auto storeOp = dyn_cast<LLVM::StoreOp>(op)) {
        if (auto gepOp = storeOp.getAddr().getDefiningOp<LLVM::GEPOp>()) {
          if (gepOp.getBase() != tensorStruct) continue;
          
          int fieldIndex = getFieldIndex(gepOp);
          if (fieldIndex < 0) continue;
          
          Value storedValue = storeOp.getValue();
          
          // Handle tensor access fields
          if (fieldIndex == 1) { // rank
            if (auto constOp = storedValue.getDefiningOp<LLVM::ConstantOp>()) {
              if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
                access.rank = intAttr.getInt();
              }
            }
          } else if (fieldIndex == 2) { // x_stride
            if (auto constOp = storedValue.getDefiningOp<LLVM::ConstantOp>()) {
              if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
                access.x_stride = intAttr.getInt();
              }
            } else {
              access.x_stride = -1; // Dynamic
              access.isDynamic = true;
            }
          } else if (fieldIndex == 3) { // y_stride
            if (auto constOp = storedValue.getDefiningOp<LLVM::ConstantOp>()) {
              if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
                access.y_stride = intAttr.getInt();
              }
            } else {
              access.y_stride = -1; // Dynamic
              access.isDynamic = true;
            }
          } else if (fieldIndex == 4) { // z_stride
            if (auto constOp = storedValue.getDefiningOp<LLVM::ConstantOp>()) {
              if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
                access.z_stride = intAttr.getInt();
              }
            } else {
              access.z_stride = -1; // Dynamic
              access.isDynamic = true;
            }
          } else if (fieldIndex == 5) { // x_index
            if (auto loadOp = storedValue.getDefiningOp<LLVM::LoadOp>()) {
              if (auto gepOp = loadOp.getAddr().getDefiningOp<LLVM::GEPOp>()) {
                if (loopVarMap.count(gepOp)) {
                  access.x_index = loopVarMap[gepOp];
                  analysis.addTensorAccessToLoop(loopVarMap[gepOp], tensorName);
                }
              }
            }
          } else if (fieldIndex == 6) { // y_index
            if (auto loadOp = storedValue.getDefiningOp<LLVM::LoadOp>()) {
              if (auto gepOp = loadOp.getAddr().getDefiningOp<LLVM::GEPOp>()) {
                if (loopVarMap.count(gepOp)) {
                  access.y_index = loopVarMap[gepOp];
                  analysis.addTensorAccessToLoop(loopVarMap[gepOp], tensorName);
                }
              }
            }
          } else if (fieldIndex == 7) { // z_index
            if (auto loadOp = storedValue.getDefiningOp<LLVM::LoadOp>()) {
              if (auto gepOp = loadOp.getAddr().getDefiningOp<LLVM::GEPOp>()) {
                if (loopVarMap.count(gepOp)) {
                  access.z_index = loopVarMap[gepOp];
                  analysis.addTensorAccessToLoop(loopVarMap[gepOp], tensorName);
                }
              }
            }
          }
        }
      }
    }
  }
  
  analysis.addTensorAccess(tensorName, access);
}

class TTLOps : public PassWrapper<TTLOps, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTLOps)

  StringRef getArgument() const final { return "ttl-ops"; }

  void runOnOperation() override {
    getOperation()->walk([&](LLVM::LLVMFuncOp funcOp) {
      if (funcOp.getName() == "TTL_matmul_kernel") {
        llvm::errs() << "Analyzing TTL_matmul_kernel:\n";
        TTLAnalysis analysis;

        for (auto& block : funcOp.getBody()) {
          for (auto& op : block) {
            if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
              if (callOp.getCallee()->str() != "TTL_loop_affine_matmul_body") continue;

              Value loopStruct = callOp.getOperand(0);
              analyzeLoopStruct(loopStruct, analysis);

              for (size_t i = 1; i < callOp.getNumOperands(); i++) {
                std::string tensorName = "Tensor" + std::to_string(i);
                analyzeTensorAccess(callOp.getOperand(i), tensorName.c_str(), loopStruct, analysis);
              }
            }
          }
        }
        
        // Print analysis results
        for (const auto& loop : analysis.loops) {
          llvm::errs() << "\nLoop " << loop.name << ":\n";
          
          // Start
          llvm::errs() << "  start = ";
          if (loop.var.start == -1) {
            llvm::errs() << "dynamic (dynamic)\n";
          } else {
            llvm::errs() << loop.var.start << " (constant)\n";
          }
          
          // End
          llvm::errs() << "  end = ";
          if (loop.var.end == -1) {
            llvm::errs() << "dynamic (dynamic)\n";
          } else {
            llvm::errs() << loop.var.end << " (constant)\n";
          }
          
          // Step
          llvm::errs() << "  step = ";
          if (loop.var.step == -1) {
            llvm::errs() << "dynamic (dynamic)\n";
          } else {
            llvm::errs() << loop.var.step << " (constant)\n";
          }
          
          // Loop dimension index (0=outer, 1=middle, 2=inner)
          llvm::errs() << "  loop_dimension = " << loop.var.current;
          if (loop.var.current == 0) {
            llvm::errs() << " (outer loop)\n";
          } else if (loop.var.current == 1) {
            llvm::errs() << " (middle loop)\n";
          } else if (loop.var.current == 2) {
            llvm::errs() << " (inner loop)\n";
          } else {
            llvm::errs() << " (dimension index)\n";
          }
          
          llvm::errs() << "  Accessed tensors: ";
          for (const auto& tensor : loop.accessedTensors) {
            llvm::errs() << tensor << " ";
          }
          llvm::errs() << "\n";
        }
        
        for (const auto& [name, access] : analysis.tensorAccesses) {
          llvm::errs() << "\nAnalyzing " << name << " tensor access:\n";
          llvm::errs() << "  rank = " << access.rank << "D\n";
          
          llvm::errs() << "  x_stride = ";
          if (access.x_stride == -1) {
            llvm::errs() << "dynamic (dynamic)\n";
          } else {
            llvm::errs() << access.x_stride << " (constant)\n";
          }
          
          llvm::errs() << "  y_stride = ";
          if (access.y_stride == -1) {
            llvm::errs() << "dynamic (dynamic)\n";
          } else {
            llvm::errs() << access.y_stride << " (constant)\n";
          }
          
          llvm::errs() << "  z_stride = ";
          if (access.z_stride == -1) {
            llvm::errs() << "dynamic (dynamic)\n";
          } else {
            llvm::errs() << access.z_stride << " (constant)\n";
          }
          
          llvm::errs() << "  x_index = " << access.x_index << " (loop var)\n";
          llvm::errs() << "  y_index = " << access.y_index << " (loop var)\n";
          llvm::errs() << "  z_index = " << access.z_index << " (loop var)\n";
          
          llvm::errs() << "  Accessed in loops: ";
          auto loops = analysis.getLoopsForTensor(name);
          for (const auto& loop : loops) {
            llvm::errs() << loop << " ";
          }
          llvm::errs() << "\n";
        }
      }
    });
  }
};

} // end anonymous namespace

namespace mlir {
std::unique_ptr<Pass> createTTLOpsPass() {
  return std::make_unique<TTLOps>();
}
} // end namespace mlir