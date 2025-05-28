#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
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

void analyzeLoopStruct(Value loopStruct, std::set<int>& seenFields) {
  for (auto& block : loopStruct.getParentRegion()->getBlocks()) {
    for (auto& op : block) {
      if (auto storeOp = dyn_cast<LLVM::StoreOp>(op)) {
        if (auto gepOp = storeOp.getAddr().getDefiningOp<LLVM::GEPOp>()) {
          if (gepOp.getBase() != loopStruct) continue;
          
          int fieldIndex = getFieldIndex(gepOp);
          if (fieldIndex < 0) continue;
          
          if (seenFields.count(fieldIndex) > 0) {
            llvm::errs() << "Error: Multiple writes to field " << fieldIndex << "\n";
            return;
          }
          seenFields.insert(fieldIndex);
          
          // Handle loop variables (indices 1-12)
          if (fieldIndex >= 1 && fieldIndex <= 12) {
            int dimIndex = (fieldIndex - 1) / 4;
            int fieldOffset = (fieldIndex - 1) % 4;
            const char* dimName = dimIndex == 0 ? "x" : dimIndex == 1 ? "y" : "z";
            const char* fieldName = fieldOffset == 0 ? "start" : fieldOffset == 1 ? "end" : fieldOffset == 2 ? "step" : "current";
            
            Value storedValue = storeOp.getValue();
            if (auto constOp = storedValue.getDefiningOp<LLVM::ConstantOp>()) {
              if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>())
                llvm::errs() << dimName << "." << fieldName << " = " << intAttr.getInt() << " (constant)\n";
            } else {
              llvm::errs() << dimName << "." << fieldName << " = dynamic\n";
            }
          }
        }
      }
    }
  }
}

void analyzeTensorAccess(Value tensorStruct, const char* tensorName, Value loopStruct) {
  llvm::errs() << "\nAnalyzing " << tensorName << " tensor access:\n";
  std::set<int> seenFields;
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
          
          if (seenFields.count(fieldIndex) > 0) {
            llvm::errs() << "Error: Multiple writes to field " << fieldIndex << "\n";
            return;
          }
          seenFields.insert(fieldIndex);
          
          // Handle tensor access fields
          if (fieldIndex == 0) { // base pointer
            llvm::errs() << "base = dynamic\n";
          } else if (fieldIndex == 1) { // rank
            Value storedValue = storeOp.getValue();
            if (auto constOp = storedValue.getDefiningOp<LLVM::ConstantOp>()) {
              if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
                int rank = intAttr.getInt();
                llvm::errs() << "rank = " << rank << "D\n";
              }
            }
          } else if (fieldIndex == 2) { // x_stride
            Value storedValue = storeOp.getValue();
            if (auto constOp = storedValue.getDefiningOp<LLVM::ConstantOp>()) {
              if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>())
                llvm::errs() << "x_stride = " << intAttr.getInt() << " (constant)\n";
            } else {
              llvm::errs() << "x_stride = dynamic\n";
            }
          } else if (fieldIndex == 3) { // y_stride
            Value storedValue = storeOp.getValue();
            if (auto constOp = storedValue.getDefiningOp<LLVM::ConstantOp>()) {
              if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>())
                llvm::errs() << "y_stride = " << intAttr.getInt() << " (constant)\n";
            } else {
              llvm::errs() << "y_stride = dynamic\n";
            }
          } else if (fieldIndex == 4) { // z_stride
            Value storedValue = storeOp.getValue();
            if (auto constOp = storedValue.getDefiningOp<LLVM::ConstantOp>()) {
              if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>())
                llvm::errs() << "z_stride = " << intAttr.getInt() << " (constant)\n";
            } else {
              llvm::errs() << "z_stride = dynamic\n";
            }
          } else if (fieldIndex == 5) { // x_index
            Value storedValue = storeOp.getValue();
            if (auto loadOp = storedValue.getDefiningOp<LLVM::LoadOp>()) {
              if (auto gepOp = loadOp.getAddr().getDefiningOp<LLVM::GEPOp>()) {
                if (loopVarMap.count(gepOp)) {
                  llvm::errs() << "x_index = " << loopVarMap[gepOp] << " (loop var)\n";
                } else {
                  llvm::errs() << "x_index = dynamic\n";
                }
              }
            } else if (auto constOp = storedValue.getDefiningOp<LLVM::ConstantOp>()) {
              if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>())
                llvm::errs() << "x_index = " << intAttr.getInt() << " (constant)\n";
            } else {
              llvm::errs() << "x_index = dynamic\n";
            }
          } else if (fieldIndex == 6) { // y_index
            Value storedValue = storeOp.getValue();
            if (auto loadOp = storedValue.getDefiningOp<LLVM::LoadOp>()) {
              if (auto gepOp = loadOp.getAddr().getDefiningOp<LLVM::GEPOp>()) {
                if (loopVarMap.count(gepOp)) {
                  llvm::errs() << "y_index = " << loopVarMap[gepOp] << " (loop var)\n";
                } else {
                  llvm::errs() << "y_index = dynamic\n";
                }
              }
            } else if (auto constOp = storedValue.getDefiningOp<LLVM::ConstantOp>()) {
              if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>())
                llvm::errs() << "y_index = " << intAttr.getInt() << " (constant)\n";
            } else {
              llvm::errs() << "y_index = dynamic\n";
            }
          } else if (fieldIndex == 7) { // z_index
            Value storedValue = storeOp.getValue();
            if (auto loadOp = storedValue.getDefiningOp<LLVM::LoadOp>()) {
              if (auto gepOp = loadOp.getAddr().getDefiningOp<LLVM::GEPOp>()) {
                if (loopVarMap.count(gepOp)) {
                  llvm::errs() << "z_index = " << loopVarMap[gepOp] << " (loop var)\n";
                } else {
                  llvm::errs() << "z_index = dynamic\n";
                }
              }
            } else if (auto constOp = storedValue.getDefiningOp<LLVM::ConstantOp>()) {
              if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>())
                llvm::errs() << "z_index = " << intAttr.getInt() << " (constant)\n";
            } else {
              llvm::errs() << "z_index = dynamic\n";
            }
          }
        }
      }
    }
  }
}

class TTLOps : public PassWrapper<TTLOps, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTLOps)

  StringRef getArgument() const final { return "ttl-ops"; }

  void runOnOperation() override {
    getOperation()->walk([&](LLVM::LLVMFuncOp funcOp) {
      if (funcOp.getName() == "TTL_matmul_kernel") {
        llvm::errs() << "Analyzing TTL_matmul_kernel:\n";
        std::set<int> seenFields;

        for (auto& block : funcOp.getBody()) {
          for (auto& op : block) {
            if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
              if (callOp.getCallee()->str() != "TTL_loop_affine_matmul_body") continue;

              Value loopStruct = callOp.getOperand(0);
              analyzeLoopStruct(loopStruct, seenFields);

              for (size_t i = 1; i < callOp.getNumOperands(); i++) {
                std::string tensorName = "Tensor" + std::to_string(i);
                analyzeTensorAccess(callOp.getOperand(i), tensorName.c_str(), loopStruct);
              }
            }
          }
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