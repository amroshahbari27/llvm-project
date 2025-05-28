#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

using namespace mlir;

namespace {

void analyzeLoopStruct(Value loopStruct, std::set<int>& seenFields) {
  // Look for stores to this struct
  for (auto& block : loopStruct.getParentRegion()->getBlocks()) {
    for (auto& op : block) {
      if (auto storeOp = dyn_cast<LLVM::StoreOp>(op)) {
        // Check if this is a direct store to the struct (dim field)
        if (storeOp.getAddr() == loopStruct) {
          if (seenFields.count(0) > 0) {
            llvm::errs() << "Error: Multiple writes to dim field\n";
            return;
          }
          seenFields.insert(0);

          Value storedValue = storeOp.getValue();
          if (auto constOp = storedValue.getDefiningOp<LLVM::ConstantOp>()) {
            if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
              int loopDim = intAttr.getInt();
              const char* dimType = loopDim == 0 ? "1D" : loopDim == 1 ? "2D" : "3D";
              llvm::errs() << "Loop type: " << dimType << "\n";
            }
          }
          continue;
        }

        if (auto gepOp = storeOp.getAddr().getDefiningOp<LLVM::GEPOp>()) {
          // Check if this GEP is operating on our loop struct
          if (gepOp.getBase() != loopStruct) continue;

          auto indices = gepOp.getIndices();
          if (indices.size() > 1) {
            int fieldIndex = -1;
            auto index = indices[1];
            if (auto intAttr = index.dyn_cast<IntegerAttr>()) {
              fieldIndex = intAttr.getInt();
            } else if (auto constOp = index.dyn_cast<Value>().getDefiningOp<LLVM::ConstantOp>()) {
              if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>())
                fieldIndex = intAttr.getInt();
            }

            // If we've already seen this field, it's an error
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
}

void analyzeTensorAccess(Value tensorStruct, const char* tensorName) {
  llvm::errs() << "\nAnalyzing " << tensorName << " tensor access:\n";
  std::set<int> seenFields;

  for (auto& block : tensorStruct.getParentRegion()->getBlocks()) {
    for (auto& op : block) {
      if (auto storeOp = dyn_cast<LLVM::StoreOp>(op)) {
        // Check for direct store to base pointer
        if (storeOp.getAddr() == tensorStruct) {
          if (seenFields.count(0) > 0) {
            llvm::errs() << "Error: Multiple writes to base field\n";
            return;
          }
          seenFields.insert(0);
          llvm::errs() << "base = dynamic\n";
          continue;
        }

        if (auto gepOp = storeOp.getAddr().getDefiningOp<LLVM::GEPOp>()) {
          if (gepOp.getBase() != tensorStruct) continue;

          auto indices = gepOp.getIndices();
          if (indices.size() > 1) {
            int fieldIndex = -1;
            auto index = indices[1];
            if (auto intAttr = index.dyn_cast<IntegerAttr>()) {
              fieldIndex = intAttr.getInt();
            } else if (auto constOp = index.dyn_cast<Value>().getDefiningOp<LLVM::ConstantOp>()) {
              if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>())
                fieldIndex = intAttr.getInt();
            }

            // If we've already seen this field, it's an error
            if (seenFields.count(fieldIndex) > 0) {
              llvm::errs() << "Error: Multiple writes to field " << fieldIndex << "\n";
              return;
            }
            seenFields.insert(fieldIndex);

            // Handle tensor access fields (indices 1-2)
            if (fieldIndex >= 1 && fieldIndex <= 2) {
              const char* fieldName = fieldIndex == 1 ? "row_stride" : "plane_stride";
              Value storedValue = storeOp.getValue();
              if (auto constOp = storedValue.getDefiningOp<LLVM::ConstantOp>()) {
                if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>())
                  llvm::errs() << fieldName << " = " << intAttr.getInt() << " (constant)\n";
              } else {
                llvm::errs() << fieldName << " = dynamic\n";
              }
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
      if (funcOp.getName() != "matmul_kernel") return;

      llvm::errs() << "Analyzing matmul_kernel:\n";
      std::set<int> seenFields;

      // First find the call to loop_affine_matmul_body
      for (auto& block : funcOp.getBody()) {
        for (auto& op : block) {
          if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
            if (callOp.getCallee()->str() != "loop_affine_matmul_body") continue;

            // Get the struct arguments
            Value loopStruct = callOp.getOperand(0);
            
            // Analyze loop struct
            analyzeLoopStruct(loopStruct, seenFields);

            // Analyze all tensor structs (all operands after the loop struct)
            for (size_t i = 1; i < callOp.getNumOperands(); i++) {
              std::string tensorName = "Tensor" + std::to_string(i);
              analyzeTensorAccess(callOp.getOperand(i), tensorName.c_str());
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