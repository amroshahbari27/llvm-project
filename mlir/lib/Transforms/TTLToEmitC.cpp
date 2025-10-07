#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Conversion/MathToEmitC/MathToEmitC.h"
#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitCPass.h"
#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"
#include "mlir/Conversion/FuncToEmitC/FuncToEmitC.h"
#include "mlir/Conversion/ConvertToEmitC/ConvertToEmitCPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

struct TTLToEmitC : public PassWrapper<TTLToEmitC, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTLToEmitC)

  TTLToEmitC() = default;
  TTLToEmitC(const TTLToEmitC &other) : PassWrapper<TTLToEmitC, OperationPass<ModuleOp>>(other) {}

  StringRef getArgument() const override { return "ttl-to-emitc"; }
  StringRef getDescription() const override { return "Convert TTL operations to EmitC dialect"; }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    PassManager pm(module->getContext());
    
    // Equivalent to:
    // --lower-affine --convert-scf-to-emitc --convert-arith-to-emitc
    // --convert-math-to-emitc --convert-func-to-emitc --convert-to-emitc
    // dump mlir between each pass
    pm.addPass(createLowerAffinePass());
    // pm.addPass(createSCFToEmitC());
    // // Expand complex arith ops like arith.minsi before lowering to EmitC
    // pm.addPass(arith::createArithExpandOpsPass());
    // pm.addPass(createConvertArithToEmitC());
    // pm.addPass(createConvertMathToEmitC());
    // pm.addPass(createConvertMemRefToEmitC());
    // pm.addPass(createConvertFuncToEmitC());
    // pm.addPass(createConvertMemRefToEmitC());
    pm.addPass(arith::createArithExpandOpsPass());
    pm.addPass(createConvertToEmitC());
    pm.addPass(createConvertMathToEmitC());

    // Reconcile unrealized casts must run at module level
    pm.addPass(createReconcileUnrealizedCastsPass());
    
    if (failed(pm.run(module))) {
      signalPassFailure();
    }
  }
};

void registerTTLToEmitC() {
  PassRegistration<TTLToEmitC>();
}

} // end anonymous namespace

namespace mlir {
std::unique_ptr<Pass> createTTLToEmitC() {
  return std::make_unique<TTLToEmitC>();
}
} // end namespace mlir 