//===- TTLLegalityCheck.cpp - Structural legality before transformation ---===//
//
// Runs BEFORE any transformation pass.
//
// Checks:
//   Function-level:
//     1. ttl.kernel function must have at least one ttl.tensor annotation
//     2. No duplicate tensor names
//     3. Tensor access must be "readonly", "writeonly", or "readwrite"
//     4. VLA shape dims (if present) must be in dim_map; rank must match memref
//
//   Loop-level:
//     5. ttl.tile / ttl.copy on a loop requires ttl.kernel on the function
//     6. ttl.tile requires a perfectly nested band
//     7. Tile size count must not exceed band depth
//     8. Tile sizes must be non-negative integers
//     9. Copy tensor names must match ttl.tensor annotations
//    10. Copy scope must be "local" (private not implemented)
//    11. No duplicate copy directives for the same tensor on the same loop
//    12. No conflicting copy modes (read + write) for same tensor on same loop
//    13. All memory accesses must be affine (no memref.load/store)
//    14. ttl.pipeline requires ttl.tile and ttl.copy on the same loop
//
// Does NOT prove semantic correctness (no dependence analysis).
// Reduction safety is assumed for the restricted supported kernel class.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace {

struct TTLLegalityCheck
    : public PassWrapper<TTLLegalityCheck, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTLLegalityCheck)

  StringRef getArgument() const override { return "ttl-legality-check"; }
  StringRef getDescription() const override {
    return "Structural legality checker for TTL pragma annotations";
  }

  void runOnOperation() override {
    bool failed = false;

    getOperation().walk([&](func::FuncOp fn) {
      if (!fn->hasAttr("ttl.kernel"))
        return;
      if (checkFunctionMetadata(fn).failed())
        failed = true;
    });

    getOperation().walk([&](affine::AffineForOp forOp) {
      if (forOp->hasAttr("ttl.tile"))
        if (checkTileRegion(forOp).failed())
          failed = true;
      if (forOp->hasAttr("ttl.copy"))
        if (checkCopyRegion(forOp).failed())
          failed = true;
      if (forOp->hasAttr("ttl.pipeline"))
        if (checkPipelineRegion(forOp).failed())
          failed = true;
    });

    if (failed)
      signalPassFailure();
  }

private:
  //===--------------------------------------------------------------------===//
  // Function-level metadata
  //===--------------------------------------------------------------------===//

  LogicalResult checkFunctionMetadata(func::FuncOp fn) {
    bool hasTensor = false;
    llvm::StringSet<> tensorNames;

    for (unsigned i = 0, e = fn.getNumArguments(); i < e; ++i) {
      auto dict = fn.getArgAttrOfType<DictionaryAttr>(i, "ttl.tensor");
      if (!dict)
        continue;
      hasTensor = true;

      auto nameAttr = dict.getAs<StringAttr>("name");
      if (!nameAttr) {
        fn.emitError("ttl-legality-check: ttl.tensor missing 'name' field "
                     "on argument ") << i;
        return failure();
      }
      StringRef name = nameAttr.getValue();

      // Duplicate tensor name check.
      if (!tensorNames.insert(name).second) {
        fn.emitError("ttl-legality-check: duplicate tensor annotation "
                     "for '") << name << "'";
        return failure();
      }

      // Access field validation.
      if (auto acc = dict.getAs<StringAttr>("access")) {
        StringRef a = acc.getValue();
        if (a != "readonly" && a != "writeonly" && a != "readwrite") {
          fn.emitError("ttl-legality-check: tensor '") << name
              << "' access must be 'readonly', 'writeonly', or 'readwrite', "
                 "got '" << a << "'";
          return failure();
        }
      }

      // VLA shape validation (if present).
      auto shape = dict.getAs<ArrayAttr>("shape");
      if (shape && !shape.empty()) {
        auto dimMapAttr = fn->getAttrOfType<DictionaryAttr>("ttl.dim_map");
        llvm::StringSet<> knownDims;
        if (dimMapAttr)
          for (auto &entry : dimMapAttr)
            knownDims.insert(entry.getName().strref());

        for (auto a : shape) {
          StringRef dim = cast<StringAttr>(a).getValue();
          if (!knownDims.count(dim)) {
            fn.emitError("ttl-legality-check: tensor '")
                << name << "' dimension '" << dim << "' not in dim_map";
            return failure();
          }
        }
        if (auto memrefType =
                dyn_cast<MemRefType>(fn.getArgument(i).getType())) {
          if ((int64_t)shape.size() != memrefType.getRank()) {
            fn.emitError("ttl-legality-check: tensor '")
                << name << "' has " << shape.size()
                << " dims but memref has rank " << memrefType.getRank();
            return failure();
          }
        }
      }
    }

    if (!hasTensor) {
      fn.emitError("ttl-legality-check: kernel has no tensor annotations");
      return failure();
    }

    return success();
  }

  //===--------------------------------------------------------------------===//
  // Tiling region checks
  //===--------------------------------------------------------------------===//

  LogicalResult checkTileRegion(affine::AffineForOp root) {
    auto fn = root->getParentOfType<func::FuncOp>();
    if (fn && !fn->hasAttr("ttl.kernel")) {
      root.emitError("ttl-legality-check: ttl.tile on loop inside "
                     "non-kernel function");
      return failure();
    }

    SmallVector<affine::AffineForOp, 6> band;
    affine::getPerfectlyNestedLoops(band, root);
    if (band.empty()) {
      root.emitError("ttl-legality-check: annotated loop does not form "
                     "a valid perfectly nested band");
      return failure();
    }

    auto tileAttr = root->getAttrOfType<ArrayAttr>("ttl.tile");
    if (!tileAttr || tileAttr.empty()) {
      root.emitError("ttl-legality-check: ttl.tile attribute is empty");
      return failure();
    }

    for (auto a : tileAttr) {
      auto intAttr = dyn_cast<IntegerAttr>(a);
      if (!intAttr || intAttr.getInt() < 0) {
        root.emitError(
            "ttl-legality-check: tile sizes must be non-negative integers");
        return failure();
      }
    }

    if (tileAttr.size() > band.size()) {
      root.emitError("ttl-legality-check: ")
          << tileAttr.size() << " tile sizes provided but loop band has only "
          << band.size() << " loops";
      return failure();
    }

    return success();
  }

  //===--------------------------------------------------------------------===//
  // Copy promotion region checks
  //===--------------------------------------------------------------------===//

  LogicalResult checkCopyRegion(affine::AffineForOp loop) {
    auto fn = loop->getParentOfType<func::FuncOp>();
    if (!fn) {
      loop.emitError(
          "ttl-legality-check: ttl.copy on loop not in a function");
      return failure();
    }
    if (!fn->hasAttr("ttl.kernel")) {
      loop.emitError("ttl-legality-check: ttl.copy on loop inside "
                     "non-kernel function");
      return failure();
    }

    auto copyAttr = loop->getAttrOfType<ArrayAttr>("ttl.copy");
    if (!copyAttr)
      return success();

    // Collect known tensor names from function metadata.
    llvm::StringSet<> tensorNames;
    for (unsigned i = 0, e = fn.getNumArguments(); i < e; ++i)
      if (auto dict = fn.getArgAttrOfType<DictionaryAttr>(i, "ttl.tensor"))
        tensorNames.insert(dict.getAs<StringAttr>("name").getValue());

    // Track per-tensor: mode and scope seen, to detect duplicates and conflicts.
    llvm::StringMap<std::pair<StringRef, StringRef>> seenTensors;

    for (auto attr : copyAttr) {
      auto dict = cast<DictionaryAttr>(attr);
      StringRef name = dict.getAs<StringAttr>("name").getValue();

      if (!tensorNames.count(name)) {
        loop.emitError("ttl-legality-check: copy promotion directive for '")
            << name << "' but no matching ttl.tensor annotation";
        return failure();
      }

      StringRef scope = dict.getAs<StringAttr>("scope").getValue();
      if (scope != "local") {
        loop.emitError("ttl-legality-check: copy scope must be 'local', "
                       "got '") << scope << "' — private is not implemented";
        return failure();
      }

      StringRef mode = dict.getAs<StringAttr>("mode").getValue();

      auto it = seenTensors.find(name);
      if (it != seenTensors.end()) {
        StringRef prevMode = it->second.first;
        StringRef prevScope = it->second.second;

        // Exact duplicate.
        if (prevMode == mode && prevScope == scope) {
          loop.emitError("ttl-legality-check: duplicate copy directive "
                         "for tensor '") << name << "'";
          return failure();
        }

        // Conflicting mode (read + write on same tensor).
        if (prevMode != mode) {
          loop.emitError("ttl-legality-check: conflicting copy directives "
                         "for tensor '") << name
              << "' (both '" << prevMode << "' and '" << mode
              << "'). Use one directive per tensor.";
          return failure();
        }

        // Conflicting scope.
        if (prevScope != scope) {
          loop.emitError("ttl-legality-check: conflicting scopes for "
                         "tensor '") << name << "' ('" << prevScope
              << "' and '" << scope << "')";
          return failure();
        }
      }

      seenTensors[name] = {mode, scope};
    }

    // Non-affine access check.
    bool hasNonAffineAccess = false;
    loop.walk([&](memref::LoadOp) { hasNonAffineAccess = true; });
    loop.walk([&](memref::StoreOp) { hasNonAffineAccess = true; });
    if (hasNonAffineAccess) {
      loop.emitError("ttl-legality-check: non-affine memory access "
                     "(memref.load/store) in copy-annotated region");
      return failure();
    }

    return success();
  }

  //===--------------------------------------------------------------------===//
  // Pipeline region checks
  //===--------------------------------------------------------------------===//

  LogicalResult checkPipelineRegion(affine::AffineForOp loop) {
    auto fn = loop->getParentOfType<func::FuncOp>();
    if (!fn || !fn->hasAttr("ttl.kernel")) {
      loop.emitError("ttl-legality-check: ttl.pipeline on loop inside "
                     "non-kernel function");
      return failure();
    }

    // Pipeline requires copy promotion (DMA ops must exist after copy-gen).
    if (!loop->hasAttr("ttl.copy")) {
      loop.emitError("ttl-legality-check: ttl.pipeline requires copy "
                     "promotion (promote_read/promote_write) on the "
                     "same loop");
      return failure();
    }

    // Pipeline requires tiling (double-buffering needs a multi-iteration loop).
    if (!loop->hasAttr("ttl.tile")) {
      loop.emitError("ttl-legality-check: ttl.pipeline requires tiling "
                     "(ttl.tile) on the same loop");
      return failure();
    }

    return success();
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createTTLLegalityCheckPass() {
  return std::make_unique<TTLLegalityCheck>();
}
} // namespace mlir
