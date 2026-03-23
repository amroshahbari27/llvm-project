add_mlir_library(MLIRTTLTransforms
  TTLCopyGenerate.cpp
  TTLLegalityCheck.cpp
  TTLLowerCopies.cpp
  TTLPipeline.cpp
  TTLPipelineDataTransfer.cpp
  TTLPromoteAccumulator.cpp
  TTLTilePass.cpp
  TTLToEmitC.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineToStandard
  MLIRAffineTransforms
  MLIRAffineUtils
  MLIRArithDialect
  MLIRArithToEmitC
  MLIRArithTransforms
  MLIRConvertToEmitC
  MLIREmitCDialect
  MLIRFuncDialect
  MLIRFuncToEmitC
  MLIRMathToEmitC
  MLIRMemRefDialect
  MLIRMemRefToEmitC
  MLIRPass
  MLIRSCFDialect
  MLIRSCFToEmitC
  MLIRSupport
  MLIRTransforms
  MLIRTTLDialect
  )
