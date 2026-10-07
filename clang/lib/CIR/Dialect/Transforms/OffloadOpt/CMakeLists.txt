add_clang_library(MLIRCIROffloadOpt
  KernelBindingTable.cpp
  KernelCloning.cpp
  SYCLKernelParams.cpp
  SYCLPointsToModels.cpp
  DeadKernelElimination.cpp
  KernelArgConstantPropagation.cpp
  LaunchBoundsPropagation.cpp
  StampReqdWorkGroupSize.cpp
  InferKernelParamNoAlias.cpp
  PropagateSYCLKernelArgs.cpp
  DetectHostUniformIntDivisor.cpp
  DetectHostDeviceClosureLayoutMismatch.cpp

  DEPENDS
  MLIRCIROpsIncGen
  MLIRCIRPassIncGen

  LINK_LIBS PUBLIC
  MLIRCIR
  MLIRCIRAnalysis
  MLIRCIRTransforms
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  )
