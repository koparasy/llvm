add_clang_library(MLIRCIROffloadOpt
  KernelBindingTable.cpp
  InternalizeLaunchPaths.cpp
  KernelCloning.cpp
  LaunchPathInternalization.cpp
  SYCLKernelParams.cpp
  SYCLHostFacts.cpp
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
