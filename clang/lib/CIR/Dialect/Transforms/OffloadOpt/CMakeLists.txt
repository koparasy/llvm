add_clang_library(MLIRCIROffloadOpt
  KernelBindingTable.cpp
  KernelCloning.cpp
  DeadKernelElimination.cpp
  KernelArgConstantPropagation.cpp
  LaunchBoundsPropagation.cpp
  StampReqdWorkGroupSize.cpp
  InferKernelParamNoAlias.cpp
  PropagateSYCLKernelArgs.cpp

  DEPENDS
  MLIRCIROpsIncGen
  MLIRCIRPassIncGen

  LINK_LIBS PUBLIC
  MLIRCIR
  MLIRCIRTransforms
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  )
