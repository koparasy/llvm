add_clang_library(CIRTestAnalysis
  TestCIRAliasAnalysis.cpp
  TestKernelBindings.cpp
  TestCopyKernel.cpp

  LINK_LIBS PUBLIC
  MLIRCIRAnalysis
  MLIRCIR
  MLIRCIROffloadOpt
  MLIRIR
  MLIRAnalysis
  MLIRPass
  MLIRTestAliasAnalysis
)

target_include_directories(CIRTestAnalysis
  PRIVATE
  ${LLVM_MAIN_SRC_DIR}/..
)
