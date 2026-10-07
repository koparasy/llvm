add_clang_library(MLIRCIRAnalysis
  CIRAliasAnalysis.cpp
  CIRBasicAliasAnalysis.cpp
  CIRPointerDecomposition.cpp
  PointsTo/GlobalNodes.cpp
  PointsTo/LocalBuilder.cpp
  PointsTo/OpClassification.cpp
  PointsTo/PointsToGraph.cpp
  PointsTo/PointsToVerifier.cpp

  LINK_LIBS PUBLIC
  MLIRCIR
  MLIRIR
  MLIRAnalysis
  MLIRSideEffectInterfaces
)
