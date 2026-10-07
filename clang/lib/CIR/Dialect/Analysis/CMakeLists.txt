add_clang_library(MLIRCIRAnalysis
  CIRAliasAnalysis.cpp
  CIRBasicAliasAnalysis.cpp
  CIRPointerDecomposition.cpp
  CIRPointsToAliasAnalysis.cpp
  PointsTo/GlobalNodes.cpp
  PointsTo/LocalBuilder.cpp
  PointsTo/OpClassification.cpp
  PointsTo/PointsToAnalysis.cpp
  PointsTo/PointsToGraph.cpp
  PointsTo/PointsToModels.cpp
  PointsTo/PointsToQueries.cpp
  PointsTo/PointsToVerifier.cpp

  LINK_LIBS PUBLIC
  MLIRCIR
  MLIRIR
  MLIRAnalysis
  MLIRSideEffectInterfaces
)
