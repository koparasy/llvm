add_clang_library(MLIRCIRAnalysis
  CIRAliasAnalysis.cpp
  CIRBasicAliasAnalysis.cpp
  CIRPointerDecomposition.cpp
  CIRPointsToAliasAnalysis.cpp
  PointsTo/GlobalNodes.cpp
  PointsTo/LocalBuilder.cpp
  PointsTo/OpClassification.cpp
  PointsTo/PointsToAnalysis.cpp
  PointsTo/PointsToBottomUp.cpp
  PointsTo/PointsToGraph.cpp
  PointsTo/PointsToModels.cpp
  PointsTo/PointsToQueries.cpp
  PointsTo/PointsToTopDown.cpp
  PointsTo/PointsToVerifier.cpp
  PointsTo/StoredValues.cpp

  LINK_LIBS PUBLIC
  MLIRCIR
  MLIRIR
  MLIRAnalysis
  MLIRSideEffectInterfaces
)
