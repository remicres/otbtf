if(OTB_USE_TENSORFLOW)
  # Tensorflow-dependent APPS
  OTB_CREATE_APPLICATION(NAME TensorflowModelServe
	SOURCES otbTensorflowModelServe.cxx ${${otb-module}_SYSTEM_INCLUDE_DIRS}
	LINK_LIBRARIES ${${otb-module}_LIBRARIES}
  )

  OTB_CREATE_APPLICATION(NAME TensorflowModelTrain
	SOURCES otbTensorflowModelTrain.cxx ${${otb-module}_SYSTEM_INCLUDE_DIRS}
	LINK_LIBRARIES ${${otb-module}_LIBRARIES}
  )

  OTB_CREATE_APPLICATION(NAME TrainClassifierFromDeepFeatures
	SOURCES otbTrainClassifierFromDeepFeatures.cxx ${${otb-module}_SYSTEM_INCLUDE_DIRS}
	LINK_LIBRARIES ${${otb-module}_LIBRARIES}
  )
  
  OTB_CREATE_APPLICATION(NAME ImageClassifierFromDeepFeatures
	SOURCES otbImageClassifierFromDeepFeatures.cxx ${${otb-module}_SYSTEM_INCLUDE_DIRS}
	LINK_LIBRARIES ${${otb-module}_LIBRARIES}
  )
endif()

# Tensorflow-independent APPS
OTB_CREATE_APPLICATION(NAME PatchesExtraction
	SOURCES otbPatchesExtraction.cxx
	LINK_LIBRARIES ${OTBCOMMON_LIBRARIES}
)
OTB_CREATE_APPLICATION(NAME LabelImageSampleSelection
	SOURCES otbLabelImageSampleSelection.cxx
	LINK_LIBRARIES ${OTBCOMMON_LIBRARIES}
)
OTB_CREATE_APPLICATION(NAME DensePolygonClassStatistics
	SOURCES otbDensePolygonClassStatistics.cxx
	LINK_LIBRARIES ${OTBCOMMON_LIBRARIES}
)
