set(DOCUMENTATION "An unofficial module enabling to play with Tensorflow")

# define the dependencies of the include module and the tests
otb_module(OTBTensorflow
	DEPENDS
		OTBCommon
		OTBApplicationEngine
		OTBStreaming
		OTBExtendedFilename
		OTBImageIO
		OTBSupervised
	TEST_DEPENDS
		OTBTestKernel
		OTBCommandLine
	DESCRIPTION
		"${DOCUMENTATION}"
)
