project(OTBTensorflow)

# Use Tensorflow, or not
option(OTB_USE_TENSORFLOW "Enable Tensorflow dependent applications" OFF)

if(OTB_USE_TENSORFLOW)
  message("Tensorflow support enabled")

  # find Tensorflow INCLUDE DIR
  set(tensorflow_include_dir "" CACHE PATH "The include directory of tensorflow")
  include_directories(${tensorflow_include_dir})

  # find Tensorflow LIBRARIES
  find_library(TENSORFLOW_CC_LIB NAMES libtensorflow_cc)
  find_library(TENSORFLOW_FRAMEWORK_LIB NAMES libtensorflow_framework)

  set(TENSORFLOW_LIBS "${TENSORFLOW_CC_LIB}" "${TENSORFLOW_FRAMEWORK_LIB}")

else()
  message("Tensorflow support disabled")
endif()

otb_module_impl()
