otb_module_test()

# Unit tests
set(${otb-module}Tests 
  otbTensorflowTests.cxx
  otbTensorflowCopyUtilsTests.cxx)

add_executable(otbTensorflowTests ${${otb-module}Tests})

target_include_directories(otbTensorflowTests PRIVATE ${tensorflow_include_dir})
target_link_libraries(otbTensorflowTests ${${otb-module}-Test_LIBRARIES} ${TENSORFLOW_CC_LIB} ${TENSORFLOW_FRAMEWORK_LIB})
otb_module_target_label(otbTensorflowTests)

# CopyUtilsTests
otb_add_test(NAME floatValueToTensorTest COMMAND otbTensorflowTests floatValueToTensorTest)
otb_add_test(NAME intValueToTensorTest COMMAND otbTensorflowTests intValueToTensorTest)
otb_add_test(NAME boolValueToTensorTest COMMAND otbTensorflowTests boolValueToTensorTest)
otb_add_test(NAME floatVecValueToTensorTest COMMAND otbTensorflowTests floatVecValueToTensorTest)
otb_add_test(NAME intVecValueToTensorTest COMMAND otbTensorflowTests intVecValueToTensorTest)
otb_add_test(NAME boolVecValueToTensorTest COMMAND otbTensorflowTests boolVecValueToTensorTest)

# Directories
set(DATADIR ${CMAKE_CURRENT_SOURCE_DIR}/data)
set(MODELSDIR ${CMAKE_CURRENT_SOURCE_DIR}/models)

# Input images
set(IMAGEXS ${DATADIR}/xs_subset.tif)
set(IMAGEPAN ${DATADIR}/pan_subset.tif)
set(IMAGEPXS ${DATADIR}/pxs_subset.tif)
set(IMAGEPXS2 ${DATADIR}/pxs_subset2.tif)
set(PATCHESA ${DATADIR}/Sentinel-2_B4328_10m_patches_A.jp2)
set(PATCHESB ${DATADIR}/Sentinel-2_B4328_10m_patches_B.jp2)
set(LABELSA ${DATADIR}/Sentinel-2_B4328_10m_labels_A.tif)
set(LABELSB ${DATADIR}/Sentinel-2_B4328_10m_labels_B.tif)
set(PATCHES01 ${DATADIR}/Sentinel-2_B4328_10m_patches_A.jp2)
set(PATCHES11 ${DATADIR}/Sentinel-2_B4328_10m_patches_B.jp2)

# Input models
set(MODEL1 ${MODELSDIR}/model1)
set(MODEL2 ${MODELSDIR}/model2)
set(MODEL3 ${MODELSDIR}/model3)
set(MODEL4 ${MODELSDIR}/model4)

# Output images and baselines
set(MODEL1_PB_OUT apTvClTensorflowModelServeCNN16x16PB.tif)
set(MODEL2_PB_OUT apTvClTensorflowModelServeCNN8x8_32x32PB.tif)
set(MODEL2_FC_OUT apTvClTensorflowModelServeCNN8x8_32x32FC.tif)
set(MODEL3_PB_OUT apTvClTensorflowModelServeFCNN16x16PB.tif)
set(MODEL3_FC_OUT apTvClTensorflowModelServeFCNN16x16FC.tif)
set(MODEL4_FC_OUT apTvClTensorflowModelServeFCNN64x64to32x32.tif)
set(MODEL1_SAVED model1_updated)
set(PATCHESIMG_01 patchimg_01.tif)
set(PATCHESIMG_11 patchimg_11.tif)

#----------- Patches selection ----------------
set(PATCHESPOS_01 ${TEMP}/out_train_32.gpkg)
set(PATCHESPOS_02 ${TEMP}/out_valid_32.gpkg)
set(PATCHESPOS_11 ${TEMP}/out_train_33.gpkg)
set(PATCHESPOS_12 ${TEMP}/out_valid_33.gpkg)
# Even patches
otb_test_application(NAME PatchesSelectionEven
  APP  PatchesSelection
  OPTIONS 
  -in ${IMAGEPXS2}
  -grid.step 32
  -grid.psize 32
  -outtrain ${PATCHESPOS_01}
  -outvalid ${PATCHESPOS_02}
  )
  
# Odd patches
otb_test_application(NAME PatchesSelectionOdd
  APP  PatchesSelection
  OPTIONS 
  -in ${IMAGEPXS2}
  -grid.step 32
  -grid.psize 33
  -outtrain ${PATCHESPOS_11}
  -outvalid ${PATCHESPOS_12}
  )

#----------- Patches extraction ----------------
# Even patches
otb_test_application(NAME PatchesExtractionEven
  APP  PatchesExtraction
  OPTIONS 
  -source1.il ${IMAGEPXS2}
  -source1.patchsizex 32
  -source1.patchsizey 32
  -source1.out ${PATCHESIMG_01}
  -vec ${PATCHESPOS_01}
  -field id
  VALID --compare-image ${EPSILON_6}
  ${DATADIR}/${PATCHESIMG_01}
  ${TEMP}/${PATCHESIMG_01}
  )

# Odd patches
otb_test_application(NAME PatchesExtractionOdd
  APP  PatchesExtraction
  OPTIONS 
  -source1.il ${IMAGEPXS2}
  -source1.patchsizex 33
  -source1.patchsizey 33
  -source1.out ${PATCHESIMG_11}
  -vec ${PATCHESPOS_11}
  -field id
  VALID --compare-image ${EPSILON_6}
  ${DATADIR}/${PATCHESIMG_11}
  ${TEMP}/${PATCHESIMG_11}
  )

#----------- Model training : 1-branch CNN (16x16) Patch-Based ----------------
set(ENV{OTB_LOGGER_LEVEL} DEBUG)
otb_test_application(NAME TensorflowModelTrainCNN16x16PB
  APP  TensorflowModelTrain
  OPTIONS 
  -training.epochs 10
  -training.source1.il ${PATCHESA}
  -training.source1.placeholder "x"
  -training.source2.il ${LABELSA}
  -training.source2.placeholder "y"
  -validation.source1.il ${PATCHESB}
  -validation.source2.il ${LABELSB}
  -validation.source1.name "x"
  -validation.source2.name "prediction"
  -training.source1.patchsizex 16 -training.source1.patchsizey 16
  -training.source2.patchsizex 1 -training.source2.patchsizey 1
  -model.dir ${MODEL1} 
  -model.saveto ${MODEL1_SAVED} 
  -training.targetnodes "optimizer"
  -validation.mode "class"
  )
set_tests_properties(TensorflowModelTrainCNN16x16PB PROPERTIES ENVIRONMENT "OTB_LOGGER_LEVEL=DEBUG;$ENV{OTB_LOGGER_LEVEL}")

#----------- Model serving : 1-branch CNN (16x16) Patch-Based ----------------
otb_test_application(NAME TensorflowModelServeCNN16x16PB
  APP  TensorflowModelServe
  OPTIONS -source1.il ${IMAGEPXS}
  -source1.rfieldx 16 -source1.rfieldy 16 -source1.placeholder x
  -model.dir ${MODEL1} -output.names prediction
  -out ${TEMP}/${MODEL1_PB_OUT}
  VALID --compare-image ${EPSILON_6}
  ${DATADIR}/${MODEL1_PB_OUT}
  ${TEMP}/${MODEL1_PB_OUT})
set_tests_properties(TensorflowModelServeCNN16x16PB PROPERTIES ENVIRONMENT "OTB_LOGGER_LEVEL=DEBUG}")

#----------- Model serving : 2-branch CNN (8x8, 32x32) Patch-Based ----------------
otb_test_application(NAME apTvClTensorflowModelServeCNN8x8_32x32PB
  APP  TensorflowModelServe
  OPTIONS -source1.il ${IMAGEXS}
  -source1.rfieldx 8 -source1.rfieldy 8 -source1.placeholder x1
  -source2.il ${IMAGEPAN}
  -source2.rfieldx 32 -source2.rfieldy 32 -source2.placeholder x2
  -model.dir ${MODEL2} -output.names prediction
  -out ${TEMP}/${MODEL2_PB_OUT}
  VALID --compare-image ${EPSILON_6}
  ${DATADIR}/${MODEL2_PB_OUT}
  ${TEMP}/${MODEL2_PB_OUT})
set_tests_properties(apTvClTensorflowModelServeCNN8x8_32x32PB PROPERTIES ENVIRONMENT "OTB_LOGGER_LEVEL=DEBUG;OTB_TF_NSOURCES=2;$ENV{OTB_TF_NSOURCES}")


#----------- Model serving : 2-branch CNN (8x8, 32x32) Fully-Conv ----------------
set(ENV{OTB_TF_NSOURCES} 2)
otb_test_application(NAME apTvClTensorflowModelServeCNN8x8_32x32FC
  APP  TensorflowModelServe
  OPTIONS -source1.il ${IMAGEXS}
  -source1.rfieldx 8 -source1.rfieldy 8 -source1.placeholder x1
  -source2.il ${IMAGEPAN}
  -source2.rfieldx 32 -source2.rfieldy 32 -source2.placeholder x2
  -model.dir ${MODEL2} -output.names prediction -output.spcscale 4
  -out ${TEMP}/${MODEL2_FC_OUT}
  VALID --compare-image ${EPSILON_6}
  ${DATADIR}/${MODEL2_FC_OUT}
  ${TEMP}/${MODEL2_FC_OUT})
set_tests_properties(apTvClTensorflowModelServeCNN8x8_32x32FC PROPERTIES ENVIRONMENT "OTB_LOGGER_LEVEL=DEBUG;OTB_TF_NSOURCES=2;$ENV{OTB_TF_NSOURCES}")

#----------- Model serving : 1-branch FCNN (16x16) Patch-Based ----------------
set(ENV{OTB_TF_NSOURCES} 1)
otb_test_application(NAME apTvClTensorflowModelServeFCNN16x16PB
  APP  TensorflowModelServe
  OPTIONS -source1.il ${IMAGEPXS}
  -source1.rfieldx 16 -source1.rfieldy 16 -source1.placeholder x
  -model.dir ${MODEL3} -output.names prediction
  -out ${TEMP}/${MODEL3_PB_OUT}
  VALID --compare-image ${EPSILON_6}
  ${DATADIR}/${MODEL3_PB_OUT}
  ${TEMP}/${MODEL3_PB_OUT})
set_tests_properties(apTvClTensorflowModelServeFCNN16x16PB PROPERTIES ENVIRONMENT "OTB_LOGGER_LEVEL=DEBUG}")


#----------- Model serving : 1-branch FCNN (16x16) Fully-conv ----------------
set(ENV{OTB_TF_NSOURCES} 1)
otb_test_application(NAME apTvClTensorflowModelServeFCNN16x16FC
  APP  TensorflowModelServe
  OPTIONS -source1.il ${IMAGEPXS}
  -source1.rfieldx 16 -source1.rfieldy 16 -source1.placeholder x
  -model.dir ${MODEL3} -output.names prediction -model.fullyconv on
  -out ${TEMP}/${MODEL3_FC_OUT}
  VALID --compare-image ${EPSILON_6}
  ${DATADIR}/${MODEL3_FC_OUT}
  ${TEMP}/${MODEL3_FC_OUT})
set_tests_properties(apTvClTensorflowModelServeFCNN16x16FC PROPERTIES ENVIRONMENT "OTB_LOGGER_LEVEL=DEBUG}")

#----------- Model serving : 1-branch FCNN (64x64)-->(32x32), Fully-conv ----------------
set(ENV{OTB_TF_NSOURCES} 1)
otb_test_application(NAME apTvClTensorflowModelServeFCNN64x64to32x32
  APP  TensorflowModelServe
  OPTIONS -source1.il ${IMAGEPXS2}
  -source1.rfieldx 64 -source1.rfieldy 64 -source1.placeholder x
  -output.efieldx 32 -output.efieldy 32 -output.names prediction_fcn
  -model.dir ${MODEL4} -model.fullyconv on
  -out ${TEMP}/${MODEL4_FC_OUT}
  VALID --compare-image ${EPSILON_6}
  ${DATADIR}/${MODEL4_FC_OUT}
  ${TEMP}/${MODEL4_FC_OUT})
set_tests_properties(apTvClTensorflowModelServeFCNN64x64to32x32 PROPERTIES ENVIRONMENT "OTB_LOGGER_LEVEL=DEBUG}")


