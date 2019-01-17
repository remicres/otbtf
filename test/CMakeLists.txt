otb_module_test()

# Directories
set(DATADIR ${CMAKE_CURRENT_SOURCE_DIR}/data)
set(MODELSDIR ${CMAKE_CURRENT_SOURCE_DIR}/models)

# Input images
set(IMAGEXS ${DATADIR}/xs_subset.tif)
set(IMAGEPAN ${DATADIR}/pan_subset.tif)
set(IMAGEPXS ${DATADIR}/pxs_subset.tif)

# Input models
set(MODEL1 ${MODELSDIR}/model1)
set(MODEL2 ${MODELSDIR}/model2)
set(MODEL3 ${MODELSDIR}/model3)

# Output images and baselines
set(MODEL1_PB_OUT apTvClTensorflowModelServeCNN16x16PB.tif)
set(MODEL2_PB_OUT apTvClTensorflowModelServeCNN8x8_32x32PB.tif)
set(MODEL2_FC_OUT apTvClTensorflowModelServeCNN8x8_32x32FC.tif)
set(MODEL3_PB_OUT apTvClTensorflowModelServeFCNN16x16PB.tif)
set(MODEL3_FC_OUT apTvClTensorflowModelServeFCNN16x16FC.tif)

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
set_tests_properties(apTvClTensorflowModelServeCNN8x8_32x32PB PROPERTIES ENVIRONMENT "OTB_TF_NSOURCES=2;$ENV{OTB_TF_NSOURCES}")

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
set_tests_properties(apTvClTensorflowModelServeCNN8x8_32x32FC PROPERTIES ENVIRONMENT "OTB_TF_NSOURCES=2;$ENV{OTB_TF_NSOURCES}")

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


