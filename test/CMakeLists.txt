otb_module_test()

#set(IMAGEXS data/xs_subset.tif)
#set(IMAGEPAN data/pan_subset.tif)
#set(IMAGEPXS data/pxs_subset.tif)
#set(MODEL1 models/model1)
#set(MODEL2 models/model2)
#set(MODEL3 models/model3)

#----------- Model serving : 1-branch CNN (16x16) Patch-Based ----------------
set(ENV{OTB_TF_NSOURCES} 1)
otb_test_application(NAME apTvClTensorflowModelServeCNN16x16
  APP  TensorflowModelServe
  OPTIONS -source1.il ${INPUTDATA}/${IMAGEXS}
  -source1.rfiedx 16 -source1.rfieldy 16 -source1.placeholder x
  -model.dir ${TEMP}/${MODEL1} -output.names y
  -out ${TEMP}/${MODEL1_PB_OUT}
  VALID --compare-image ${EPSILON_6}
  ${INPUTDATA}/${MODEL1_PB_REF}
  ${TEMP}/${MODEL1_PB_OUT})

#----------- Model serving : 2-branch CNN (8x8, 32x32) Patch-Based ----------------
set(ENV{OTB_TF_NSOURCES} 2)
otb_test_application(NAME apTvClTensorflowModelServeCNN8x8_32x32
  APP  TensorflowModelServe
  OPTIONS -source1.il ${INPUTDATA}/${IMAGEXS}
  -source1.rfiedx 8 -source1.rfieldy 8 -source1.placeholder x1
  -source2.il ${INPUTDATA}/${IMAGEPAN}
  -source2.rfiedx 32 -source2.rfieldy 32 -source2.placeholder x2
  -model.dir ${TEMP}/${MODEL2} -output.names y
  -out ${TEMP}/${MODEL2_PB_OUT}
  VALID --compare-image ${EPSILON_6}
  ${INPUTDATA}/${MODEL2_PB_REF}
  ${TEMP}/${MODEL2_PB_OUT})

#----------- Model serving : 2-branch CNN (8x8, 32x32) Fully-Conv ----------------
set(ENV{OTB_TF_NSOURCES} 2)
otb_test_application(NAME apTvClTensorflowModelServeCNN8x8_32x32
  APP  TensorflowModelServe
  OPTIONS -source1.il ${INPUTDATA}/${IMAGEXS}
  -source1.rfiedx 8 -source1.rfieldy 8 -source1.placeholder x1
  -source2.il ${INPUTDATA}/${IMAGEPAN}
  -source2.rfiedx 32 -source2.rfieldy 32 -source2.placeholder x2
  -model.dir ${TEMP}/${MODEL2} -output.names y -output.spcscale 4
  -out ${TEMP}/${MODEL2_FC_OUT}
  VALID --compare-image ${EPSILON_6}
  ${INPUTDATA}/${MODEL2_FC_REF}
  ${TEMP}/${MODEL2_FC_OUT})

#----------- Model serving : 1-branch FCNN (16x16) Patch-Based ----------------
set(ENV{OTB_TF_NSOURCES} 1)
otb_test_application(NAME apTvClTensorflowModelServeFCNN16x16PB
  APP  TensorflowModelServe
  OPTIONS -source1.il ${INPUTDATA}/${IMAGEPXS}
  -source1.rfiedx 16 -source1.rfieldy 16 -source1.placeholder x
  -model.dir ${TEMP}/${MODEL3} -output.names y
  -out ${TEMP}/${MODEL3_PB_OUT}
  VALID --compare-image ${EPSILON_6}
  ${INPUTDATA}/${MODEL3_PB_REF}
  ${TEMP}/${MODEL3_PB_OUT})

#----------- Model serving : 1-branch FCNN (16x16) Fully-conv ----------------
set(ENV{OTB_TF_NSOURCES} 1)
otb_test_application(NAME apTvClTensorflowModelServeFCNN16x16FC
  APP  TensorflowModelServe
  OPTIONS -source1.il ${INPUTDATA}/${IMAGEXS}
  -source1.rfiedx 16 -source1.rfieldy 16 -source1.placeholder x
  -model.dir ${TEMP}/${MODEL3} -output.names y -model.fullyconv on
  -out ${TEMP}/${MODEL3_FC_OUT}
  VALID --compare-image ${EPSILON_6}
  ${INPUTDATA}/${MODEL3_FC_REF}
  ${TEMP}/${MODEL3_FC_OUT})


