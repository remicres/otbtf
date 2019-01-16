otb_module_test()

#----------- Model serving : 1-branch CNN (16x16) ----------------
set(ENV{OTB_TF_NSOURCES} 1)
otb_test_application(NAME apTvClTensorflowModelServe1CNN16x16
  APP  TensorflowModelServe
  OPTIONS -source1.il ${INPUTDATA}/${IMAGEXS}
  -source1.rfiedx 16 -source1.rfieldy 16 -source1.placeholder x
  -model.dir ${TEMP}/${MODEL1} -output.names y
  -out ${TEMP}/${MODEL1OUT}
  VALID --compare-image ${EPSILON_6}
  ${INPUTDATA}/${MODEL1REF}
  ${TEMP}/${MODEL1OUT})

#----------- Model serving : 2-branch CNN (8x8, 32x32) ----------------
set(ENV{OTB_TF_NSOURCES} 2)
otb_test_application(NAME apTvClTensorflowModelServe1CNN16x16
  APP  TensorflowModelServe
  OPTIONS -source1.il ${INPUTDATA}/${IMAGEXS}
  -source1.rfiedx 8 -source1.rfieldy 8 -source1.placeholder x1
  -source2.rfiedx 32 -source2.rfieldy 32 -source2.placeholder x2
  -model.dir ${TEMP}/${MODEL2} -output.names y
  -out ${TEMP}/${MODEL2OUT}
  VALID --compare-image ${EPSILON_6}
  ${INPUTDATA}/${MODEL2REF}
  ${TEMP}/${MODEL2OUT})


