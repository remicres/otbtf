#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest
import unittest
from test_utils import run_command, run_command_and_test_exist, run_command_and_compare

INFERENCE_MAE_TOL = 10.0  # Dummy value: we don't really care of the mae value but rather the image size etc


class TutorialTest(unittest.TestCase):

    @pytest.mark.order(1)
    def test_sample_selection(self):
        self.assertTrue(
            run_command_and_test_exist(
                command="otbcli_LabelImageSampleSelection "
                        "-inref $DATADIR/terrain_truth_epsg32654_A.tif "
                        "-nodata 255 "
                        "-outvec $TMPDIR/outvec_A.gpkg",
                file_list=["$TMPDIR/outvec_A.gpkg"]))
        self.assertTrue(
            run_command_and_test_exist(
                command="otbcli_LabelImageSampleSelection "
                        "-inref $DATADIR/terrain_truth_epsg32654_B.tif "
                        "-nodata 255 "
                        "-outvec $TMPDIR/outvec_B.gpkg",
                file_list=["$TMPDIR/outvec_B.gpkg"]))

    @pytest.mark.order(2)
    def test_patches_extraction(self):
        self.assertTrue(
            run_command_and_compare(
                command="otbcli_PatchesExtraction "
                        "-source1.il $DATADIR/s2_stack.jp2 "
                        "-source1.out $TMPDIR/s2_patches_A.tif "
                        "-source1.patchsizex 16 "
                        "-source1.patchsizey 16 "
                        "-vec $TMPDIR/outvec_A.gpkg "
                        "-field class "
                        "-outlabels $TMPDIR/s2_labels_A.tif",
                to_compare_dict={"$DATADIR/s2_patches_A.tif": "$TMPDIR/s2_patches_A.tif",
                                 "$DATADIR/s2_labels_A.tif": "$TMPDIR/s2_labels_A.tif"}))
        self.assertTrue(
            run_command_and_compare(
                command="otbcli_PatchesExtraction "
                        "-source1.il $DATADIR/s2_stack.jp2 "
                        "-source1.out $TMPDIR/s2_patches_B.tif "
                        "-source1.patchsizex 16 "
                        "-source1.patchsizey 16 "
                        "-vec $TMPDIR/outvec_B.gpkg "
                        "-field class "
                        "-outlabels $TMPDIR/s2_labels_B.tif",
                to_compare_dict={"$DATADIR/s2_patches_B.tif": "$TMPDIR/s2_patches_B.tif",
                                 "$DATADIR/s2_labels_B.tif": "$TMPDIR/s2_labels_B.tif"}))

    @pytest.mark.order(3)
    def test_generate_model1(self):
        run_command("git clone https://github.com/remicres/otbtf_tutorials_resources.git "
                    "$TMPDIR/otbtf_tuto_repo")
        self.assertTrue(
            run_command_and_test_exist(
                command="python $TMPDIR/otbtf_tuto_repo/01_patch_based_classification/models/create_model1.py "
                        "$TMPDIR/model1",
                file_list=["$TMPDIR/model1/saved_model.pb"]))

    @pytest.mark.order(4)
    def test_model1_train(self):
        self.assertTrue(
            run_command_and_test_exist(
                command="otbcli_TensorflowModelTrain "
                        "-training.source1.il $DATADIR/s2_patches_A.tif "
                        "-training.source1.patchsizex 16 "
                        "-training.source1.patchsizey 16 "
                        "-training.source1.placeholder x "
                        "-training.source2.il $DATADIR/s2_labels_A.tif "
                        "-training.source2.patchsizex 1 "
                        "-training.source2.patchsizey 1 "
                        "-training.source2.placeholder y "
                        "-model.dir $TMPDIR/model1 "
                        "-training.targetnodes optimizer "
                        "-training.epochs 10 "
                        "-validation.mode class "
                        "-validation.source1.il $DATADIR/s2_patches_B.tif "
                        "-validation.source1.name x "
                        "-validation.source2.il $DATADIR/s2_labels_B.tif "
                        "-validation.source2.name prediction "
                        "-model.saveto $TMPDIR/model1/variables/variables",
                file_list=["$TMPDIR/model1/variables/variables.index"]
            )
        )

    @pytest.mark.order(5)
    def test_model1_inference_pb(self):
        self.assertTrue(
            run_command_and_compare(
                command="otbcli_TensorflowModelServe "
                        "-source1.il $DATADIR/s2_stack.jp2 "
                        "-source1.rfieldx 16 "
                        "-source1.rfieldy 16 "
                        "-source1.placeholder x "
                        "-model.dir $TMPDIR/model1 "
                        "-output.names prediction "
                        "-out \"$TMPDIR/classif_model1.tif?&box=4000:4000:1000:1000\" uint8",
                to_compare_dict={"$DATADIR/classif_model1.tif": "$TMPDIR/classif_model1.tif"},
                tol=INFERENCE_MAE_TOL))

    @pytest.mark.order(6)
    def test_model1_inference_fcn(self):
        self.assertTrue(
            run_command_and_compare(
                command="otbcli_TensorflowModelServe "
                        "-source1.il $DATADIR/s2_stack.jp2 "
                        "-source1.rfieldx 16 "
                        "-source1.rfieldy 16 "
                        "-source1.placeholder x "
                        "-model.dir $TMPDIR/model1 "
                        "-output.names prediction "
                        "-model.fullyconv on "
                        "-output.spcscale 4 "
                        "-out \"$TMPDIR/classif_model1.tif?&box=1000:1000:256:256\" uint8",
                to_compare_dict={"$DATADIR/classif_model1.tif": "$TMPDIR/classif_model1.tif"},
                tol=INFERENCE_MAE_TOL))

    @pytest.mark.order(7)
    def test_rf_sampling(self):
        self.assertTrue(
            run_command_and_test_exist(
                command="otbcli_SampleExtraction "
                        "-in $DATADIR/s2_stack.jp2 "
                        "-vec $TMPDIR/outvec_A.gpkg "
                        "-field class "
                        "-out $TMPDIR/pixelvalues_A.gpkg",
                file_list=["$TMPDIR/pixelvalues_A.gpkg"]))
        self.assertTrue(
            run_command_and_test_exist(
                command="otbcli_SampleExtraction "
                        "-in $DATADIR/s2_stack.jp2 "
                        "-vec $TMPDIR/outvec_B.gpkg "
                        "-field class "
                        "-out $TMPDIR/pixelvalues_B.gpkg",
                file_list=["$TMPDIR/pixelvalues_B.gpkg"]))

    @pytest.mark.order(8)
    def test_rf_training(self):
        self.assertTrue(
            run_command_and_test_exist(
                command="otbcli_TrainVectorClassifier "
                        "-io.vd $TMPDIR/pixelvalues_A.gpkg "
                        "-valid.vd $TMPDIR/pixelvalues_B.gpkg "
                        "-feat value_0 value_1 value_2 value_3 "
                        "-cfield class "
                        "-classifier rf "
                        "-io.out $TMPDIR/randomforest_model.yaml ",
                file_list=["$TMPDIR/randomforest_model.yaml"]))

    @pytest.mark.order(9)
    def test_generate_model2(self):
        self.assertTrue(
            run_command_and_test_exist(
                command="python $TMPDIR/otbtf_tuto_repo/01_patch_based_classification/models/create_model2.py "
                        "$TMPDIR/model2",
                file_list=["$TMPDIR/model2/saved_model.pb"]))

    @pytest.mark.order(10)
    def test_model2_train(self):
        self.assertTrue(
            run_command_and_test_exist(
                command="otbcli_TensorflowModelTrain "
                        "-training.source1.il $DATADIR/s2_patches_A.tif "
                        "-training.source1.patchsizex 16 "
                        "-training.source1.patchsizey 16 "
                        "-training.source1.placeholder x "
                        "-training.source2.il $DATADIR/s2_labels_A.tif "
                        "-training.source2.patchsizex 1 "
                        "-training.source2.patchsizey 1 "
                        "-training.source2.placeholder y "
                        "-model.dir $TMPDIR/model2 "
                        "-training.targetnodes optimizer "
                        "-training.epochs 10 "
                        "-validation.mode class "
                        "-validation.source1.il $DATADIR/s2_patches_B.tif "
                        "-validation.source1.name x "
                        "-validation.source2.il $DATADIR/s2_labels_B.tif "
                        "-validation.source2.name prediction "
                        "-model.saveto $TMPDIR/model2/variables/variables",
                file_list=["$TMPDIR/model2/variables/variables.index"]))

    @pytest.mark.order(11)
    def test_model2_inference_fcn(self):
        self.assertTrue(
            run_command_and_compare(command="otbcli_TensorflowModelServe "
                                            "-source1.il $DATADIR/s2_stack.jp2 "
                                            "-source1.rfieldx 16 "
                                            "-source1.rfieldy 16 "
                                            "-source1.placeholder x "
                                            "-model.dir $TMPDIR/model2 "
                                            "-model.fullyconv on "
                                            "-output.names prediction "
                                            "-out \"$TMPDIR/classif_model2.tif?&box=4000:4000:1000:1000\" uint8",
                                    to_compare_dict={"$DATADIR/classif_model2.tif": "$TMPDIR/classif_model2.tif"},
                                    tol=INFERENCE_MAE_TOL))

    @pytest.mark.order(12)
    def test_model2rf_train(self):
        self.assertTrue(
            run_command_and_test_exist(
                command="otbcli_TrainClassifierFromDeepFeatures "
                        "-source1.il $DATADIR/s2_stack.jp2 "
                        "-source1.rfieldx 16 "
                        "-source1.rfieldy 16 "
                        "-source1.placeholder x "
                        "-model.dir $TMPDIR/model2 "
                        "-model.fullyconv on "
                        "-optim.tilesizex 999999 "
                        "-optim.tilesizey 128 "
                        "-output.names features "
                        "-vd $TMPDIR/outvec_A.gpkg "
                        "-valid $TMPDIR/outvec_B.gpkg "
                        "-sample.vfn class "
                        "-sample.bm 0 "
                        "-classifier rf "
                        "-out $TMPDIR/RF_model_from_deep_features.yaml",
                file_list=["$TMPDIR/RF_model_from_deep_features.yaml"]))

    @pytest.mark.order(13)
    def test_model2rf_inference(self):
        self.assertTrue(
            run_command_and_compare(
                command="otbcli_ImageClassifierFromDeepFeatures "
                        "-source1.il $DATADIR/s2_stack.jp2 "
                        "-source1.rfieldx 16 "
                        "-source1.rfieldy 16 "
                        "-source1.placeholder x "
                        "-deepmodel.dir $TMPDIR/model2 "
                        "-deepmodel.fullyconv on "
                        "-output.names features "
                        "-model $TMPDIR/RF_model_from_deep_features.yaml "
                        "-out \"$TMPDIR/RF_model_from_deep_features_map.tif?&box=4000:4000:1000:1000\" uint8",
                to_compare_dict={
                    "$DATADIR/RF_model_from_deep_features_map.tif": "$TMPDIR/RF_model_from_deep_features_map.tif"},
                tol=INFERENCE_MAE_TOL))

    @pytest.mark.order(14)
    def test_patch_extraction_20m(self):
        self.assertTrue(
            run_command_and_compare(
                command="OTB_TF_NSOURCES=2 otbcli_PatchesExtraction "
                        "-source1.il $DATADIR/s2_20m_stack.jp2 "
                        "-source1.patchsizex 8 "
                        "-source1.patchsizey 8 "
                        "-source1.out $TMPDIR/s2_20m_patches_A.tif "
                        "-source2.il $DATADIR/s2_stack.jp2 "
                        "-source2.patchsizex 16 "
                        "-source2.patchsizey 16 "
                        "-source2.out $TMPDIR/s2_10m_patches_A.tif "
                        "-vec $TMPDIR/outvec_A.gpkg "
                        "-field class "
                        "-outlabels $TMPDIR/s2_10m_labels_A.tif uint8",
                to_compare_dict={"$DATADIR/s2_10m_labels_A.tif": "$TMPDIR/s2_10m_labels_A.tif",
                                 "$DATADIR/s2_10m_patches_A.tif": "$TMPDIR/s2_10m_patches_A.tif",
                                 "$DATADIR/s2_20m_patches_A.tif": "$TMPDIR/s2_20m_patches_A.tif"}))
        self.assertTrue(
            run_command_and_compare(
                command="OTB_TF_NSOURCES=2 otbcli_PatchesExtraction "
                        "-source1.il $DATADIR/s2_20m_stack.jp2 "
                        "-source1.patchsizex 8 "
                        "-source1.patchsizey 8 "
                        "-source1.out $TMPDIR/s2_20m_patches_B.tif "
                        "-source2.il $DATADIR/s2_stack.jp2 "
                        "-source2.patchsizex 16 "
                        "-source2.patchsizey 16 "
                        "-source2.out $TMPDIR/s2_10m_patches_B.tif "
                        "-vec $TMPDIR/outvec_B.gpkg "
                        "-field class "
                        "-outlabels $TMPDIR/s2_10m_labels_B.tif uint8",
                to_compare_dict={"$DATADIR/s2_10m_labels_B.tif": "$TMPDIR/s2_10m_labels_B.tif",
                                 "$DATADIR/s2_10m_patches_B.tif": "$TMPDIR/s2_10m_patches_B.tif",
                                 "$DATADIR/s2_20m_patches_B.tif": "$TMPDIR/s2_20m_patches_B.tif"}))

    @pytest.mark.order(15)
    def test_generate_model3(self):
        self.assertTrue(
            run_command_and_test_exist(
                command="python $TMPDIR/otbtf_tuto_repo/01_patch_based_classification/models/create_model3.py "
                        "$TMPDIR/model3",
                file_list=["$TMPDIR/model3/saved_model.pb"]))

    @pytest.mark.order(16)
    def test_model3_train(self):
        self.assertTrue(
            run_command_and_test_exist(
                command="OTB_TF_NSOURCES=2 otbcli_TensorflowModelTrain "
                        "-training.source1.il $DATADIR/s2_20m_patches_A.tif "
                        "-training.source1.patchsizex 8 "
                        "-training.source1.patchsizey 8 "
                        "-training.source1.placeholder x1 "
                        "-training.source2.il $DATADIR/s2_10m_patches_A.tif "
                        "-training.source2.patchsizex 16 "
                        "-training.source2.patchsizey 16 "
                        "-training.source2.placeholder x2 "
                        "-training.source3.il $DATADIR/s2_10m_labels_A.tif "
                        "-training.source3.patchsizex 1 "
                        "-training.source3.patchsizey 1 "
                        "-training.source3.placeholder y "
                        "-model.dir $TMPDIR/model3 "
                        "-training.targetnodes optimizer "
                        "-training.epochs 10 "
                        "-validation.mode class "
                        "-validation.source1.il $DATADIR/s2_20m_patches_B.tif "
                        "-validation.source1.name x1 "
                        "-validation.source2.il $DATADIR/s2_10m_patches_B.tif "
                        "-validation.source2.name x2 "
                        "-validation.source3.il $DATADIR/s2_10m_labels_B.tif "
                        "-validation.source3.name prediction "
                        "-model.saveto $TMPDIR/model3/variables/variables",
                file_list=["$TMPDIR/model3/variables/variables.index"]))

    @pytest.mark.order(17)
    def test_model3_inference_pb(self):
        self.assertTrue(
            run_command_and_compare(
                command=
                "OTB_TF_NSOURCES=2 otbcli_TensorflowModelServe "
                "-source1.il $DATADIR/s2_20m_stack.jp2 "
                "-source1.rfieldx 8 "
                "-source1.rfieldy 8 "
                "-source1.placeholder x1 "
                "-source2.il $DATADIR/s2_stack.jp2 "
                "-source2.rfieldx 16 "
                "-source2.rfieldy 16 "
                "-source2.placeholder x2 "
                "-model.dir $TMPDIR/model3 "
                "-output.names prediction "
                "-out \"$TMPDIR/classif_model3_pb.tif?&box=2000:2000:500:500&gdal:co:compress=deflate\"",
                to_compare_dict={"$DATADIR/classif_model3_pb.tif": "$TMPDIR/classif_model3_pb.tif"},
                tol=INFERENCE_MAE_TOL))

    @pytest.mark.order(18)
    def test_model3_inference_fcn(self):
        self.assertTrue(
            run_command_and_compare(
                command=
                "OTB_TF_NSOURCES=2 otbcli_TensorflowModelServe "
                "-source1.il $DATADIR/s2_20m_stack.jp2 "
                "-source1.rfieldx 8 "
                "-source1.rfieldy 8 "
                "-source1.placeholder x1 "
                "-source2.il $DATADIR/s2_stack.jp2 "
                "-source2.rfieldx 16 "
                "-source2.rfieldy 16 "
                "-source2.placeholder x2 "
                "-model.dir $TMPDIR/model3 "
                "-model.fullyconv on "
                "-output.names prediction "
                "-out \"$TMPDIR/classif_model3_fcn.tif?&box=2000:2000:500:500&gdal:co:compress=deflate\"",
                to_compare_dict={"$DATADIR/classif_model3_fcn.tif": "$TMPDIR/classif_model3_fcn.tif"},
                tol=INFERENCE_MAE_TOL))

    @pytest.mark.order(19)
    def test_generate_model4(self):
        self.assertTrue(
            run_command_and_test_exist(
                command="python $TMPDIR/otbtf_tuto_repo/02_semantic_segmentation/models/create_model4.py "
                        "$TMPDIR/model4",
                file_list=["$TMPDIR/model4/saved_model.pb"]))

    @pytest.mark.order(20)
    def test_patches_selection_semseg(self):
        self.assertTrue(
            run_command_and_test_exist(
                command="otbcli_PatchesSelection "
                        "-in $DATADIR/fake_spot6.jp2 "
                        "-grid.step 64 "
                        "-grid.psize 64 "
                        "-outtrain $TMPDIR/outvec_A_semseg.gpkg "
                        "-outvalid $TMPDIR/outvec_B_semseg.gpkg",
                file_list=["$TMPDIR/outvec_A_semseg.gpkg",
                           "$TMPDIR/outvec_B_semseg.gpkg"]))

    @pytest.mark.order(21)
    def test_patch_extraction_semseg(self):
        self.assertTrue(
            run_command_and_compare(
                command="OTB_TF_NSOURCES=2 otbcli_PatchesExtraction "
                        "-source1.il $DATADIR/fake_spot6.jp2 "
                        "-source1.patchsizex 64 "
                        "-source1.patchsizey 64 "
                        "-source1.out \"$TMPDIR/amsterdam_patches_A.tif?&gdal:co:compress=deflate\" "
                        "-source2.il $TMPDIR/otbtf_tuto_repo/02_semantic_segmentation/"
                        "amsterdam_dataset/terrain_truth/amsterdam_labelimage.tif "
                        "-source2.patchsizex 64 "
                        "-source2.patchsizey 64 "
                        "-source2.out \"$TMPDIR/amsterdam_labels_A.tif?&gdal:co:compress=deflate\" "
                        "-vec $TMPDIR/outvec_A_semseg.gpkg "
                        "-field id ",
                to_compare_dict={"$DATADIR/amsterdam_labels_A.tif": "$TMPDIR/amsterdam_labels_A.tif",
                                 "$DATADIR/amsterdam_patches_A.tif": "$TMPDIR/amsterdam_patches_A.tif"}))
        self.assertTrue(
            run_command_and_compare(
                command="OTB_TF_NSOURCES=2 otbcli_PatchesExtraction "
                        "-source1.il $DATADIR/fake_spot6.jp2 "
                        "-source1.patchsizex 64 "
                        "-source1.patchsizey 64 "
                        "-source1.out \"$TMPDIR/amsterdam_patches_B.tif?&gdal:co:compress=deflate\" "
                        "-source2.il $TMPDIR/otbtf_tuto_repo/02_semantic_segmentation/"
                        "amsterdam_dataset/terrain_truth/amsterdam_labelimage.tif "
                        "-source2.patchsizex 64 "
                        "-source2.patchsizey 64 "
                        "-source2.out \"$TMPDIR/amsterdam_labels_B.tif?&gdal:co:compress=deflate\" "
                        "-vec $TMPDIR/outvec_B_semseg.gpkg "
                        "-field id ",
                to_compare_dict={"$DATADIR/amsterdam_labels_B.tif": "$TMPDIR/amsterdam_labels_B.tif",
                                 "$DATADIR/amsterdam_patches_B.tif": "$TMPDIR/amsterdam_patches_B.tif"}))

    @pytest.mark.order(22)
    def test_model4_train(self):
        self.assertTrue(
            run_command_and_test_exist(
                command="OTB_TF_NSOURCES=1 otbcli_TensorflowModelTrain "
                        "-training.source1.il $DATADIR/amsterdam_patches_A.tif "
                        "-training.source1.patchsizex 64 "
                        "-training.source1.patchsizey 64 "
                        "-training.source1.placeholder x "
                        "-training.source2.il $DATADIR/amsterdam_labels_A.tif "
                        "-training.source2.patchsizex 64 "
                        "-training.source2.patchsizey 64 "
                        "-training.source2.placeholder y "
                        "-model.dir $TMPDIR/model4 "
                        "-training.targetnodes optimizer "
                        "-training.epochs 10 "
                        "-validation.mode class "
                        "-validation.source1.il $DATADIR/amsterdam_patches_B.tif "
                        "-validation.source1.name x "
                        "-validation.source2.il $DATADIR/amsterdam_labels_B.tif "
                        "-validation.source2.name prediction "
                        "-model.saveto $TMPDIR/model4/variables/variables",
                file_list=["$TMPDIR/model4/variables/variables.index"]))

    @pytest.mark.order(23)
    def test_model4_inference(self):
        self.assertTrue(
            run_command_and_compare(
                command=
                "otbcli_TensorflowModelServe "
                "-source1.il $DATADIR/fake_spot6.jp2 "
                "-source1.rfieldx 64 "
                "-source1.rfieldy 64 "
                "-source1.placeholder x "
                "-model.dir $TMPDIR/model4 "
                "-model.fullyconv on "
                "-output.names prediction_fcn "
                "-output.efieldx 32 "
                "-output.efieldy 32 "
                "-out \"$TMPDIR/classif_model4.tif?&gdal:co:compress=deflate\" uint8",
                to_compare_dict={"$DATADIR/classif_model4.tif": "$TMPDIR/classif_model4.tif"},
                tol=INFERENCE_MAE_TOL))


if __name__ == '__main__':
    unittest.main()
