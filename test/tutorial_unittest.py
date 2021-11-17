#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest
import unittest
import os
from pathlib import Path
import test_utils

INFERENCE_MAE_TOL = 10.0  # Dummy value: we don't really care of the mae value but rather the image size etc


def resolve_paths(path):
    """
    Resolve a path with the environment variables
    """
    return test_utils.resolve_paths(path, var_list=["TMPDIR", "DATADIR"])


def run_command(command):
    """
    Run a command
    :param command: the command to run
    """
    full_command = resolve_paths(command)
    print("Running command: \n\t {}".format(full_command))
    os.system(full_command)


def run_command_and_test_exist(command, file_list):
    """
    :param command: the command to run (str)
    :param file_list: list of files to check
    :return True or False
    """
    run_command(command)
    print("Checking if files exist...")
    for file in file_list:
        print("\t{}".format(file))
        path = Path(resolve_paths(file))
        if not path.is_file():
            print("File {} does not exist!".format(file))
            return False
        print("\tOk")
    return True


def run_command_and_compare(command, to_compare_dict, tol=0.01):
    """
    :param command: the command to run (str)
    :param to_compare_dict: a dict of {baseline1: output1, ..., baselineN: outputN}
    :param tol: tolerance (float)
    :return True or False
    """

    run_command(command)
    for baseline, output in to_compare_dict.items():
        if not test_utils.compare(resolve_paths(baseline), resolve_paths(output), tol):
            print("Baseline {} and output {} differ.".format(baseline, output))
            return False
    return True


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
            run_command_and_compare(command="otbcli_TensorflowModelServe "
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
            run_command_and_compare(command="otbcli_TensorflowModelServe "
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
            run_command_and_test_exist(command="otbcli_SampleExtraction "
                                               "-in $DATADIR/s2_stack.jp2 "
                                               "-vec $TMPDIR/outvec_A.gpkg "
                                               "-field class "
                                               "-out $TMPDIR/pixelvalues_A.gpkg",
                                       file_list=["$TMPDIR/pixelvalues_A.gpkg"]))
        self.assertTrue(
            run_command_and_test_exist(command="otbcli_SampleExtraction "
                                               "-in $DATADIR/s2_stack.jp2 "
                                               "-vec $TMPDIR/outvec_B.gpkg "
                                               "-field class "
                                               "-out $TMPDIR/pixelvalues_B.gpkg",
                                       file_list=["$TMPDIR/pixelvalues_B.gpkg"]))

    @pytest.mark.order(8)
    def test_rf_training(self):
        self.assertTrue(
            run_command_and_test_exist(command="otbcli_TrainVectorClassifier "
                                               "-io.vd $TMPDIR/pixelvalues_A.gpkg "
                                               "-valid.vd $TMPDIR/pixelvalues_B.gpkg "
                                               "-feat value_0 value_1 value_2 value_3 "
                                               "-cfield class "
                                               "-classifier rf "
                                               "-io.out $TMPDIR/randomforest_model.yaml ",
                                       file_list=["$TMPDIR/randomforest_model.yaml"]))


if __name__ == '__main__':
    unittest.main()
