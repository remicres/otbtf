#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os
from pathlib import Path
import test_utils


def run_command(command):
    tmpdir = os.environ["TMPDIR"]
    datadir = os.environ["DATADIR"]
    com = "TMPDIR={} DATADIR={} {}".format(tmpdir, datadir, command)
    print("Running command {}".format(com))
    os.system(com)


def run_command_and_test_exist(command, file_list):
    """
    :param command: the command to run (str)
    :param file_list: list of files to check
    """
    run_command(command)
    for file in file_list:
        path = Path(file)
        if not path.is_file():
            print("File {} does not exist!".format(file))
            return False
    return True


def run_command_and_compare(command, to_compare_dict, tol=0.01):
    """
    :param command: the command to run (str)
    :param to_compare_dict: a dict of {baseline1: output1, ..., baselineN: outputN}
    """
    run_command(command)
    for baseline, output in to_compare_dict.items():
        if not test_utils.compare(baseline, output, tol):
            print("Baseline {} and output {} differ.".format(baseline, output))
            return False
    return True


class TutorialTest(unittest.TestCase):

    def test_sample_selection(self):
        self.assertTrue(
            run_command_and_test_exist(
                "otbcli_LabelImageSampleSelection "
                "-inref $DATADIR/terrain_truth_epsg32654_A.tif "
                "-nodata 255"
                "-outvec $TMPDIR/outvec_A.gpkg", "outvec_A.gpkg"))
        self.assertTrue(
            run_command_and_test_exist(
                "otbcli_LabelImageSampleSelection "
                "-inref $DATADIR/terrain_truth_epsg32654_B.tif "
                "-nodata 255"
                "-outvec $TMPDIR/outvec_B.gpkg", "outvec_B.gpkg"))

    def test_patches_extraction(self):
        self.assertTrue(
            run_command_and_compare(
                "otbcli_PatchesExtraction "
                "-source1.il $DATADIR/s2_stack.jp2 "
                "-source1.out $TMPDIR/s2_patches_A.tif "
                "-source1.patchsizex 16 "
                "-source1.patchsizey 16 "
                "-vec $TMPDIR/outvec_A.gpkg "
                "-field class "
                "-outlabels $TMPDIR/s2_labels_A.tif",
                {"$DATADIR/s2_patches_A.tif": "$TMPDIR/s2_patches_A.tif",
                 "$DATADIR/s2_labels_A.tif": "$TMPDIR/s2_labels_A.tif"}))
        self.assertTrue(
            run_command_and_compare(
                "otbcli_PatchesExtraction "
                "-source1.il $DATADIR/s2_stack.jp2 "
                "-source1.out $TMPDIR/s2_patches_B.tif "
                "-source1.patchsizex 16 "
                "-source1.patchsizey 16 "
                "-vec $TMPDIR/outvec_B.gpkg "
                "-field class "
                "-outlabels $TMPDIR/s2_labels_B.tif",
                {"$DATADIR/s2_patches_B.tif": "$TMPDIR/s2_patches_B.tif",
                 "$DATADIR/s2_labels_B.tif": "$TMPDIR/s2_labels_B.tif"}))

    def test_generate_model1(self):
        run_command("git clone https://github.com/remicres/otbtf_tutorials_resources.git $TMPDIR/otbtf_tuto_repo")
        self.assertTrue(
            run_command_and_test_exist(
                "python $TMPDIR/otbtf_tuto_repo/01_patch_based_classification/models/create_model1.py "
                "$TMPDIR/model1",
                "$TMPDIR/model1/saved_model.pb"))

    def test_model1_train(self):
        self.assertTrue(
            run_command_and_test_exist(
                "otbcli_TensorflowModelTrain "
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
                "-model.saveto $TMPDIR/model1/variables/variables"
            )
        )

    def test_model1_inference_pb(self):
        self.assertTrue(
            run_command_and_compare("otbcli_TensorflowModelServe "
                                    "-source1.il $DATADIR/s2_stack.jp2 "
                                    "-source1.rfieldx 16 "
                                    "-source1.rfieldy 16 "
                                    "-source1.placeholder x "
                                    "-model.dir $TMPDIR/model1 "
                                    "-output.names prediction "
                                    "-out \"$TMPDIR/classif_model1.tif?&box=4000:4000:1000:1000\" uint8",
                                    {"$DATADIR/classif_model1.tif": "$TMPDIR/classif_model1.tif"}))

    def test_model1_inference_fcn(self):
        self.assertTrue(
            run_command_and_compare("otbcli_TensorflowModelServe "
                                    "-source1.il $DATADIR/s2_stack.jp2 "
                                    "-source1.rfieldx 16 "
                                    "-source1.rfieldy 16 "
                                    "-source1.placeholder x "
                                    "-model.dir $TMPDIR/model1 "
                                    "-output.names prediction "
                                    "-model.fullyconv on "
                                    "-output.spcscale 4 "
                                    "-out \"$TMPDIR/classif_model1.tif?&box=1000:1000:256:256\" uint8",
                                    {"$DATADIR/classif_model1.tif": "$TMPDIR/classif_model1.tif"},
                                    tol=0.1))


if __name__ == '__main__':
    unittest.main()
