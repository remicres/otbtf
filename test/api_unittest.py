#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest

import pytest

from otbtf.examples.tensorflow_v2x.fcnn import create_tfrecords
from otbtf.examples.tensorflow_v2x.fcnn import train_from_patchesimages
from otbtf.examples.tensorflow_v2x.fcnn import train_from_tfrecords
from otbtf.examples.tensorflow_v2x.fcnn.fcnn_model import INPUT_NAME, \
    OUTPUT_SOFTMAX_NAME
from otbtf.model import cropped_tensor_name
from test_utils import resolve_paths, files_exist, run_command_and_compare

INFERENCE_MAE_TOL = 10.0  # Dummy value: we don't really care of the mae value but rather the image size etc


class APITest(unittest.TestCase):

    @pytest.mark.order(1)
    def test_train_from_patchesimages(self):
        params = train_from_patchesimages.parser.parse_args([
            '--model_dir', resolve_paths('$TMPDIR/model_from_pimg'),
            '--nb_epochs', '1',
            '--train_xs',
            resolve_paths('$DATADIR/amsterdam_patches_A.tif'),
            '--train_labels',
            resolve_paths('$DATADIR/amsterdam_labels_A.tif'),
            '--valid_xs',
            resolve_paths('$DATADIR/amsterdam_patches_B.tif'),
            '--valid_labels',
            resolve_paths('$DATADIR/amsterdam_labels_B.tif')
        ])
        train_from_patchesimages.train(params=params)
        self.assertTrue(files_exist([
            '$TMPDIR/model_from_pimg/keras_metadata.pb',
            '$TMPDIR/model_from_pimg/saved_model.pb',
            '$TMPDIR/model_from_pimg/variables/variables.data-00000-of-00001',
            '$TMPDIR/model_from_pimg/variables/variables.index'
        ]))

    @pytest.mark.order(2)
    def test_model_inference1(self):
        self.assertTrue(
            run_command_and_compare(
                command=
                "otbcli_TensorflowModelServe "
                "-source1.il $DATADIR/fake_spot6.jp2 "
                "-source1.rfieldx 64 "
                "-source1.rfieldy 64 "
                f"-source1.placeholder {INPUT_NAME} "
                "-model.dir $TMPDIR/model_from_pimg "
                "-model.fullyconv on "
                f"-output.names {cropped_tensor_name(OUTPUT_SOFTMAX_NAME, 16)} "
                "-output.efieldx 32 "
                "-output.efieldy 32 "
                "-out \"$TMPDIR/classif_model4_softmax.tif?&gdal:co:compress=deflate\" uint8",
                to_compare_dict={
                    "$DATADIR/classif_model4_softmax.tif": "$TMPDIR/classif_model4_softmax.tif"},
                tol=INFERENCE_MAE_TOL))
        self.assertTrue(
            run_command_and_compare(
                command=
                "otbcli_TensorflowModelServe "
                "-source1.il $DATADIR/fake_spot6.jp2 "
                "-source1.rfieldx 128 "
                "-source1.rfieldy 128 "
                f"-source1.placeholder {INPUT_NAME} "
                "-model.dir $TMPDIR/model_from_pimg "
                "-model.fullyconv on "
                f"-output.names {cropped_tensor_name(OUTPUT_SOFTMAX_NAME, 32)} "
                "-output.efieldx 64 "
                "-output.efieldy 64 "
                "-out \"$TMPDIR/classif_model4_softmax.tif?&gdal:co:compress=deflate\" uint8",
                to_compare_dict={
                    "$DATADIR/classif_model4_softmax.tif": "$TMPDIR/classif_model4_softmax.tif"},
                tol=INFERENCE_MAE_TOL))

    @pytest.mark.order(3)
    def test_create_tfrecords(self):
        params = create_tfrecords.parser.parse_args([
            '--xs', resolve_paths('$DATADIR/amsterdam_patches_A.tif'),
            '--labels', resolve_paths('$DATADIR/amsterdam_labels_A.tif'),
            '--outdir', resolve_paths('$TMPDIR/train')
        ])
        create_tfrecords.create_tfrecords(params=params)
        self.assertTrue(files_exist([
            '$TMPDIR/train/output_shapes.json',
            '$TMPDIR/train/output_types.json',
            '$TMPDIR/train/0.records'
        ]))
        params = create_tfrecords.parser.parse_args([
            '--xs', resolve_paths('$DATADIR/amsterdam_patches_B.tif'),
            '--labels', resolve_paths('$DATADIR/amsterdam_labels_B.tif'),
            '--outdir', resolve_paths('$TMPDIR/valid')
        ])
        create_tfrecords.create_tfrecords(params=params)
        self.assertTrue(files_exist([
            '$TMPDIR/valid/output_shapes.json',
            '$TMPDIR/valid/output_types.json',
            '$TMPDIR/valid/0.records'
        ]))

    @pytest.mark.order(4)
    def test_train_from_tfrecords(self):
        params = train_from_tfrecords.parser.parse_args([
            '--model_dir', resolve_paths('$TMPDIR/model_from_tfrecs'),
            '--nb_epochs', '1',
            '--tfrecords_dir', resolve_paths('$TMPDIR')
        ])
        train_from_tfrecords.train(params=params)
        self.assertTrue(files_exist([
            '$TMPDIR/model_from_tfrecs/keras_metadata.pb',
            '$TMPDIR/model_from_tfrecs/saved_model.pb',
            '$TMPDIR/model_from_tfrecs/variables/variables.data-00000-of-00001',
            '$TMPDIR/model_from_tfrecs/variables/variables.index'
        ]))

    @pytest.mark.order(5)
    def test_model_inference2(self):
        self.assertTrue(
            run_command_and_compare(
                command=
                "otbcli_TensorflowModelServe "
                "-source1.il $DATADIR/fake_spot6.jp2 "
                "-source1.rfieldx 64 "
                "-source1.rfieldy 64 "
                f"-source1.placeholder {INPUT_NAME} "
                "-model.dir $TMPDIR/model_from_pimg "
                "-model.fullyconv on "
                f"-output.names {cropped_tensor_name(OUTPUT_SOFTMAX_NAME, 16)} "
                "-output.efieldx 32 "
                "-output.efieldy 32 "
                "-out \"$TMPDIR/classif_model4_softmax.tif?&gdal:co:compress=deflate\" uint8",
                to_compare_dict={
                    "$DATADIR/classif_model4_softmax.tif":
                        "$TMPDIR/classif_model4_softmax.tif"
                },
                tol=INFERENCE_MAE_TOL))

        self.assertTrue(
            run_command_and_compare(
                command=
                "otbcli_TensorflowModelServe "
                "-source1.il $DATADIR/fake_spot6.jp2 "
                "-source1.rfieldx 128 "
                "-source1.rfieldy 128 "
                f"-source1.placeholder {INPUT_NAME} "
                "-model.dir $TMPDIR/model_from_pimg "
                "-model.fullyconv on "
                f"-output.names {cropped_tensor_name(OUTPUT_SOFTMAX_NAME, 32)} "
                "-output.efieldx 64 "
                "-output.efieldy 64 "
                "-out \"$TMPDIR/classif_model4_softmax.tif?&gdal:co:compress=deflate\" uint8",
                to_compare_dict={
                    "$DATADIR/classif_model4_softmax.tif":
                        "$TMPDIR/classif_model4_softmax.tif"
                },
                tol=INFERENCE_MAE_TOL))


if __name__ == '__main__':
    unittest.main()
