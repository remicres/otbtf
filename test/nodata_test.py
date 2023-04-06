#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest
import unittest
import otbtf
import otbApplication
import tensorflow as tf
from test_utils import resolve_paths

class NodataInferenceTest(unittest.TestCase):

    def test_infersimple(self):
        sm_dir = resolve_paths("$TMPDIR/l2_norm_savedmodel")

        # Create model
        x = tf.keras.Input(shape=[None, None, None], name="x")
        y = tf.norm(x, axis=-1)
        model = tf.keras.Model(inputs={"x": x}, outputs={"y": y})
        model.save(sm_dir)

        # OTB pipeline
        bmx = otbApplication.Registry.CreateApplication("BandMathX")
        bmx.SetParameterString("exp", "{idxX>idxY?idxX*idxY:0}")
        bmx.SetParameterStringList(
            "il", [resolve_paths("$DATADIR/fake_spot6.jp2")]
        )
        bmx.Execute()

        infer = otbApplication.Registry.CreateApplication(
            "TensorflowModelServe"
        )
        infer.SetParameterString("model.dir", sm_dir)
        infer.SetParameterString("model.fullyconv", "on")
        infer.AddImageToParameterInputImageList(
            "source1.il", bmx.GetParameterOutputImage("out")
        )
        infer.SetParameterFloat("source1.nodata", 0.0)
        for param in [
            "source1.rfieldx", "source1.rfieldy", "output.efieldx", "output.efieldy"
        ]:
            infer.SetParameterInt(param, 16)
        infer.SetParameterString("out", resolve_paths("$TMPDIR/nd_out.tif"))
        infer.ExecuteAndWriteOutput()


if __name__ == '__main__':
    NodataInferenceTest().test_infersimple()
    #unittest.main()
