#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import otbApplication
import pytest
import tensorflow as tf
import unittest

import otbtf
from test_utils import resolve_paths, compare


class NodataInferenceTest(unittest.TestCase):

    def test_infersimple(self):
        """
        In this test, we create a synthetic image:
            f(x, y) = x * y if x > y else 0

        Then we use an input no-data value (`source1.nodata 0`) and a
        background value for the output (`output.bv 1024`).

        We use the l2_norm SavedModel, forcing otbtf to use a tiling scheme
        of 4x4. If the test succeeds, the output pixels in 4x4 areas where
        there is at least one no-data pixel (i.e. 0), should be filled with
        the `bv` value (i.e. 1024).

        """
        sm_dir = resolve_paths("$TMPDIR/l2_norm_savedmodel")

        # Create model
        x = tf.keras.Input(shape=[None, None, None], name="x")
        y = tf.norm(x, axis=-1)
        model = tf.keras.Model(inputs={"x": x}, outputs={"y": y})
        model.save(sm_dir)

        # Input image: f(x, y) = x * y if x > y else 0
        bmx = otbApplication.Registry.CreateApplication("BandMathX")
        bmx.SetParameterString("exp", "{idxX>idxY?idxX*idxY:0}")
        bmx.SetParameterStringList(
            "il", [resolve_paths("$DATADIR/xs_subset.tif")]
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
            "source1.rfieldx",
            "source1.rfieldy",
            "output.efieldx",
            "output.efieldy",
            "optim.tilesizex",
            "optim.tilesizey",
        ]:
            infer.SetParameterInt(param, 4)

        infer.SetParameterFloat("output.bv", 1024)
        infer.SetParameterString("out", resolve_paths("$TMPDIR/nd_out.tif"))
        infer.ExecuteAndWriteOutput()

        self.assertTrue(
            compare(
                raster1=resolve_paths("$TMPDIR/nd_out.tif"),
                raster2=resolve_paths("$DATADIR/nd_out.tif"),
            )
        )


if __name__ == '__main__':
    unittest.main()
