#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os

import gdal
import otbApplication as otb


class SR4RSTest(unittest.TestCase):

    def test_inference(self):
        root_dir = os.environ["CI_PROJECT_DIR"]
        out_img = "/tmp/sr4rs.tif"
        baseline = "{}/sr4rs_data/baseline/sr4rs.tif".format(root_dir)

        command = "python {}/sr4rs/code/sr.py ".format(root_dir)
        command += "--input {}/sr4rs_data/input/".format(root_dir)
        command += "SENTINEL2B_20200929-104857-489_L2A_T31TEJ_C_V2-2_FRE_10m.tif "
        command += "--savedmodel {}/sr4rs_sentinel2_bands4328_france2020_savedmodel/ ".format(root_dir)
        command += "--output '{}?&box=256:256:512:512'".format(out_img)
        os.system(command)

        nbchannels_reconstruct = gdal.Open(out_img).RasterCount
        nbchannels_baseline = gdal.Open(baseline).RasterCount

        self.assertTrue(nbchannels_reconstruct == nbchannels_baseline)

        for i in range(1, 1+nbchannels_baseline):
            comp = otb.Registry.CreateApplication('CompareImages')
            comp.SetParameterString('ref.in', baseline)
            comp.SetParameterInt('ref.channel', i)
            comp.SetParameterString('meas.in', out_img)
            comp.SetParameterInt('meas.channel', i)
            comp.Execute()
            mae = comp.GetParameterFloat('mae')

            self.assertTrue(mae < 0.01)


if __name__ == '__main__':
    unittest.main()

