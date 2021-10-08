#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os

import gdal
import otbApplication as otb


class SR4RSTest(unittest.TestCase):

    def test_dem(self):
        os.system("python /builds/remi.cresson/sr4rs/code/sr.py "
                  "--input /builds/remi.cresson/sr4rs_data/input/"
                  "SENTINEL2B_20200929-104857-489_L2A_T31TEJ_C_V2-2_FRE_10m.tif "
                  "--savedmodel /builds/remi.cresson/sr4rs_sentinel2_bands4328_france2020_savedmodel/ "
                  "--output '/tmp/sr4rs.tif?&box=256:256:512:512'")

        with gdal.Open("/tmp/sr4rs.tif") as reconstruct:
            nbchannels_reconstruct = reconstruct.RasterCount

        with gdal.Open("/builds/remi.cresson/sr4rs_data/baseline/sr4rs.tif") as baseline:
            nbchannels_baseline = baseline.RasterCount

        self.assertTrue(nbchannels_reconstruct == nbchannels_baseline)

        for i in range(1, 1+nbchannels_baseline):
            comp = otb.Registry.CreateApplication('CompareImages')
            comp.SetParameterString('ref.in', "/builds/remi.cresson/sr4rs_data/baseline/sr4rs.tif")
            comp.SetParameterInt('ref.channel', i)
            comp.SetParameterString('meas.in', "/tmp/sr4rs.tif")
            comp.SetParameterInt('meas.channel', i)
            comp.Execute()
            mae = comp.GetParameterFloat('mae')

            self.assertTrue(mae < 0.01)


if __name__ == '__main__':
    unittest.main()
