#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os


class DEMTest(unittest.TestCase):

    def test_dem(self):
        os.system("python /builds/remi.cresson/sr4rs/code/sr.py "
                  "--input /builds/remi.cresson/sr4rs_data/input/"
                  "SENTINEL2B_20200929-104857-489_L2A_T31TEJ_C_V2-2_FRE_10m.tif "
                  "--savedmodel /builds/remi.cresson/sr4rs_sentinel2_bands4328_france2020_savedmodel/ "
                  "--output '/tmp/sr4rs.tif?&box=256:256:512:512'")

        with os.popen('gdalinfo /builds/remi.cresson/sr4rs_data/baseline/sr4rs.tf | grep '
                      '--invert-match -e "Files:" -e "METADATATYPE" -e "OTB_VERSION" -e "NoData Value"') as file:
            baseline_sr4rs_gdalinfo = file.read()

        with os.popen('gdalinfo /tmp/sr4rs.tif | grep --invert-match -e '
                      '"Files:" -e "METADATATYPE" -e '
                      '"OTB_VERSION" -e "NoData Value"') as file:
            reconstructed_sr4rs_gdalinfo = file.read()

        self.assertEqual(baseline_sr4rs_gdalinfo, reconstructed_sr4rs_gdalinfo)


if __name__ == '__main__':
    unittest.main()
