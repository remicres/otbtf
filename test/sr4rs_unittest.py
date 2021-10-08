#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os


class DEMTest(unittest.TestCase):

    def test_dem(self):
        os.system("python /builds/remi.cresson/sr4rs/code/sr.py "
                  "--input /builds/remi.cresson/decloud_data/baseline/PREPARE/S2_PREPARE/T31TEJ/"
                  "SENTINEL2A_20201024-104859-766_L2A_T31TEJ_C_V2-2/"
                  "SENTINEL2A_20201024-104859-766_L2A_T31TEJ_C_V2-2_FRE_10m.tif "
                  "--savedmodel /builds/remi.cresson/sr4rs_sentinel2_bands4328_france2020_savedmodel/ "
                  "--output '/tmp/test.tif&box=0:0:256:256'")


if __name__ == '__main__':
    unittest.main()
