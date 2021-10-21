#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import os
from pathlib import Path
import gdal
import otbApplication as otb


class SR4RSv1Test(unittest.TestCase):

    def test_train_nostream(self):
        root_dir = os.environ["CI_PROJECT_DIR"]
        ckpt_file = "/tmp/sr4rs_train_ckpt"

        def _input(file_name):
            return "{}/sr4rs_data/input/{}".format(root_dir, file_name)

        command = "python {}/sr4rs/code/train.py ".format(root_dir)
        command += "--lr_patches "
        command += _input("DIM_SPOT6_MS_202007290959110_ORT_ORTHO-MS-193_posA_s2.jp2 ")
        command += _input("DIM_SPOT7_MS_202004111036186_ORT_ORTHO-MS-081_posA_s2.jp2 ")
        command += _input("DIM_SPOT7_MS_202006201000507_ORT_ORTHO-MS-054_posA_s2.jp2 ")
        command += "--hr_patches "
        command += _input("DIM_SPOT6_MS_202007290959110_ORT_ORTHO-MS-193_posA_s6_cal.jp2 ")
        command += _input("DIM_SPOT7_MS_202004111036186_ORT_ORTHO-MS-081_posA_s6_cal.jp2 ")
        command += _input("DIM_SPOT7_MS_202006201000507_ORT_ORTHO-MS-054_posA_s6_cal.jp2 ")
        command += "--save_ckpt {} ".format(ckpt_file)
        command += "--depth 4 "
        command += "--nresblocks 1 "
        command += "--epochs 1 "
        os.system(command)

        file = Path("{}/checkpoint".format(ckpt_file))
        self.assertTrue(file.is_file())

    def test_train_stream(self):
        root_dir = os.environ["CI_PROJECT_DIR"]
        ckpt_file = "/tmp/sr4rs_train_ckpt"

        def _input(file_name):
            return "{}/sr4rs_data/input/{}".format(root_dir, file_name)

        command = "python {}/sr4rs/code/train.py ".format(root_dir)
        command += "--lr_patches "
        command += _input("DIM_SPOT6_MS_202007290959110_ORT_ORTHO-MS-193_posA_s2.jp2 ")
        command += _input("DIM_SPOT7_MS_202004111036186_ORT_ORTHO-MS-081_posA_s2.jp2 ")
        command += _input("DIM_SPOT7_MS_202006201000507_ORT_ORTHO-MS-054_posA_s2.jp2 ")
        command += "--hr_patches "
        command += _input("DIM_SPOT6_MS_202007290959110_ORT_ORTHO-MS-193_posA_s6_cal.jp2 ")
        command += _input("DIM_SPOT7_MS_202004111036186_ORT_ORTHO-MS-081_posA_s6_cal.jp2 ")
        command += _input("DIM_SPOT7_MS_202006201000507_ORT_ORTHO-MS-054_posA_s6_cal.jp2 ")
        command += "--save_ckpt {} ".format(ckpt_file)
        command += "--depth 4 "
        command += "--nresblocks 1 "
        command += "--epochs 1 "
        command += "--streaming"
        os.system(command)

        file = Path("{}/checkpoint".format(ckpt_file))
        self.assertTrue(file.is_file())

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

        for i in range(1, 1 + nbchannels_baseline):
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
