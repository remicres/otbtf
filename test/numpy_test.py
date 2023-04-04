#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest
import unittest
import otbApplication
from osgeo import gdal
from test_utils import resolve_paths

FILENAME = resolve_paths('$DATADIR/fake_spot6.jp2')

class NumpyTest(unittest.TestCase):

    def test_gdal_as_nparr(self):
        gdal_ds = gdal.Open(FILENAME)
        band = gdal_ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        self.assertTrue(arr.shape)


    def test_otb_as_nparr(self):
        app = otbApplication.Registry.CreateApplication('ExtractROI')
        app.SetParameterString("in", FILENAME)
        app.Execute()
        arr = app.GetVectorImageAsNumpyArray('out')
        self.assertTrue(arr.shape)

    def test_gdal_and_otb_np(self):
        gdal_ds = gdal.Open(FILENAME)
        band = gdal_ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        app = otbApplication.Registry.CreateApplication('ExtractROI')
        app.SetImageFromNumpyArray('in', arr)
        app.SetParameterInt('startx', 0)
        app.SetParameterInt('starty', 0)
        app.SetParameterInt('sizex', 10)
        app.SetParameterInt('sizey', 10)
        app.Execute()
        arr2 = app.GetVectorImageAsNumpyArray('out')
        self.assertTrue(arr2.shape)

if __name__ == '__main__':
    unittest.main()
