#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest
import unittest
import rasterio
import rasterio.features
import rasterio.warp
from test_utils import resolve_paths

FILENAME = resolve_paths('$DATADIR/fake_spot6.jp2')

class NumpyTest(unittest.TestCase):

    def test_rio_read_md(self):
        with rasterio.open(FILENAME) as dataset:
            # Read the dataset's valid data mask as a ndarray.
            mask = dataset.dataset_mask()

            # Extract feature shapes and values from the array.
            for geom, val in rasterio.features.shapes(
                    mask, transform=dataset.transform
            ):
                # Transform shapes from the dataset's own coordinate
                # reference system to CRS84 (EPSG:4326).
                geom = rasterio.warp.transform_geom(
                    dataset.crs, 'EPSG:4326', geom, precision=6
                )
                self.assertTrue(geom)


    def test_import_all(self):
        import otbApplication
        self.assertTrue(otbApplication.Registry_GetAvailableApplications())
        import tensorflow
        self.assertTrue(tensorflow.__version__)
        from osgeo import gdal
        self.assertTrue(gdal.__version__)
        import numpy
        self.assertTrue(numpy.__version__)
        self.test_rio_read_md()


if __name__ == '__main__':
    unittest.main()
