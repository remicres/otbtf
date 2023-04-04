#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest
import unittest
import rasterio
import rasterio.features
import rasterio.warp

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

if __name__ == '__main__':
    unittest.main()
