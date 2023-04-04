#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest
import unittest

class ImportsTest(unittest.TestCase):

    def test_import_both1(self):
        import tensorflow
        self.assertTrue(tensorflow.__version__)
        import otbApplication
        self.assertTrue(otbApplication.Registry_GetAvailableApplications())


    def test_import_both2(self):
        import otbApplication
        self.assertTrue(otbApplication.Registry_GetAvailableApplications())
        import tensorflow
        self.assertTrue(tensorflow.__version__)


def test_import_all(self):
    import otbApplication
    self.assertTrue(otbApplication.Registry_GetAvailableApplications())
    import tensorflow
    self.assertTrue(tensorflow.__version__)
    from osgeo import gdal
    self.assertTrue(gdal.__version__)
    import numpy
    self.assertTrue(numpy.__version__)


if __name__ == '__main__':
    unittest.main()
