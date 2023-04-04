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


if __name__ == '__main__':
    unittest.main()
