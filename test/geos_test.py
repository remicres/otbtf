#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest
import unittest
from osgeo import ogr

class APITest(unittest.TestCase):

    def test_geosu_support(self):
        wkt1 = "POLYGON ((1208064.271243039 624154.6783778917, 1208064.271243039 601260.9785661874, 1231345.9998651114 601260.9785661874, 1231345.9998651114 624154.6783778917, 1208064.271243039 624154.6783778917))"
        wkt2 = "POLYGON ((1199915.6662253144 633079.3410163528, 1199915.6662253144 614453.958118695, 1219317.1067437078 614453.958118695, 1219317.1067437078 633079.3410163528, 1199915.6662253144 633079.3410163528)))"
        poly1 = ogr.CreateGeometryFromWkt(wkt1)
        poly2 = ogr.CreateGeometryFromWkt(wkt2)
        intersection = poly1.Intersection(poly2)

if __name__ == '__main__':
    unittest.main()
