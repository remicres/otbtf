#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytest
import unittest
import planetary_computer
import pystac_client
import otbApplication

class PCTest(unittest.TestCase):

    def test_pc(self):
        api = pystac_client.Client.open(
            'https://planetarycomputer.microsoft.com/api/stac/v1',
            modifier=planetary_computer.sign_inplace,
        )

        res = api.search(
            bbox=[4, 42.99, 4.5, 43.05], 
            datetime=["2022-01-01", "2022-01-09"],
            collections=["sentinel-2-l2a"]
        )

        r = next(res.items())
        url = r.assets["B04"].href
        info = otbApplication.Registry.CreateApplication("ReadImageInfo")
        info.SetParameterString("in", "/vsicurl/" + url)
        info.Execute()
        assert len(info.GetParameterString("projectionref")) > 1


if __name__ == '__main__':
    unittest.main()
