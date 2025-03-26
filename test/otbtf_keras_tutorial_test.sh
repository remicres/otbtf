#!/bin/bash
set -e
cd /tmp

# Clone repository
git clone https://forgemia.inra.fr/orfeo-toolbox/otbtf-keras-tutorial.git
cd otbtf-keras-tutorial/python

# Use smaller input ROI
sed -i 's#, ext_fname=\"gdal:co:COMPRESS=DEFLATE\"##g' part_1_download.py
sed -i 's#/data/s2_tokyo_10m.tif#/data/s2_tokyo_10m.tif\?\&box=2000:2000:3000:3000\&gdal:co:COMPRESS=DEFLATE#g' part_1_download.py
sed -i 's#/data/s2_tokyo_20m.tif#/data/s2_tokyo_20m.tif\?\&box=1000:1000:1500:1500\&gdal:co:COMPRESS=DEFLATE#g' part_1_download.py

# Use smaller output ROIs
find . -type f -exec sed -i 's#4000:4000:1000:1000#0:0:1000:1000#g' {} \;
find . -type f -exec sed -i 's#2000:2000:500:500#0:0:500:500#g' {} \;

# Remane `/data` --> `/tmp/data` everywhere
mkdir -p /tmp/data
find . -type f -exec sed -i 's#/data/#/tmp/data/#g' {} \;
sh run_all.sh