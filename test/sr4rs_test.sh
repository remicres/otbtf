#!/bin/bash
set -e
export DATASET_S2=https://nextcloud.inrae.fr/s/EZL2JN7SZyDK8Cf/download/sr4rs_sentinel2_bands4328_france2020_savedmodel.zip
export DATASET_SR4RS=https://nextcloud.inrae.fr/s/kDms9JrRMQE2Q5z/download
wget -qO sr4rs_sentinel2_bands4328_france2020_savedmodel.zip $DATASET_S2
unzip -o sr4rs_sentinel2_bands4328_france2020_savedmodel.zip
wget -qO sr4rs_data.zip $DATASET_SR4RS
unzip -o sr4rs_data.zip
rm -rf sr4rs
git clone https://github.com/remicres/sr4rs.git
export PYTHONPATH=$PYTHONPATH:$PWD/sr4rs
pytest -v --junitxml=report_sr4rs.xml test/sr4rs_unittest.py