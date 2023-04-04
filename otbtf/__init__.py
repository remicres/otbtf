# -*- coding: utf-8 -*-
# ==========================================================================
#
#   Copyright 2018-2019 IRSTEA
#   Copyright 2020-2022 INRAE
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0.txt
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ==========================================================================*/
"""
OTBTF python module
"""
import pkg_resources
try:
    from otbtf.utils import read_as_np_arr, gdal_open  # noqa
    from otbtf.dataset import Buffer, PatchesReaderBase, PatchesImagesReader, \
        IteratorBase, RandomIterator, Dataset, DatasetFromPatchesImages  # noqa
except ImportError:
    print(
        "Warning: otbtf.utils and otbtf.dataset were not imported. "
        "Using OTBTF without GDAL."
    )

from otbtf.tfrecords import TFRecords  # noqa
from otbtf.model import ModelBase  # noqa
__version__ = pkg_resources.require("otbtf")[0].version
