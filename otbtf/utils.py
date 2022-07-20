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
The utils module provides some helpers to read patches using gdal
"""
from osgeo import gdal
import numpy as np


# ----------------------------------------------------- Helpers --------------------------------------------------------

def gdal_open(filename):
    """
    Open a GDAL raster
    :param filename: raster file
    :return: a GDAL dataset instance
    """
    gdal_ds = gdal.Open(filename)
    if not gdal_ds:
        raise Exception(f"Unable to open file {filename}")
    return gdal_ds


def read_as_np_arr(gdal_ds, as_patches=True, dtype=None):
    """
    Read a GDAL raster as numpy array
    :param gdal_ds: a GDAL dataset instance
    :param as_patches: if True, the returned numpy array has the following shape (n, psz_x, psz_x, nb_channels). If
        False, the shape is (1, psz_y, psz_x, nb_channels)
    :param dtype: if not None array dtype will be cast to given numpy data type (np.float32, np.uint16...)
    :return: Numpy array of dim 4
    """
    buffer = gdal_ds.ReadAsArray()
    size_x = gdal_ds.RasterXSize
    if len(buffer.shape) == 3:
        buffer = np.transpose(buffer, axes=(1, 2, 0))
    if as_patches:
        n_elems = int(gdal_ds.RasterYSize / size_x)
        size_y = size_x
    else:
        n_elems = 1
        size_y = gdal_ds.RasterYSize

    buffer = buffer.reshape((n_elems, size_y, size_x, gdal_ds.RasterCount))
    if dtype is not None:
        buffer = buffer.astype(dtype)

    return buffer


def _is_chief(strategy):
    """
    Tell if the current worker is the chief.

    :param strategy: strategy
    :return: True if the current worker is the chief, False else
    """
    # Note: there are two possible `TF_CONFIG` configuration.
    #   1) In addition to `worker` tasks, a `chief` task type is use;
    #      in this case, this function should be modified to
    #      `return task_type == 'chief'`.
    #   2) Only `worker` task type is used; in this case, worker 0 is
    #      regarded as the chief. The implementation demonstrated here
    #      is for this case.
    # For the purpose of this Colab section, the `task_type is None` case
    # is added because it is effectively run with only a single worker.

    if strategy.cluster_resolver:  # this means MultiWorkerMirroredStrategy
        task_type, task_id = strategy.cluster_resolver.task_type, strategy.cluster_resolver.task_id
        return (task_type == 'chief') or (task_type == 'worker' and task_id == 0) or task_type is None
    # strategy with only one worker
    return True
