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
        raise Exception("Unable to open file {}".format(filename))
    return gdal_ds


def read_as_np_arr(gdal_ds, as_patches=True):
    """
    Read a GDAL raster as numpy array
    :param gdal_ds: a GDAL dataset instance
    :param as_patches: if True, the returned numpy array has the following shape (n, psz_x, psz_x, nb_channels). If
        False, the shape is (1, psz_y, psz_x, nb_channels)
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
    return np.float32(buffer.reshape((n_elems, size_y, size_x, gdal_ds.RasterCount)))
