import otbApplication
import os


def get_nb_of_channels(raster):
    """
    Return the number of channels in the input raster
    :param raster: raster filename (str)
    :return the number of channels in the image (int)
    """
    info = otbApplication.Registry.CreateApplication("ReadImageInfo")
    info.SetParameterString("in", raster)
    info.ExecuteAndWriteOutput()
    return info.GetParameterInt('numberbands')


def compare(raster1, raster2, tol=0.01):
    """
    Return True if the two rasters have the same contents in each bands
    :param raster1: raster 1 filename (str)
    :param raster2: raster 2 filename (str)
    :param tol: tolerance (float)
    """
    n_bands1 = get_nb_of_channels(raster1)
    n_bands2 = get_nb_of_channels(raster2)
    if n_bands1 != n_bands2:
        print("The images have not the same number of channels")
        return False

    for i in range(1, 1 + n_bands1):
        comp = otbApplication.Registry.CreateApplication('CompareImages')
        comp.SetParameterString('ref.in', raster1)
        comp.SetParameterInt('ref.channel', i)
        comp.SetParameterString('meas.in', raster2)
        comp.SetParameterInt('meas.channel', i)
        comp.Execute()
        mae = comp.GetParameterFloat('mae')
        if mae > tol:
            print("The images have not the same content in channel {} "
                  "(Mean average error: {})".format(i, mae))
            return False
    return True


def resolve_paths(filename, var_list):
    """
    Retrieve environment variables in paths
    :param filename: file name
    :params var_list: variable list
    :return filename with retrieved environment variables
    """
    new = filename
    for var in var_list:
        new = new.replace(filename, "${}".format(var), os.environ[var])
    return new
