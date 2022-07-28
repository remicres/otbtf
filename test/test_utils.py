import otbApplication
import os
from pathlib import Path


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


def resolve_paths(path):
    """
    Resolve a path with the environment variables
    """
    return os.path.expandvars(path)


def files_exist(file_list):
    """
    Check is all files exist
    """
    print("Checking if files exist...")
    for file in file_list:
        print("\t{}".format(file))
        path = Path(resolve_paths(file))
        if not path.is_file():
            print("File {} does not exist!".format(file))
            return False
        print("\tOk")
    return True


def run_command(command):
    """
    Run a command
    :param command: the command to run
    """
    full_command = resolve_paths(command)
    print("Running command: \n\t {}".format(full_command))
    os.system(full_command)


def run_command_and_test_exist(command, file_list):
    """
    :param command: the command to run (str)
    :param file_list: list of files to check
    :return True or False
    """
    run_command(command)
    return files_exist(file_list)


def run_command_and_compare(command, to_compare_dict, tol=0.01):
    """
    :param command: the command to run (str)
    :param to_compare_dict: a dict of {baseline1: output1, ..., baselineN: outputN}
    :param tol: tolerance (float)
    :return True or False
    """

    run_command(command)
    for baseline, output in to_compare_dict.items():
        if not compare(resolve_paths(baseline), resolve_paths(output), tol):
            print("Baseline {} and output {} differ.".format(baseline, output))
            return False
    return True
