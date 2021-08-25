# -*- coding: utf-8 -*-
# ==========================================================================
#
#   Copyright 2018-2019 Remi Cresson (IRSTEA)
#   Copyright 2020 Remi Cresson (INRAE)
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
import gdal
import numpy as np
import tensorflow.compat.v1 as tf
from deprecated import deprecated

tf.disable_v2_behavior()


def read_image_as_np(filename, as_patches=False):
    """
    Read an image as numpy array.
    @param filename File name of patches image
    @param as_patches True if the image must be read as patches
    @return 4D numpy array [batch, h, w, c]
    """

    # Open a GDAL dataset
    ds = gdal.Open(filename)
    if ds is None:
        raise Exception("Unable to open file {}".format(filename))

    # Raster infos
    n_bands = ds.RasterCount
    szx = ds.RasterXSize
    szy = ds.RasterYSize

    # Raster array
    myarray = ds.ReadAsArray()

    # Re-order bands (when there is > 1 band)
    if (len(myarray.shape) == 3):
        axes = (1, 2, 0)
        myarray = np.transpose(myarray, axes=axes)

    if (as_patches):
        n = int(szy / szx)
        return myarray.reshape((n, szx, szx, n_bands))

    return myarray.reshape((1, szy, szx, n_bands))


def create_savedmodel(sess, inputs, outputs, directory):
    """
    Create a SavedModel
    @param sess      TF session
    @param inputs    List of inputs names (e.g. ["x_cnn_1:0", "x_cnn_2:0"])
    @param outputs   List of outputs names (e.g. ["prediction:0", "features:0"])
    @param directory Path for the generated SavedModel
    """
    print("Create a SavedModel in " + directory)
    graph = tf.compat.v1.get_default_graph()
    inputs_names = {i: graph.get_tensor_by_name(i) for i in inputs}
    outputs_names = {o: graph.get_tensor_by_name(o) for o in outputs}
    tf.compat.v1.saved_model.simple_save(sess, directory, inputs=inputs_names, outputs=outputs_names)

def ckpt_to_savedmodel(ckpt_path, inputs, outputs, savedmodel_path, clear_devices=False):
    """
    Read a Checkpoint and build a SavedModel
    @param ckpt_path       Path to the checkpoint file (without the ".meta" extension)
    @param inputs          List of inputs names (e.g. ["x_cnn_1:0", "x_cnn_2:0"])
    @param outputs         List of outputs names (e.g. ["prediction:0", "features:0"])
    @param savedmodel_path Path for the generated SavedModel
    @param clear_devices   Clear TF devices positionning (True/False)
    """
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        # Restore variables from disk
        model_saver = tf.compat.v1.train.import_meta_graph(ckpt_path + ".meta", clear_devices=clear_devices)
        model_saver.restore(sess, ckpt_path)

        # Create a SavedModel
        create_savedmodel(sess, inputs=inputs, outputs=outputs, directory=savedmodel_path)

@deprecated
def read_samples(filename):
   """
   Read a patches image.
   @param filename: raster file name
   """
   return read_image_as_np(filename, as_patches=True)

@deprecated
def CreateSavedModel(sess, inputs, outputs, directory):
    """
    Create a SavedModel
    @param sess      TF session
    @param inputs    List of inputs names (e.g. ["x_cnn_1:0", "x_cnn_2:0"])
    @param outputs   List of outputs names (e.g. ["prediction:0", "features:0"])
    @param directory Path for the generated SavedModel
    """
    create_savedmodel(sess, inputs, outputs, directory)

@deprecated
def CheckpointToSavedModel(ckpt_path, inputs, outputs, savedmodel_path, clear_devices=False):
    """
    Read a Checkpoint and build a SavedModel
    @param ckpt_path       Path to the checkpoint file (without the ".meta" extension)
    @param inputs          List of inputs names (e.g. ["x_cnn_1:0", "x_cnn_2:0"])
    @param outputs         List of outputs names (e.g. ["prediction:0", "features:0"])
    @param savedmodel_path Path for the generated SavedModel
    @param clear_devices   Clear TF devices positionning (True/False)
    """
    ckpt_to_savedmodel(ckpt_path, inputs, outputs, savedmodel_path, clear_devices)
