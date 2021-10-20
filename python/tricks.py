# -*- coding: utf-8 -*-
# ==========================================================================
#
#   Copyright 2018-2019 IRSTEA
#   Copyright 2020-2021 INRAE
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
This module contains a set of python functions to interact with geospatial data
and TensorFlow models.
Starting from OTBTF >= 3.0.0, tricks is only used as a backward compatible stub
for TF 1.X versions.
"""
import tensorflow.compat.v1 as tf
from deprecated import deprecated
from otbtf import gdal_open, read_as_np_arr as read_as_np_arr_from_gdal_ds
tf.disable_v2_behavior()


@deprecated(version="3.0.0", reason="Please use otbtf.read_image_as_np() instead")
def read_image_as_np(filename, as_patches=False):
    """
    Read a patches-image as numpy array.
    :param filename: File name of the patches-image
    :param as_patches: True if the image must be read as patches
    :return 4D numpy array [batch, h, w, c] (batch = 1 when as_patches is False)
    """

    # Open a GDAL dataset
    gdal_ds = gdal_open(filename)

    # Return patches
    return read_as_np_arr_from_gdal_ds(gdal_ds=gdal_ds, as_patches=as_patches)


@deprecated(version="3.0.0", reason="Please consider using TensorFlow >= 2 to build your nets")
def create_savedmodel(sess, inputs, outputs, directory):
    """
    Create a SavedModel from TF 1.X graphs
    :param sess: The Tensorflow V1 session
    :param inputs: List of inputs names (e.g. ["x_cnn_1:0", "x_cnn_2:0"])
    :param outputs: List of outputs names (e.g. ["prediction:0", "features:0"])
    :param directory: Path for the generated SavedModel
    """
    print("Create a SavedModel in " + directory)
    graph = tf.compat.v1.get_default_graph()
    inputs_names = {i: graph.get_tensor_by_name(i) for i in inputs}
    outputs_names = {o: graph.get_tensor_by_name(o) for o in outputs}
    tf.compat.v1.saved_model.simple_save(sess, directory, inputs=inputs_names, outputs=outputs_names)


@deprecated(version="3.0.0", reason="Please consider using TensorFlow >= 2 to build and save your nets")
def ckpt_to_savedmodel(ckpt_path, inputs, outputs, savedmodel_path, clear_devices=False):
    """
    Read a Checkpoint and build a SavedModel for some TF 1.X graph
    :param ckpt_path: Path to the checkpoint file (without the ".meta" extension)
    :param inputs: List of inputs names (e.g. ["x_cnn_1:0", "x_cnn_2:0"])
    :param outputs: List of outputs names (e.g. ["prediction:0", "features:0"])
    :param savedmodel_path: Path for the generated SavedModel
    :param clear_devices: Clear TensorFlow devices positioning (True/False)
    """
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        # Restore variables from disk
        model_saver = tf.compat.v1.train.import_meta_graph(ckpt_path + ".meta", clear_devices=clear_devices)
        model_saver.restore(sess, ckpt_path)

        # Create a SavedModel
        create_savedmodel(sess, inputs=inputs, outputs=outputs, directory=savedmodel_path)


@deprecated(version="3.0.0", reason="Please use otbtf.read_image_as_np() instead")
def read_samples(filename):
    """
   Read a patches image.
   @param filename: raster file name
   """
    return read_image_as_np(filename, as_patches=True)


# Aliases for backward compatibility
CreateSavedModel = create_savedmodel
CheckpointToSavedModel = ckpt_to_savedmodel
