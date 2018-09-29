# -*- coding: utf-8 -*-
#==========================================================================
#
#   Copyright Remi Cresson (IRSTEA)
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
#==========================================================================*/
import sys
import os
import numpy as np
import math
import time
import otbApplication
import tensorflow as tf
import shutil

def flatten_nparray(np_arr):
  """ Returns a 1D numpy array retulting from the flatten of the input
  """
  return np_arr.reshape((len(np_arr)))

def print_histo(np_arr, title=""):
  """ Prints the histogram of the input numpy array
  """
  np_flat = flatten_nparray(np_arr)
  np_hist = np.bincount(np_flat)
  np_vals = np.unique(np_flat)
  if (len(title) > 0):
    print(title + ":")
  print("Values : "+str(np_vals))
  print("Count  : "+str(np_hist))
  
def print_tensor_live(name, tensor):
  """ Print the shape of a tensor during a session run
  """
  return tf.Print(tensor, [tf.shape(tensor)], name + " shape")

def print_tensor_info(name, tensor):
  """ Print the shape of a tensor
  Args:
    name : the tensor's name (as we want it to be displayed)
    tensor : the tensor 
  """

  print(name+" : "+str(tensor.shape)+" (dtype="+str(tensor.dtype)+")")
  
def read_samples(fn, single=False):
  """ Read an image of patches and return a 4D numpy array
  TODO: Add an optional argument for the y-patchsize
  Args:
    fn: file name
    single: a boolean telling if there is only 1 image in the batch. 
            In this case, the image can be rectangular (not squared)
  """

  # Get input image size
  imageInfo = otbApplication.Registry.CreateApplication('ReadImageInfo')
  imageInfo.SetParameterString('in', fn)
  imageInfo.Execute()
  size_x = imageInfo.GetParameterInt('sizex')
  size_y = imageInfo.GetParameterInt('sizey')
  nbands = imageInfo.GetParameterInt('numberbands')

  print("Loading image "+str(fn)+" ("+str(size_x)+" x "+str(size_y)+" x "+str(nbands)+")")
  
  # Prepare the PixelValue application
  imageReader = otbApplication.Registry.CreateApplication('ExtractROI')
  imageReader.SetParameterString('in', fn)
  imageReader.SetParameterInt('sizex', size_x)
  imageReader.SetParameterInt('sizey', size_y)
  imageReader.Execute()
  outimg=imageReader.GetVectorImageAsNumpyArray('out', 'float')
  
  # quick stats
  print("Quick stats: min="+str(np.amin(outimg))+", max="+str(np.amax(outimg)) )
  
  # reshape
  if (single):
    return np.copy(outimg.reshape((1, size_y, size_x, nbands)))

  n_samples = int(size_y / size_x)
  outimg = outimg.reshape((n_samples, size_x, size_x, nbands))
  
  print("Returned numpy array shape: "+str(outimg.shape))
  return np.copy(outimg)
  
def getBatch(X, Y, i, batch_size):
	start_id = i*batch_size
	end_id = min( (i+1) * batch_size, X.shape[0])
	batch_x = X[start_id:end_id]
	batch_y = Y[start_id:end_id]
 
	return batch_x, batch_y

def CreateSavedModel(sess, inputs, outputs, directory):
  """
  Create a SavedModel
  
  Args:
    sess: the session
    inputs: the list of input names
    outputs: the list of output names
    directory: the output path for the SavedModel
  """

  print("Create a SavedModel in " + directory)

      
  # Get graph
  graph = tf.get_default_graph()
  
  # Get inputs
  input_dict = { i : graph.get_tensor_by_name(i) for i in inputs }
  output_dict = { o : graph.get_tensor_by_name(o) for o in outputs }  
        
  # Build the SavedModel
  builder = tf.saved_model.builder.SavedModelBuilder(directory)
  signature_def_map= {
    "model": tf.saved_model.signature_def_utils.predict_signature_def(
    input_dict,
    output_dict)
  }
  builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.TRAINING],signature_def_map)
  builder.add_meta_graph([tf.saved_model.tag_constants.SERVING])
  builder.save()

def CheckpointToSavedModel(ckpt_path, inputs, outputs, savedmodel_path):
  """
  Read a Checkpoint and build a SavedModel
  
  Args:
    ckpt_path: path to the checkpoint file (without the ".meta" extension)
    inputs: input list of placeholders names (e.g. ["x_cnn_1:0", "x_cnn_2:0"])
    outputs: output list of tensor outputs names (e.g. ["prediction:0", "features:0"])
    savedmodel_path: path to the SavedModel
  """
  tf.reset_default_graph()
  with tf.Session() as sess:
    
    # Restore variables from disk.
    model_saver = tf.train.import_meta_graph(ckpt_path+".meta")
    model_saver.restore(sess, ckpt_path)
    
    # Create a SavedModel
    CreateSavedModel(sess, inputs, outputs, savedmodel_path)