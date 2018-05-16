import sys
import os
import numpy as np
import math
import time
import otbApplication

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
  
def read_samples(fn):
  """ Read an image of patches and return a 4D numpy array
  Args:
    fn: file name
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
