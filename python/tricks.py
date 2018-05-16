import sys
import os
import numpy as np
import math
from operator import itemgetter, attrgetter, methodcaller
import tensorflow as tf
#from tensorflow.contrib import rnn
#from tensorflow.contrib.rnn import DropoutWrapper
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import time
import otbApplication
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

def conv_reduce_mean(input_tensor, shape, strides=[1,1,1,1]):
	"""
	This function replaces the tf.reduce_mean with a 2D convolution, to enable the network to perform full convolution

	Example:
		# "y" is a -1x3x3x128 tensor, and we want a -1x128 tensor of means of all 3x3 slices of y in dimension [1,2] :
		# We can do :
		features = tf.reduce_mean(y ,[1,2])

		# But to enable the full convolution, we can do:
		features = conv_reduce_mean(y, [3,3,128,1])
	
	"""
	filt = tf.fill(shape, 1.0 / (shape[0] * shape[1]))
	features = tf.nn.depthwise_conv2d(input_tensor, filt, strides=strides, padding="VALID")
	return tf.reshape(features, [-1, shape[2]])


def conv_reduce_max(input_tensor, ksize, nfeat):
	"""
	This function replaces the tf.reduce_max with a 2D max pooling, to enable the network to perform full convolution

	Example:
		# "y" is a -1x3x3x128 tensor, and we want a -1x128 tensor of maximum values of all 3x3 slices of y in dimension [1,2] :
		# We can do :
		features = tf.reduce_max(y ,[1,2])

		# But to enable the full convolution, we can do:
		features = conv_reduce_max(y, [1,3,3,1], 128)
	
	"""
	features = tf.nn.max_pool(input_tensor, ksize, strides=[1, 1, 1, 1], padding="VALID")

	return tf.reshape(features, [-1, nfeat])


def getBatch(X, Y, i, batch_size):
	start_id = i*batch_size
	end_id = min( (i+1) * batch_size, X.shape[0])
	batch_x = X[start_id:end_id]
	batch_y = Y[start_id:end_id]
 
	return batch_x, batch_y
