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

# Reference:
#
# Maggiori, E., Tarabalka, Y., Charpiat, G., & Alliez, P. (2016). 
# "Convolutional neural networks for large-scale remote-sensing image classification."
# IEEE Transactions on Geoscience and Remote Sensing, 55(2), 645-657.

from tricks import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", help="Output directory for SavedModel", required=True)
parser.add_argument("--n_channels", type=int, default=4, help="number of channels in the input image")
params = parser.parse_args()

# Build the graph
with tf.Graph().as_default():

  # Size of patches  
  patch_size_xs = 80
  patch_size_label = 16
  
  # placeholder for images and labels
  istraining_placeholder = tf.placeholder_with_default(tf.constant(False , dtype=tf.bool, shape=[]), shape=[], name="is_training")
  learning_rate          = tf.placeholder_with_default(tf.constant(0.0002, dtype=tf.float32, shape=[]), shape=[], name="learning_rate")
  xs_placeholder         = tf.placeholder(tf.float32, shape=(None, patch_size_xs, patch_size_xs, params.n_channels), name="x")
  labels_placeholder     = tf.placeholder(tf.int32,   shape=(None, patch_size_label, patch_size_label, 1),  name="y")
 
  # Convolutional Layer #1 
  conv1 = tf.layers.conv2d(
    inputs=xs_placeholder,
    filters=64,
    kernel_size=[12, 12],
    padding="valid",
    activation=tf.nn.crelu)

  # Normalization of output of layer 1
  norm1 = tf.layers.batch_normalization(conv1)

  # pooling layer #1
  pool1 = tf.layers.max_pooling2d(inputs=norm1, pool_size=[4, 4], strides=4)

  # Convolutional Layer #2
  conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=112,
    kernel_size=[4, 4],
    padding="valid",
    activation=tf.nn.crelu)

   # Normalization of output of layer 2
  norm2 = tf.layers.batch_normalization(conv2)
 
  # Convolutional Layer #3 
  conv3 = tf.layers.conv2d(
    inputs=norm2,
    filters=80,
    kernel_size=[3, 3],
    padding="valid",
    activation=tf.nn.crelu)
 
   # Normalization of output of layer 3
  norm3 = tf.layers.batch_normalization(conv3)

  # Convolutional Layer #4
  conv4 = tf.layers.conv2d(
    inputs=norm3,
    filters=1,
    kernel_size=[8, 8],
    padding="valid",
    activation=tf.nn.crelu)
  print_tensor_info("conv4",conv4)
  
  # Deconv = conv on the padded/strided input, that is an (5+1)*4
  deconv1 = tf.layers.conv2d_transpose(
    inputs=conv4,
    filters=1,
    strides=(4,4),
    kernel_size=[8, 8],
    padding="valid",
    activation=tf.nn.sigmoid)
  print_tensor_info("deconv1",deconv1)
 
  numbatch = tf.shape(deconv1)[0]
  szx =      tf.shape(deconv1)[1]
  szy =      tf.shape(deconv1)[2]
  estimated = tf.slice(deconv1, [0, 4, 4, 0], [numbatch, szx - 8, szy - 8, 1], "estimated")
  print_tensor_info("estimated", estimated)
 
  # Loss
  estimated_resized = tf.reshape(estimated, [-1, patch_size_label*patch_size_label])
  labels_resized = tf.reshape(labels_placeholder, [-1, patch_size_label*patch_size_label])
  labels_resized = tf.cast(labels_resized, tf.float32)
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_resized, logits=estimated_resized))
  
  # Optimizer
  train_op = tf.train.AdamOptimizer(learning_rate, name="optimizer").minimize(loss)
  
  # Initializer, saver, session
  init = tf.global_variables_initializer()
  saver = tf.train.Saver( max_to_keep=20 )
  sess = tf.Session()
  sess.run(init)

  # Let's export a SavedModel
  CreateSavedModel(sess, ["x:0", "y:0", "is_training:0"], ["estimated:0"], params.outdir)