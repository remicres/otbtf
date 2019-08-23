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
from tricks import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--nclasses",  type=int, default=8,  help="number of classes")
parser.add_argument("--outdir", help="Output directory for SavedModel", required=True)
params = parser.parse_args()

def myModel(x1,x2):
  
  # The XS branch (input patches: 8x8x4)
  conv1_x1 = tf.layers.conv2d(inputs=x1, filters=16, kernel_size=[5,5], padding="valid",
                              activation=tf.nn.relu) # out size: 4x4x16
  conv2_x1 = tf.layers.conv2d(inputs=conv1_x1, filters=32, kernel_size=[3,3], padding="valid",
                              activation=tf.nn.relu) # out size: 2x2x32
  conv3_x1 = tf.layers.conv2d(inputs=conv2_x1, filters=64, kernel_size=[2,2], padding="valid",
                              activation=tf.nn.relu) # out size: 1x1x64
  
  # The PAN branch (input patches: 32x32x1)
  conv1_x2 = tf.layers.conv2d(inputs=x2, filters=16, kernel_size=[5,5], padding="valid",
                              activation=tf.nn.relu) # out size: 28x28x16
  pool1_x2 = tf.layers.max_pooling2d(inputs=conv1_x2, pool_size=[2, 2], 
                              strides=2) # out size: 14x14x16
  conv2_x2 = tf.layers.conv2d(inputs=pool1_x2, filters=32, kernel_size=[5,5], padding="valid",
                              activation=tf.nn.relu) # out size: 10x10x32
  pool2_x2 = tf.layers.max_pooling2d(inputs=conv2_x2, pool_size=[2, 2],
                              strides=2) # out size: 5x5x32
  conv3_x2 = tf.layers.conv2d(inputs=pool2_x2, filters=64, kernel_size=[3,3], padding="valid",
                              activation=tf.nn.relu) # out size: 3x3x64
  conv4_x2 = tf.layers.conv2d(inputs=conv3_x2, filters=64, kernel_size=[3,3], padding="valid",
                              activation=tf.nn.relu) # out size: 1x1x64
  
  # Stack features
  features = tf.reshape(tf.stack([conv3_x1, conv4_x2], axis=3), 
                        shape=[-1, 128], name="features")
  
  # 8 neurons for 8 classes
  estimated = tf.layers.dense(inputs=features, units=params.nclasses, activation=None)
  estimated_label = tf.argmax(estimated, 1, name="prediction")
  
  return estimated, estimated_label
 
# Create the graph
with tf.Graph().as_default():
  
  # Placeholders
  x1 = tf.placeholder(tf.float32, [None, None, None, 4], name="x1")
  x2 = tf.placeholder(tf.float32, [None, None, None, 1], name="x2")
  y  = tf.placeholder(tf.int32  , [None, None, None, 1], name="y")
  lr = tf.placeholder_with_default(tf.constant(0.0002, dtype=tf.float32, shape=[]), 
                                   shape=[], name="lr")
  
  # Output
  y_estimated, y_label = myModel(x1,x2)
  
  # Loss function
  cost = tf.losses.sparse_softmax_cross_entropy(labels=tf.reshape(y, [-1, 1]), 
                                                logits=tf.reshape(y_estimated, [-1, params.nclasses]))
  
  # Optimizer
  optimizer = tf.train.AdamOptimizer(learning_rate=lr, name="optimizer").minimize(cost)
  
  # Initializer, saver, session
  init = tf.global_variables_initializer()
  saver = tf.train.Saver( max_to_keep=20 )
  sess = tf.Session()
  sess.run(init)

  # Create a SavedModel
  CreateSavedModel(sess, ["x1:0", "x2:0", "y:0"], ["features:0", "prediction:0"], params.outdir)
