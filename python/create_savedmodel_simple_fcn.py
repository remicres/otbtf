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

def myModel(x):
  
  # input patches: 16x16x4
  conv1 = tf.layers.conv2d(inputs=x, filters=16, kernel_size=[5,5], padding="valid", 
                           activation=tf.nn.relu) # out size: 12x12x16
  conv2 = tf.layers.conv2d(inputs=conv1, filters=16, kernel_size=[5,5], padding="valid", 
                           activation=tf.nn.relu) # out size: 8x8x16
  conv3 = tf.layers.conv2d(inputs=conv2, filters=32, kernel_size=[5,5], padding="valid",
                           activation=tf.nn.relu) # out size: 4x4x32
  conv4 = tf.layers.conv2d(inputs=conv3, filters=32, kernel_size=[4,4], padding="valid",
                           activation=tf.nn.relu) # out size: 1x1x32
  
  # Features
  features = tf.reshape(conv4, shape=[-1, 32], name="features")
  
  # 8 neurons for 8 classes
  estimated = tf.layers.dense(inputs=features, units=params.nclasses, activation=None)
  estimated_label = tf.argmax(estimated, 1, name="prediction")

  return estimated, estimated_label

# Create the TensorFlow graph
with tf.Graph().as_default():
  
  # Placeholders
  x = tf.placeholder(tf.float32, [None, None, None, 4], name="x")
  y = tf.placeholder(tf.int32  , [None, None, None, 1], name="y")
  lr = tf.placeholder_with_default(tf.constant(0.0002, dtype=tf.float32, shape=[]),
                                   shape=[], name="lr")
  
  # Output
  y_estimated, y_label = myModel(x)
  
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
  CreateSavedModel(sess, ["x:0", "y:0"], ["features:0", "prediction:0"], params.outdir)
