from tricks import *
import sys
import os

nclasses=8

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
  estimated = tf.layers.dense(inputs=features, units=nclasses, activation=None)
  estimated_label = tf.argmax(estimated, 1, name="prediction")
  
  return estimated, estimated_label
 
""" Main """
# check number of arguments
if len(sys.argv) != 2:
  print("Usage : <output directory for SavedModel>")
  sys.exit(1)

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
                                                logits=tf.reshape(y_estimated, [-1, nclasses]))
  
  # Optimizer
  optimizer = tf.train.AdamOptimizer(learning_rate=lr, name="optimizer").minimize(cost)
  
  # Initializer, saver, session
  init = tf.global_variables_initializer()
  saver = tf.train.Saver( max_to_keep=20 )
  sess = tf.Session()
  sess.run(init)

  # Create a SavedModel
  CreateSavedModel(sess, ["x1:0", "x2:0", "y:0"], ["features:0", "prediction:0"], sys.argv[1])
