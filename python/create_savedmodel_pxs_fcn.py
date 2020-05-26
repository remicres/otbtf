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
import argparse
from tricks import create_savedmodel
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

parser = argparse.ArgumentParser()
parser.add_argument("--nclasses", type=int, default=8, help="number of classes")
parser.add_argument("--outdir", help="Output directory for SavedModel", required=True)
params = parser.parse_args()


def conv2d_valid(x, kernel_size, filters, activation="relu"):
    conv_op = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation)
    return conv_op(x)


def my_model(x1, x2):
    max_pool_2x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

    # The XS branch (input patches: 8x8x4)
    conv1_x1 = conv2d_valid(x1, filters=16, kernel_size=5)  # out size: 4x4x16
    conv2_x1 = conv2d_valid(conv1_x1, filters=32, kernel_size=3)  # out size: 2x2x32
    conv3_x1 = conv2d_valid(conv2_x1, filters=64, kernel_size=2)  # out size: 1x1x64

    # The PAN branch (input patches: 32x32x1)
    conv1_x2 = conv2d_valid(x2, filters=16, kernel_size=5)  # out size: 28x28x16
    pool1_x2 = max_pool_2x(conv1_x2)  # out size: 14x14x16
    conv2_x2 = conv2d_valid(pool1_x2, filters=32, kernel_size=5)  # out size: 10x10x32
    pool2_x2 = max_pool_2x(conv2_x2)  # out size: 5x5x32
    conv3_x2 = conv2d_valid(pool2_x2, filters=64, kernel_size=3)  # out size: 3x3x64
    conv4_x2 = conv2d_valid(conv3_x2, filters=64, kernel_size=3)  # out size: 1x1x64

    # Stack features
    features = tf.reshape(tf.stack([conv3_x1, conv4_x2], axis=3), shape=[-1, 128], name="features")

    # Neurons for classes
    estimated = tf.keras.layers.Dense(params.nclasses)(features)
    estimated_label = tf.argmax(estimated, name="prediction")

    return estimated, estimated_label


# Create the graph
with tf.compat.v1.Graph().as_default():
    # Placeholders
    x1 = tf.compat.v1.placeholder(tf.float32, [None, None, None, 4], name="x1")
    x2 = tf.compat.v1.placeholder(tf.float32, [None, None, None, 1], name="x2")
    y = tf.compat.v1.placeholder(tf.int32, [None, None, None, 1], name="y")
    lr = tf.compat.v1.placeholder_with_default(tf.constant(0.0002, dtype=tf.float32, shape=[]), shape=[], name="lr")

    # Output
    y_estimated, y_label = my_model(x1, x2)

    # Loss function
    cost = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=tf.reshape(y, [-1, 1]),
                                                            logits=tf.reshape(y_estimated, [-1, params.nclasses]))

    # Optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, name="optimizer").minimize(cost)

    # Initializer, saver, session
    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver(max_to_keep=20)
    sess = tf.compat.v1.Session()
    sess.run(init)

    # Create a SavedModel
    create_savedmodel(sess, ["x1:0", "x2:0", "y:0"], ["features:0", "prediction:0"], params.outdir)
