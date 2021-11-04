#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================================
#
#   Copyright 2018-2019 Remi Cresson, Dino Ienco (IRSTEA)
#   Copyright 2020-2021 Remi Cresson, Dino Ienco (INRAE)
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
# =========================================================================

# Reference:
#
# Benedetti, P., Ienco, D., Gaetano, R., Ose, K., Pensa, R. G., & Dupuy, S. (2018)
# M3Fusion: A Deep Learning Architecture for Multiscale Multimodal Multitemporal
# Satellite Data Fusion. IEEE Journal of Selected Topics in Applied Earth
# Observations and Remote Sensing, 11(12), 4939-4949.

import argparse
from tricks import create_savedmodel
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.nn.rnn_cell as rnn

tf.disable_v2_behavior()

parser = argparse.ArgumentParser()
parser.add_argument("--nunits", type=int, default=1024, help="number of units")
parser.add_argument("--n_levels_lstm", type=int, default=1, help="number of lstm levels")
parser.add_argument("--hm_epochs", type=int, default=400, help="hm epochs")
parser.add_argument("--n_timestamps", type=int, default=37, help="number of images in timeseries")
parser.add_argument("--n_dims", type=int, default=16, help="number of channels in timeseries images")
parser.add_argument("--patch_window", type=int, default=25, help="patch size for the high-res image")
parser.add_argument("--n_channels", type=int, default=4, help="number of channels in the high-res image")
parser.add_argument("--nclasses", type=int, default=8, help="number of classes")
parser.add_argument("--outdir", help="Output directory for SavedModel", required=True)
params = parser.parse_args()


def RnnAttention(x, nunits, nlayer, n_dims, n_timetamps, is_training_ph):
    N = tf.shape(x)[0]  # size of batch
    x = tf.reshape(x, [N, n_dims, n_timetamps])
    # at this point x must be 1 tensor of shape [N, n_dims, n_timestamps]

    # (before unstack) x is 1 tensor of shape [N, n_dims, n_timestamps]
    x = tf.unstack(x, n_timetamps, axis=2)
    # (after unstack)  x is a list of "n_timestamps" tensors of shape: [N, n_dims]

    # NETWORK DEF
    # MORE THEN ONE LAYER: list of LSTMcell,nunits hidden units each, for each layer
    if nlayer > 1:
        cells = []
        for _ in range(nlayer):
            cell = rnn.GRUCell(nunits)
            cells.append(cell)
        cell = tf.compat.v1.contrib.rnn.MultiRNNCell(cells)
        # SINGLE LAYER: single GRUCell, nunits hidden units each
    else:
        cell = rnn.GRUCell(nunits)
    outputs, _ = tf.compat.v1.nn.static_rnn(cell, x, dtype="float32")
    # At this point, outputs is a list of "n_timestamps" tensors [N, B, C]
    outputs = tf.stack(outputs, axis=1)
    # At this point, outputs is a tensor of size [N, n_timestamps, B, C]

    # Trainable parameters
    attention_size = nunits  # int(nunits / 2)
    W_omega = tf.Variable(tf.random_normal([nunits, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
    v = tf.tanh(tf.tensordot(outputs, W_omega, axes=1) + b_omega)
    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1)  # (B,T) shape
    alphas = tf.nn.softmax(vu)  # (B,T) shape also

    output = tf.reduce_sum(outputs * tf.expand_dims(alphas, -1), 1)
    output = tf.reshape(output, [-1, nunits])

    return output


def CNN(x, nunits):
    # nunits = 512

    conv1 = tf.compat.v1.layers.conv2d(
        inputs=x,
        filters=nunits / 2,  # 256
        kernel_size=[7, 7],
        padding="valid",
        activation=tf.nn.relu)

    conv1 = tf.compat.v1.layers.batch_normalization(conv1)

    pool1 = tf.compat.v1.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.compat.v1.layers.conv2d(
        inputs=pool1,
        filters=nunits,
        kernel_size=[3, 3],
        padding="valid",
        activation=tf.nn.relu)

    conv2 = tf.compat.v1.layers.batch_normalization(conv2)

    conv3 = tf.compat.v1.layers.conv2d(
        inputs=conv2,
        filters=nunits,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    conv3 = tf.compat.v1.layers.batch_normalization(conv3)

    conv3 = tf.compat.v1.concat([conv2, conv3], 3)

    conv4 = tf.compat.v1.layers.conv2d(
        inputs=conv3,
        filters=nunits,
        kernel_size=[1, 1],
        padding="valid",
        activation=tf.nn.relu)

    conv4 = tf.compat.v1.layers.batch_normalization(conv4)

    cnn = tf.reduce_mean(conv4, [1, 2])

    tensor_shape = cnn.get_shape()

    return cnn, tensor_shape[1].value


def get_prediction(x_rnn, x_cnn, nunits, nlayer, nclasses, n_dims, n_timetamps):
    vec_rnn = RnnAttention(x_rnn, nunits, nlayer, n_dims, n_timetamps, is_training_ph)
    vec_cnn, cnn_dim = CNN(x_cnn, 512)

    features_learnt = tf.concat([vec_rnn, vec_cnn], axis=1, name="features")
    first_dim = cnn_dim + nunits

    # Classifier1 #RNN Branch
    outb1 = tf.Variable(tf.truncated_normal([nclasses]), name='B1')
    outw1 = tf.Variable(tf.truncated_normal([nunits, nclasses]), name='W1')
    pred_c1 = tf.matmul(vec_rnn, outw1) + outb1

    # Classifier2 #CNN Branch
    outb2 = tf.Variable(tf.truncated_normal([nclasses]), name='B2')
    outw2 = tf.Variable(tf.truncated_normal([cnn_dim, nclasses]), name='W2')
    pred_c2 = tf.matmul(vec_cnn, outw2) + outb2

    # ClassifierFull
    outb = tf.Variable(tf.truncated_normal([nclasses]), name='B')
    outw = tf.Variable(tf.truncated_normal([first_dim, nclasses]), name='W')
    pred_full = tf.matmul(features_learnt, outw) + outb

    return pred_c1, pred_c2, pred_full, features_learnt


# Create the TensorFlow graph
with tf.compat.v1.Graph().as_default():
    x_rnn = tf.compat.v1.placeholder(tf.float32, [None, 1, 1, params.n_dims * params.n_timestamps], name="x_rnn")
    x_cnn = tf.compat.v1.placeholder(tf.float32, [None, params.patch_window, params.patch_window, params.n_channels],
                                     name="x_cnn")
    y = tf.compat.v1.placeholder(tf.int32, [None, 1, 1, 1], name="y")

    learning_rate = tf.compat.v1.placeholder_with_default(tf.constant(0.0002, dtype=tf.float32, shape=[]), shape=[],
                                                          name="learning_rate")
    is_training_ph = tf.compat.v1.placeholder_with_default(tf.constant(False, dtype=tf.bool, shape=[]), shape=[],
                                                           name="is_training")
    dropout = tf.compat.v1.placeholder_with_default(tf.constant(0.5, dtype=tf.float32, shape=[]), shape=[],
                                                    name="drop_rate")

    pred_c1, pred_c2, pred_full, features_learnt = get_prediction(x_rnn,
                                                                  x_cnn,
                                                                  params.nunits,
                                                                  params.n_levels_lstm,
                                                                  params.nclasses,
                                                                  params.n_dims,
                                                                  params.n_timestamps)

    testPrediction = tf.argmax(pred_full, 1, name="prediction")

    loss_full = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=tf.reshape(y, [-1, 1]),
                                                                 logits=tf.reshape(pred_full, [-1, params.nclasses]))
    loss_c1 = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=tf.reshape(y, [-1, 1]),
                                                               logits=tf.reshape(pred_c1, [-1, params.nclasses]))
    loss_c2 = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=tf.reshape(y, [-1, 1]),
                                                               logits=tf.reshape(pred_c2, [-1, params.nclasses]))

    cost = loss_full + (0.3 * loss_c1) + (0.3 * loss_c2)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, name="optimizer").minimize(cost)

    correct = tf.equal(tf.argmax(pred_full, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float64))

    # Initializer, saver, session
    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver(max_to_keep=20)
    sess = tf.compat.v1.Session()
    sess.run(init)

    create_savedmodel(sess, ["x_cnn:0", "x_rnn:0", "y:0"], ["prediction:0"], params.outdir)
