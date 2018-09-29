# -*- coding: utf-8 -*-
#==========================================================================
#
#   Copyright Remi Cresson, Dino Ienco (IRSTEA)
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
from operator import itemgetter, attrgetter, methodcaller
import tensorflow as tf
from tensorflow.contrib import rnn
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from tricks import *
  
def checkTest(ts_data, vhsr_data, batchsz, label_test):
  tot_pred = []
#  gt_test = []
  iterations = ts_data.shape[0] / batchsz

  if ts_data.shape[0] % batchsz != 0:
      iterations+=1

  for ibatch in range(iterations):
    batch_rnn_x, _ = getBatch(ts_data, label_test, ibatch, batchsz)
    
    batch_cnn_x, batch_y = getBatch(vhsr_data, label_test, ibatch, batchsz)

    pred_temp = sess.run(testPrediction,feed_dict={x_rnn:batch_rnn_x,
                             is_training_ph:True,
                             dropout:0.0,
                             x_cnn:batch_cnn_x})

    for el in pred_temp:
      tot_pred.append( el )

  print_histo(np.asarray(tot_pred),"prediction distrib")
  print_histo(label_test,"test distrib")

  # flatten the classes_test
  label_test = flatten_nparray(label_test)

  print "TEST F-Measure: %f" % f1_score(label_test, tot_pred, average='weighted')
  print f1_score(label_test, tot_pred, average=None)
  print "TEST Accuracy: %f" % accuracy_score(label_test, tot_pred)
  sys.stdout.flush()  
  return accuracy_score(label_test, tot_pred)

def RnnAttention(x, nunits, nlayer, n_dims, n_timetamps, is_training_ph):

  N = tf.shape(x)[0] # size of batch
  x = tf.reshape(x, [N, n_dims, n_timetamps])
  # at this point x must be 1 tensor of shape [N, n_dims, n_timestamps]
  
  # (before unstack) x is 1 tensor of shape [N, n_dims, n_timestamps]
  x = tf.unstack(x, n_timetamps, axis=2)
  # (after unstack)  x is a list of "n_timestamps" tensors of shape: [N, n_dims]
  
  #NETWORK DEF
  #MORE THEN ONE LAYER: list of LSTMcell,nunits hidden units each, for each layer
  if nlayer>1:
    cells=[]
    for _ in range(nlayer):
      cell = rnn.GRUCell(nunits)
      cells.append(cell)
    cell = tf.contrib.rnn.MultiRNNCell(cells)
    #SIGNLE LAYER: single GRUCell, nunits hidden units each
  else:
    cell = rnn.GRUCell(nunits)
  outputs,_=rnn.static_rnn(cell, x, dtype="float32")
  # At this point, outputs is a list of "n_timestamps" tensors [N, B, C]
  outputs = tf.stack(outputs, axis=1)
  # At this point, outputs is a tensor of size [N, n_timestamps, B, C]
  
  # Trainable parameters
  attention_size = nunits #int(nunits / 2)
  W_omega = tf.Variable(tf.random_normal([nunits, attention_size], stddev=0.1))
  b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
  u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

  # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
  #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
  v = tf.tanh(tf.tensordot(outputs, W_omega, axes=1) + b_omega)
  # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
  vu = tf.tensordot(v, u_omega, axes=1)   # (B,T) shape
  alphas = tf.nn.softmax(vu)              # (B,T) shape also

  output = tf.reduce_sum(outputs * tf.expand_dims(alphas, -1), 1)
  output = tf.reshape(output, [-1, nunits])

  return output

def CNN(x, nunits):
  #nunits = 512
  
  conv1 = tf.layers.conv2d(
        inputs=x,
        filters=nunits/2, #256
        kernel_size=[7, 7],
        padding="valid",
        activation=tf.nn.relu)
  
  conv1 = tf.layers.batch_normalization(conv1)
  print_tensor_info("conv1", conv1)
  
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  print_tensor_info("pool1", pool1)
  
  conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=nunits,
        kernel_size=[3, 3],
        padding="valid",
        activation=tf.nn.relu)
  
  conv2 = tf.layers.batch_normalization(conv2)

  print_tensor_info("conv2", conv2)
  
  conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=nunits,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
  
  conv3 = tf.layers.batch_normalization(conv3)

  print_tensor_info("conv3", conv3)
  
  conv3 = tf.concat([conv2,conv3],3)
 
  print_tensor_info("conv3 (final)", conv3)
   
  conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=nunits,
        kernel_size=[1, 1],
        padding="valid",
        activation=tf.nn.relu)
  
  conv4 = tf.layers.batch_normalization(conv4)

  print_tensor_info("conv4", conv4)
  
  cnn = tf.reduce_mean(conv4, [1,2])
  
  print_tensor_info("cnn", cnn)

  tensor_shape = cnn.get_shape()

  return cnn, tensor_shape[1].value
  
def getBatch(X, Y, i, batch_size):
    start_id = i*batch_size
    end_id = min( (i+1) * batch_size, X.shape[0])
    batch_x = X[start_id:end_id]
    batch_y = Y[start_id:end_id]
    return batch_x, batch_y

def getPrediction(x_rnn, x_cnn, nunits, nlayer, nclasses, dropout, is_training, n_dims, n_timetamps):
  features_learnt = None

  vec_rnn = RnnAttention(x_rnn, nunits, nlayer, n_dims, n_timetamps, is_training_ph)
  vec_cnn, cnn_dim = CNN(x_cnn, 512)    
  
  features_learnt=tf.concat([vec_rnn,vec_cnn],axis=1, name="features")
  first_dim = cnn_dim + nunits
  
  #Classifier1 #RNN Branch
  print "RNN Features:"
  print vec_rnn.get_shape()
  outb1 = tf.Variable(tf.truncated_normal([nclasses]),name='B1')
  outw1 = tf.Variable(tf.truncated_normal([nunits,nclasses]),name='W1')  
  pred_c1 = tf.matmul(vec_rnn,outw1)+outb1
  
  #Classifier2 #CNN Branch
  print "CNN Features:"
  print vec_cnn.get_shape()
  outb2 = tf.Variable(tf.truncated_normal([nclasses]),name='B2')
  outw2 = tf.Variable(tf.truncated_normal([cnn_dim,nclasses]),name='W2')  
  pred_c2 = tf.matmul(vec_cnn,outw2)+outb2
  
  #ClassifierFull
  print "FULL features_learnt:"
  print features_learnt.get_shape()
  outb = tf.Variable(tf.truncated_normal([nclasses]),name='B')
  outw = tf.Variable(tf.truncated_normal([first_dim,nclasses]),name='W')  
  pred_full = tf.matmul(features_learnt,outw)+outb
      
  return pred_c1, pred_c2, pred_full, features_learnt

###############################################################################

#Model parameters
nunits = 1024
batchsz = 64
hm_epochs = 400
n_levels_lstm = 1
#dropout = 0.2

#Data INformation
n_timestamps = 37
n_dims       = 16
patch_window = 25
n_channels   = 4
nclasses     = 8

# check number of arguments
if len(sys.argv) != 8:
  print("Usage : <ts_train> <vhs_train> <label_train> <ts_valid> <vhs_valid> <label_valid> <export_dir>")
  sys.exit(1)

ts_train    = read_samples(sys.argv[1])
vhsr_train  = read_samples(sys.argv[2])
label_train = read_samples(sys.argv[3])
label_train = np.int32(label_train)
print_histo(label_train, "label_train")

ts_test     = read_samples(sys.argv[4])
vhsr_test   = read_samples(sys.argv[5])
label_test  = read_samples(sys.argv[6])
label_test = np.int32(label_test)
print_histo(label_test, "label_test")

export_dir = read_samples(sys.argv[7])

x_rnn = tf.placeholder(tf.float32,[None, 1, 1, n_dims*n_timestamps],name="x_rnn")
x_cnn = tf.placeholder(tf.float32,[None, patch_window, patch_window, n_channels],name="x_cnn")
y     = tf.placeholder(tf.int32,[None, 1, 1, 1],name="y")

learning_rate  = tf.placeholder(tf.float32, shape=(), name="learning_rate")
is_training_ph = tf.placeholder(tf.bool,    shape=(), name="is_training")
dropout        = tf.placeholder(tf.float32, shape=(), name="drop_rate")

sess = tf.InteractiveSession()

pred_c1, pred_c2, pred_full, features_learnt = getPrediction(x_rnn, 
                                                             x_cnn,
                                                             nunits, 
                                                             n_levels_lstm, 
                                                             nclasses, 
                                                             dropout, 
                                                             is_training_ph,
                                                             n_dims,
                                                             n_timestamps)

testPrediction = tf.argmax(pred_full, 1, name="prediction")

loss_full = tf.losses.sparse_softmax_cross_entropy(labels=tf.reshape(y, [-1, 1]), logits=tf.reshape(pred_full, [-1, nclasses]))
loss_c1 = tf.losses.sparse_softmax_cross_entropy(labels=tf.reshape(y, [-1, 1]), logits=tf.reshape(pred_c1, [-1, nclasses]))
loss_c2 = tf.losses.sparse_softmax_cross_entropy(labels=tf.reshape(y, [-1, 1]), logits=tf.reshape(pred_c2, [-1, nclasses]))

cost = loss_full + (0.3 * loss_c1) + (0.3 * loss_c2)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="optimizer").minimize(cost)

correct = tf.equal(tf.argmax(pred_full,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct,tf.float64))

tf.global_variables_initializer().run()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

iterations = ts_train.shape[0] / batchsz

if ts_train.shape[0] % batchsz != 0:
    iterations+=1

best_loss = sys.float_info.max

for e in range(hm_epochs):
  lossi = 0
  accS = 0
  
  ts_train, vhsr_train, label_train = shuffle(ts_train, vhsr_train, label_train, random_state=0)
  print "shuffle DONE"
  
  
  for ibatch in range(iterations):
    #BATCH_X BATCH_Y: i-th batches of train_indices_x and train_y
    batch_rnn_x, _ = getBatch(ts_train, label_train, ibatch, batchsz)
    batch_cnn_x, batch_y = getBatch(vhsr_train, label_train, ibatch, batchsz)

    acc,_,loss = sess.run([accuracy,optimizer,cost],feed_dict={x_rnn:batch_rnn_x,
                                  x_cnn:batch_cnn_x,
                                  y:batch_y,
                                  is_training_ph:True,
                                  dropout:0.2,
                                  learning_rate:0.0002})    
    lossi+=loss
    accS+=acc
    
  print "Epoch:",e,"Train loss:",lossi/iterations,"| accuracy:",accS/iterations
  
  c_loss = lossi/iterations
  
  if c_loss < best_loss:
    best_loss = c_loss
    CreateSavedModel(sess, ["x_cnn:0","x_rnn:0","is_training:0"], ["prediction:0"], export_dir)

  test_acc = checkTest(ts_test, vhsr_test, 1024, label_test)