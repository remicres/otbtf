from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import numpy as np
import random
import os
import shutil
import sys
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

from tricks import *

tf.logging.set_verbosity(tf.logging.INFO)

def do_eval2(sess,
            testPrediction,
            xs_placeholder,
            labels_placeholder,
            istraining_placeholder,
            ds_data_xs,
            ds_labels,
            batch_size):
  """Runs one evaluation against the full epoch of data.
  """
  # total predictions
  tot_pred = np.array([])
  
  # And run one epoch of eval.
  n_data = ds_data_xs.shape[0]
  n_steps = int(n_data / batch_size)
  for step in range(0, n_steps):
    start_idx = batch_size * step
    end_idx = start_idx + batch_size
    if step == n_steps - 1:
        end_idx = n_data
        print(end_idx)

    feed_dict = {
        xs_placeholder: ds_data_xs[start_idx:end_idx,:],
        labels_placeholder: ds_labels[start_idx:end_idx],
        istraining_placeholder: False,
    }
    ( predicted_y) = sess.run(testPrediction, feed_dict=feed_dict)
    
    if tot_pred.size == 0:
      tot_pred = predicted_y
    else:
      tot_pred = np.concatenate( (tot_pred, predicted_y), axis=0)

  print(ds_labels.shape)
  print(tot_pred.shape)
    
  tot_pred = tot_pred.reshape((tot_pred.size))
  ds_labels = ds_labels.reshape((ds_labels.size))
  print(tot_pred.shape)

  print("F-Measure: "+str(f1_score(ds_labels, tot_pred, average='weighted')))
  print("F1 score: " + str(f1_score(ds_labels, tot_pred, average=None)))
  print("Accuracy: "+str(accuracy_score(ds_labels, tot_pred)))

def main(unused_argv):
  """ Main function
  In this function we do:
    1. Import a dataset
    2. Build a model implementing a CNN
    3. Perform training of the CNN
    4. Export the model
  """

  # Training params
  n_epoch = 100
  batch_size = 32
  learning_rate = 0.0001

  ############################################################
  #                     import a dataset 
  ############################################################


  # check number of arguments
  if len(sys.argv) != 4:
    print("Usage : <patches> <labels> <output_model_dir>")
    sys.exit(1)

  # Export dir
  log_dir = sys.argv[3] + '/model_checkpoints/'
  export_dir = sys.argv[3] + '/model_export/'

  print("loading dataset")

  # Create a dataset of size imp_size
  imp_ds_patches = read_samples(sys.argv[1])
  imp_ds_labels  = read_samples(sys.argv[2])
  
  # Shuffle the dataset
  imp_ds_patches,imp_ds_labels = shuffle(imp_ds_patches,imp_ds_labels, random_state=0)
  
  print("ok")
  
   # Number of samples
  if (imp_ds_patches.shape[0] != imp_ds_labels.shape[0]):
    print("Number of samples should be the same as number of patches!")
    sys.exit(1)

  # Number of samples for training  
  n_data_train = int(imp_ds_patches.shape[0] / 2)

  # Size of patches  
  nb_bands = 4
  patch_size_xs = 80
  patch_size_label = 16
  
  # Divide the dataset in two subdatasets : training and validation
  ds_data_train = imp_ds_patches[0:n_data_train,:]
  ds_data_valid = imp_ds_patches[n_data_train:imp_ds_patches.shape[0],:]
  ds_labels_train = imp_ds_labels[0:n_data_train,:]
  ds_labels_valid = imp_ds_labels[n_data_train:imp_ds_patches.shape[0],:]

  ############################################################
  #                    Build the graph
  ############################################################
  
  with tf.Graph().as_default():

    # placeholder for images and labels
    istraining_placeholder = tf.placeholder(tf.bool, shape=(), name="istraining") # Used only for dropout...
    xs_placeholder = tf.placeholder(tf.float32, shape=(None, patch_size_xs, patch_size_xs, nb_bands), name="x1")
    labels_placeholder = tf.placeholder(tf.int32, shape=(None, patch_size_label, patch_size_label, 1), name="y1")
 
    print_tensor_info("xs_placeholder",xs_placeholder)
 
    # Convolutional Layer #1 
    conv1 = tf.layers.conv2d(
      inputs=xs_placeholder,
      filters=32,
      kernel_size=[12, 12],
      padding="valid",
      activation=tf.nn.crelu)
    print_tensor_info("conv1",conv1)
  
    # Normalization of output of layer 1
    norm1 = tf.layers.batch_normalization(conv1)
    print_tensor_info("norm1",norm1)
  
    # pooling layer #1
    pool1 = tf.layers.max_pooling2d(inputs=norm1, pool_size=[4, 4], strides=4)
    print_tensor_info("pool1",pool1)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=56,
      kernel_size=[4, 4],
      padding="valid",
      activation=tf.nn.crelu)
    print_tensor_info("conv2",conv2)

     # Normalization of output of layer 2
    norm2 = tf.layers.batch_normalization(conv2)
    print_tensor_info("norm2",norm2)
   
    # Convolutional Layer #3 
    conv3 = tf.layers.conv2d(
      inputs=norm2,
      filters=40,
      kernel_size=[3, 3],
      padding="valid",
      activation=tf.nn.crelu)
    print_tensor_info("conv3",conv3)
 
     # Normalization of output of layer 3
    norm3 = tf.layers.batch_normalization(conv3)
    print_tensor_info("norm3",norm3)
  
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
   
    ##################### Loss calculation ###################

    estimated_resized = tf.reshape(estimated, [-1, patch_size_label*patch_size_label])
    labels_resized = tf.reshape(labels_placeholder, [-1, patch_size_label*patch_size_label])
    labels_resized = tf.cast(labels_resized, tf.float32)
    
    testPrediction = tf.to_int32(estimated > 0.5)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_resized, logits=estimated_resized))
    
    ###################### Gradient descent ##################
    
    train_op = tf.train.AdamOptimizer(learning_rate, name="optimizer").minimize(loss)
    
    ############### Variable initializer Op ##################

    init = tf.global_variables_initializer()
    
    ######################### Saver ##########################

    saver = tf.train.Saver()
    
    #################### Create a session ####################
    
    sess = tf.Session()
    sess.run(init)

    ############## Here we start the training ################
    
    for curr_epoch in range(n_epoch):

      print("Epoch #" + str(curr_epoch))
      
      ds_data_train,ds_labels_train = shuffle(ds_data_train,ds_labels_train, random_state=0)
  
      # Start the training loop.
      n_steps = int(n_data_train / batch_size)
      for step in range(n_steps):
  
        start_time = time.time()
        
        # Fill a feed dictionary with the actual set of images and labels
        # for this particular training step.
        # Load training and eval data
        start_idx = batch_size * step
        end_idx = start_idx + batch_size 
        feed_dict = {
          xs_placeholder: ds_data_train[start_idx:end_idx,:],
          labels_placeholder: ds_labels_train[start_idx:end_idx,:],
          istraining_placeholder: True,
        }
  
        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

        duration = time.time() - start_time
  
        # Print an overview fairly often.
        if step % 10 == 0:
          # Print status to stdout.
          print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
          #print('Step %d: (%.3f sec)' % (step, duration))


      # Save a checkpoint and evaluate the model periodically.
      if (curr_epoch + 1) % 1 == 0:
        checkpoint_file = os.path.join(log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_eval2(sess,
                testPrediction,
                xs_placeholder,
                labels_placeholder,
                istraining_placeholder,
                ds_data_train,
                ds_labels_train,
                batch_size)
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        do_eval2(sess,
                testPrediction,
                xs_placeholder,
                labels_placeholder,
                istraining_placeholder,
                ds_data_valid,
                ds_labels_valid,
                batch_size)
                
        # Let's export a SavedModel
        shutil.rmtree(export_dir)
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        signature_def_map= {
        "model": tf.saved_model.signature_def_utils.predict_signature_def(
            inputs= {"x1": xs_placeholder},
            outputs= {"prediction": testPrediction})
        }
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING], signature_def_map)
        builder.add_meta_graph([tf.saved_model.tag_constants.SERVING])
        builder.save()

  quit()
  
if __name__ == "__main__":
  
  tf.add_check_numerics_ops()
  tf.app.run(main)
