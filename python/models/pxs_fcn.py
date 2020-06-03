import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--nclasses",  type=int, default=8,  help="number of classes")
parser.add_argument("--out", help="Output SavedModel", required=True)
params = parser.parse_args()

def conv2d(x, kernel_size, filters, strides=1, padding='valid', activation="relu"):
  conv_op = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation,
                                strides=strides)
  return conv_op(x)

def maxpool2x(x):
  pool_op = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
  return pool_op(x)

# Define two sets of inputs
x1 = tf.keras.Input(shape=(16, 16, 4), name="x1")
x2 = tf.keras.Input(shape=(32, 32, 1,), name="x2")
y = tf.keras.Input(shape=(1, 1, 1,), name="y")
lr = tf.keras.Input(shape=(), name="lr")

# The XS branch (input patches: 8x8x4)
conv1_x1 = conv2d(x1, filters=16, kernel_size=5)  # out size: 4x4x16
conv2_x1 = conv2d(conv1_x1, filters=32, kernel_size=3)  # out size: 2x2x32
conv3_x1 = conv2d(conv2_x1, filters=64, kernel_size=2)  # out size: 1x1x64

# The PAN branch (input patches: 32x32x1)
conv1_x2 = conv2d(x2, filters=16, kernel_size=5)  # out size: 28x28x16
pool1_x2 = maxpool2x(conv1_x2)  # out size: 14x14x16
conv2_x2 = conv2d(pool1_x2, filters=32, kernel_size=5)  # out size: 10x10x32
pool2_x2 = maxpool2x(conv2_x2)  # out size: 5x5x32
conv3_x2 = conv2d(pool2_x2, filters=64, kernel_size=3)  # out size: 3x3x64
conv4_x2 = conv2d(conv3_x2, filters=64, kernel_size=3)  # out size: 1x1x64

# Stack features of the two branches
features = tf.keras.backend.stack([conv3_x1, conv4_x2], axis=3)
features = tf.identity(features, "features")

# 8 neurons for 8 classes
estimated = tf.keras.layers.Dense(params.nclasses)(features)
estimated_label = tf.keras.backend.argmax(estimated)
estimated_label = tf.identity(estimated_label, "prediction")

# Loss function
cost = tf.losses.sparse_softmax_cross_entropy(labels=tf.reshape(y, [-1, 1]),
                                              logits=tf.reshape(estimated_label, [-1, params.nclasses]))

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=lr, name="optimizer").minimize(cost)
model = tf.keras.Model(inputs=[x1, x2], outputs=[features, estimated, estimated_label, optimizer])
model.save(params.out, save_format='tf')
