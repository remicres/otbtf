"""
This test checks that the output tensor shapes are supported.
The input of this model must be a mono channel image.
All 4 different output shapes supported in OTBTF are tested.

"""
import tensorflow as tf

# Input
x = tf.keras.Input(shape=[None, None, None], name="x")  # [b, h, w, c=1]

# Create reshaped outputs
shape = tf.shape(x)
b = shape[0]
h = shape[1]
w = shape[2]
y1 = tf.reshape(x, shape=(b*h*w,))  # [b*h*w]
y2 = tf.reshape(x, shape=(b*h*w, 1))  # [b*h*w, 1]
y3 = tf.reshape(x, shape=(b, h, w))  # [b, h, w]
y4 = tf.reshape(x, shape=(b, h, w, 1))  # [b, h, w, 1]

# Create model
model = tf.keras.Model(inputs={"x": x}, outputs={"y1": y1, "y2": y2, "y3": y3, "y4": y4})
model.save("model5")

