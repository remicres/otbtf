"""
This test checks that the output tensor shapes are supported.
The input of this model must be a mono channel image.
All 4 different output shapes supported in OTBTF are tested.

"""
import keras

# Input
x = keras.Input(shape=[None, None, None], name="x")  # [b, h, w, c=1]

# Create reshaped outputs
shape = keras.ops.shape(x)
b = shape[0]
h = shape[1]
w = shape[2]
y1 = keras.ops.reshape(x, shape=(b*h*w,))  # [b*h*w]
y2 = keras.ops.reshape(x, shape=(b*h*w, 1))  # [b*h*w, 1]
y3 = keras.ops.reshape(x, shape=(b, h, w))  # [b, h, w]
y4 = keras.ops.reshape(x, shape=(b, h, w, 1))  # [b, h, w, 1]

# Create model
model = keras.Model(inputs={"x": x}, outputs={"y1": y1, "y2": y2, "y3": y3, "y4": y4})
model.export("model5")
