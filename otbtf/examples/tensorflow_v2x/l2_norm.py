"""
This code implements a simple model that inputs one multispectral image ("x"),
and computes the euclidean norm of each pixel ("y").
The model is exported as a SavedModel named "l2_norm_savedmodel"

To run the SavedModel:

otbcli_TensorflowModelServe   \
-source1.il image1.tif        \
-model.dir l2_norm_savedmodel \
-model.fullyconv on           \
-out output.tif               \
-optim.disabletiling on  # Tiling is not helping here, since its a pixel wise op.
"""
import tensorflow as tf

# Input
x = tf.keras.Input(shape=[None, None, None], name="x")  # [1, h, w, N]

# Compute norm on the last axis
y = tf.norm(x, axis=-1)

# Create model
model = tf.keras.Model(inputs={"x": x}, outputs={"y": y})
model.save("l2_norm_savedmodel")
