"""
This code implements a simple model that inputs two multispectral image ("x1" and "x2"),
and computes the scalar product between each pixels of the two images.
The model is exported as a SavedModel named "scalar_product_savedmodel"

To run the SavedModel:

OTB_TF_NSOURCES=2 otbcli_TensorflowModelServe \
-source1.il image1.tif                        \
-source2.il image2.tif                        \
-model.dir scalar_product_savedmodel          \
-model.fullyconv on                           \
-out output.tif                               \
-optim.disabletiling on  # Tiling is not helping here, since its a pixel wise op.
"""
import tensorflow as tf

# Input
x1 = tf.keras.Input(shape=[None, None, None], name="x1")  # [1, h, w, N]
x2 = tf.keras.Input(shape=[None, None, None], name="x2")  # [1, h, w, N]

# Compute scalar product
y = tf.reduce_sum(tf.multiply(x1, x2), axis=-1)

# Create model
model = tf.keras.Model(inputs={"x1": x1, "x2": x2}, outputs={"y": y})
model.save("scalar_product_savedmodel")
