"""
[Source code :fontawesome-brands-github:](https://github.com/remicres/otbtf/
tree/master/otbtf/examples/tensorflow_v2x/deterministic){ .md-button }

This section contains two examples of very simple models that are not
trainable, called **deterministic models**.
Sometimes it can be useful to consider deterministic approaches (e.g. modeling)
and tensorflow is a powerful numerical library that can run smoothly on many
kind of devices such as GPUs.

In this section, we will consider two deterministic models:

- [L2 norm](#l2-norm): a model that computes the l2 norm of the input image
channels, for each pixel,
- [Scalar product](#scalar-product): a model computing the scalar product
between two images with the same number of channels, for each pixel

# L2 norm

We consider a very simple model that implements the computation of the l2 norm.
The model inputs one multispectral image (*x*), and computes the l2 norm of
each pixel (*y*). The model is exported as a SavedModel named
*l2_norm_savedmodel*

```python
import tensorflow as tf

# Input
x = tf.keras.Input(shape=[None, None, None], name="x")  # [1, h, w, N]

# Compute norm on the last axis
y = tf.norm(x, axis=-1)

# Create model
model = tf.keras.Model(inputs={"x": x}, outputs={"y": y})
model.save("l2_norm_savedmodel")
```

Run the code. The *l2_norm_savedmodel* file is created.
Now run the SavedModel with `TensorflowModelServe`:

```commandline
otbcli_TensorflowModelServe \\
-source1.il image1.tif \\
-model.dir l2_norm_savedmodel \\
-model.fullyconv on \\
-out output.tif \\
-optim.disabletiling on
```

!!! Note

    As you can notice, we have set the `optim.disabletiling` to `on` which
    disables the tiling for the processing. This means that OTB will drive the
    regions size based on the ram value defined in OTB. We can do that safely
    since our process has a small memory footprint, and it is not optimized
    with tiling because it does not use any neighborhood based approach.
    Tiling is enabled by default in `TensorflowModelServe` since it is mostly
    intended to perform inference using 2D convolutions.

# Scalar product

Let's consider a simple model that inputs two multispectral image (*x1* and
*x2*), and computes the scalar product between each pixels of the two images.
The model is exported as a SavedModel named *scalar_product_savedmodel*

```python
import tensorflow as tf

# Input
x1 = tf.keras.Input(shape=[None, None, None], name="x1")  # [1, h, w, N]
x2 = tf.keras.Input(shape=[None, None, None], name="x2")  # [1, h, w, N]

# Compute scalar product
y = tf.reduce_sum(tf.multiply(x1, x2), axis=-1)

# Create model
model = tf.keras.Model(inputs={"x1": x1, "x2": x2}, outputs={"y": y})
model.save("scalar_product_savedmodel")
```

Run the code. The *scalar_product_savedmodel* file is created.
Now run the SavedModel with `TensorflowModelServe`:

```commandline
OTB_TF_NSOURCES=2 otbcli_TensorflowModelServe \\
-source1.il image1.tif \\
-source2.il image2.tif \\
-model.dir scalar_product_savedmodel \\
-model.fullyconv on \\
-out output.tif \\
-optim.disabletiling on  # Small memory footprint, we can remove tiling
```

"""
