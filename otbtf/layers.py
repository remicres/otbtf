# ==========================================================================
#
#   Copyright 2018-2019 IRSTEA
#   Copyright 2020-2025 INRAE
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
"""
[Source code :fontawesome-brands-github:](https://github.com/remicres/otbtf/
tree/master/otbtf/layers.py){ .md-button }

The utils module provides some useful keras layers to build deep nets.
"""

from typing import List, Tuple, Any
import tensorflow as tf
import keras


Tensor = Any
Scalars = List[float] | Tuple[float]


@keras.saving.register_keras_serializable()
class DilatedMask(keras.layers.Layer):
    """Layer to dilate a binary mask (Experimental)."""

    def __init__(self, *args, nodata_value: float, radius: int, **kwargs):
        """
        Params:
            nodata_value: the no-data value of the binary mask
            radius: dilatation radius

        """
        self.nodata_value = nodata_value
        self.radius = radius
        super().__init__(*args, **kwargs)

    def call(self, inp: Tensor):
        """
        Params:
            inp: input layer

        """
        # Compute a binary mask from the input
        nodata_mask = keras.ops.cast(tf.math.equal(inp, self.nodata_value), "uint8")

        se_size = 1 + 2 * self.radius
        # Create a morphological kernel suitable for binary dilatation, see
        # https://stackoverflow.com/q/54686895/13711499
        kernel = tf.zeros((se_size, se_size, 1), dtype=tf.uint8)
        conv2d_out = tf.nn.dilation2d(
            input=nodata_mask,
            filters=kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC",
            dilations=[1, 1, 1, 1],
            name="dilatation_conv2d",
        )
        return keras.ops.cast(conv2d_out, "uint8")

    def get_config(self):
        base_config = super().get_config()
        config = {"nodata_value": self.nodata_value, "radius": self.radius}
        return {**base_config, **config}


@keras.saving.register_keras_serializable()
class ApplyMask(keras.layers.Layer):
    """Layer to apply a binary mask to one input (Experimental)."""

    def __init__(self, *args, out_nodata: float, **kwargs):
        """
        Params:
            out_nodata: output no-data value, set when the mask is 1

        """
        super().__init__(*args, **kwargs)
        self.out_nodata = out_nodata

    def call(self, inputs: Tuple[Tensor, Tensor] | List[Tensor]):
        """
        Params:
            inputs: (mask, input). list or tuple of size 2. First element is
                the binary mask, second element is the input. In the binary
                mask, values at 1 indicate where to replace input values with
                no-data.

        """
        mask, inp = inputs
        return tf.where(mask == 1, float(self.out_nodata), inp)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "out_nodata": self.out_nodata,
        }
        return {**base_config, **config}


@keras.saving.register_keras_serializable()
class ScalarsTile(keras.layers.Layer):
    """
    Layer to duplicate some scalars in a whole array (Experimental).
    Simple example with only one scalar = 0.152:
        output [[0.152, 0.152, 0.152],
                [0.152, 0.152, 0.152],
                [0.152, 0.152, 0.152]]

    """

    def call(self, inputs: List[Tensor | Scalars] | Tuple[Tensor | Scalars, Tensor | Scalars]):
        """
        Params:
            inputs: [reference, scalar inputs]. Reference is the tensor whose
                shape has to be matched, is expected to be of shape [x, y, n].
                scalar inputs are expected to be of shape [1] or [n] so that
                they fill the last dimension of the output.

        """
        ref, scalar_inputs = inputs
        inp = tf.stack(scalar_inputs, axis=-1)
        inp = tf.expand_dims(tf.expand_dims(inp, axis=1), axis=1)
        return tf.tile(inp, [1, tf.shape(ref)[1], tf.shape(ref)[2], 1])


@keras.saving.register_keras_serializable()
class Argmax(keras.layers.Layer):
    """
    Layer to compute the argmax of a tensor.

    For example, for a vector A=[0.1, 0.3, 0.6], the output is 2 because
    A[2] is the max.
    Useful to transform a softmax into a "categorical" map for instance.

    """

    def __init__(self, *args, expand_last_dim: bool = True, **kwargs):
        """
        Params:
            expand_last_dim: expand the last dimension when True

        """
        super().__init__(*args, **kwargs)
        self.expand_last_dim = expand_last_dim

    def call(self, inputs):
        """
        Params:
            inputs: softmax tensor, or any tensor with last dimension of
                size nb_classes

        Returns:
            Index of the maximum value, in the last dimension. Int32.
            The output tensor has same shape length as input, but with last
            dimension of size 1. Contains integer values ranging from 0 to
            (nb_classes - 1).

        """
        argmax = tf.math.argmax(inputs, axis=-1)
        if self.expand_last_dim:
            return tf.expand_dims(argmax, axis=-1)
        return argmax

    def get_config(self):
        base_config = super().get_config()
        config = {
            "expand_last_dim": self.expand_last_dim,
        }
        return {**base_config, **config}


@keras.saving.register_keras_serializable()
class Max(keras.layers.Layer):
    """
    Layer to compute the max of a tensor.

    For example, for a vector [0.1, 0.3, 0.6], the output is 0.6
    Useful to transform a softmax into a "confidence" map for instance

    """

    def call(self, inputs):
        """
        Params:
            inputs: softmax tensor

        Returns:
            Maximum value along the last axis of the input.
            The output tensor has same shape length as input, but with last
            dimension of size 1.

        """
        return tf.expand_dims(tf.math.reduce_max(inputs, axis=-1), axis=-1)
