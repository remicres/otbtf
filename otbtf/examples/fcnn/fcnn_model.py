"""
Implementation of a small U-Net like model
"""

import logging

import tensorflow as tf
import keras

from otbtf.model import ModelBase

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Number of classes estimated by the model
N_CLASSES = 2

# Name of the input in the `FCNNModel` instance, also name of the input node
# in the SavedModel
INPUT_NAME = "input_xs"
INPUT_SIGNATURE = tf.TensorSpec(
    shape=[None, None, None, 4], dtype=tf.float32, name=INPUT_NAME
)

# Name of the output in the `FCNNModel` instance
TARGET_NAME = "predictions"


class FCNNModel(ModelBase):
    """
    A Simple Fully Convolutional U-Net like model
    """

    def normalize_inputs(self, inputs: dict):
        """
        Inherits from `ModelBase`

        The model will use this function internally to normalize its inputs,
        before applying `get_outputs()` that actually builds the operations
        graph (convolutions, etc). This function will hence work at training
        time and inference time.

        In this example, we assume that we have an input 12 bits multispectral
        image with values ranging from [0, 10000], that we process using a
        simple stretch to roughly match the [0, 1] range.

        Params:
            inputs: dict of inputs

        Returns:
            dict of normalized inputs, ready to be used from `get_outputs()`
        """
        return {INPUT_NAME: keras.ops.cast(inputs[INPUT_NAME], "float32") * 1e-4}

    def get_outputs(self, normalized_inputs: dict) -> dict:
        """
        Inherits from `ModelBase`

        This small model produces an output which has the same physical
        spacing as the input. The model generates [1 x 1 x N_CLASSES] output
        pixel for [32 x 32 x <nb channels>] input pixels.

        Params:
            normalized_inputs: dict of normalized inputs

        Returns:
            dict of model outputs
        """

        norm_inp = normalized_inputs[INPUT_NAME]

        def _conv(inp, depth, name):
            conv_op = keras.layers.Conv2D(
                filters=depth,
                kernel_size=3,
                strides=2,
                activation="relu",
                padding="same",
                name=name,
            )
            return conv_op(inp)

        def _tconv(inp, depth, name, activation="relu"):
            tconv_op = keras.layers.Conv2DTranspose(
                filters=depth,
                kernel_size=3,
                strides=2,
                activation=activation,
                padding="same",
                name=name,
            )
            return tconv_op(inp)

        out_conv1 = _conv(norm_inp, 16, "conv1")
        out_conv2 = _conv(out_conv1, 32, "conv2")
        out_conv3 = _conv(out_conv2, 64, "conv3")
        out_conv4 = _conv(out_conv3, 64, "conv4")
        out_tconv1 = _tconv(out_conv4, 64, "tconv1") + out_conv3
        out_tconv2 = _tconv(out_tconv1, 32, "tconv2") + out_conv2
        out_tconv3 = _tconv(out_tconv2, 16, "tconv3") + out_conv1
        out_tconv4 = _tconv(out_tconv3, N_CLASSES, "classifier", None)

        softmax_op = keras.layers.Softmax()
        predictions = softmax_op(out_tconv4)

        # Model outputs are returned in a `dict`, where each key is an output
        # name, and the value is the layer output. This naming have two
        # functions:
        #  - the output layer name is used at inference time, to identify
        #    the output tensor from which generate the output image,
        #  - the output layer name identifies the output tensors, to fit the
        #    targets to model outputs, compute metrics, etc. during training
        #    process. It can also be used to access the tensors as tf/keras
        #    objects, for instance to display previews images in TensorBoard.
        #
        # Note that we could also add additional outputs, even outputs which
        # are useless for the optimization process, for instance the argmax :
        #   ```
        #   argmax_op = otbtf.layers.Argmax()
        #   labels = argmax_op(predictions)
        #   return {TARGET_NAME: predictions, OUTPUT_ARGMAX_NAME: labels}
        #   ```
        # The default extra outputs (i.e. output tensors with cropping in
        # physical domain) are append by `otbtf.ModelBase` for all returned
        # outputs of this function to be used at inference time (e.g.
        # "labels_crop32", "labels_crop64", ..., "predictions__crop16", ...,
        # etc).
        return {TARGET_NAME: predictions}


def dataset_preprocessing_fn(examples: dict):
    """
    Preprocessing function for the training dataset.
    This function is only used at training time, to put the data in the
    expected format for the training step.
    DO NOT USE THIS FUNCTION TO NORMALIZE THE INPUTS ! (see
    `otbtf.ModelBase.normalize_inputs` for that).
    Note that this function is not called here, but in the code that prepares
    the datasets.

    Params:
        examples: dict for examples (i.e. inputs and targets stored in a single
            dict)

    Returns:
        preprocessed examples

    """
    return {
        INPUT_NAME: examples["input_xs_patches"],
        TARGET_NAME: keras.ops.one_hot(
            keras.ops.squeeze(
                keras.ops.cast(examples["labels_patches"], tf.int32), axis=-1
            ),
            N_CLASSES,
        ),
    }


def train(params, ds_train, ds_valid, ds_test):
    """
    Create, train, and save the model.

    Params:
        params: contains batch_size, learning_rate, nb_epochs, and model_dir
        ds_train: training dataset
        ds_valid: validation dataset
        ds_test: testing dataset

    """

    strategy = tf.distribute.MirroredStrategy()  # For single or multi-GPUs
    with strategy.scope():
        # Model instantiation. Note that the normalize_fn is now part of the
        # model. It is mandatory to instantiate the model inside the strategy
        # scope.
        model = FCNNModel(dataset_element_spec=ds_train.element_spec)

        # Compile the model
        # Since Keras 3 it is mandatory to use a `dict` to explicitly name the
        # outputs over which the losses/metrics are computed, e.g.
        # `loss: {TARGET_NAME: "categorical_crossentropy"}`
        model.compile(
            loss={TARGET_NAME: keras.losses.CategoricalCrossentropy()},
            optimizer=keras.optimizers.Adam(learning_rate=params.learning_rate),
            metrics={
                TARGET_NAME: [
                    keras.metrics.Precision(class_id=1),
                    keras.metrics.Recall(class_id=1),
                ]
            },
        )

        # Summarize the model (in CLI)
        model.summary()

        # Train
        model.fit(ds_train, epochs=params.nb_epochs, validation_data=ds_valid)

        # Evaluate against test data
        if ds_test is not None:
            model.evaluate(ds_test, batch_size=params.batch_size)

        # Save trained model as SavedModel
        model.export(params.model_dir)
