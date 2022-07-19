from otbtf.model import ModelBase
import tensorflow as tf
import tensorflow.keras.layers as layers
import argparse
import pathlib
import os
import helper

# Application parameters
parser = argparse.ArgumentParser(description="Train a FCNN model")
parser.add_argument("-p", "--dataset_dir", required=True, help="Directory of subdirs: train, valid(, test)")
parser.add_argument("-f", "--dataset_format", default="tfrecords", const="tfrecords", nargs="?",
                    choices=["tfrecords", "patches_images"], help="Format of the dataset (TFRecords or Patches images")
parser.add_argument("-b", "--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("-r", "--learning_rate", type=float, default=0.00001, help="Learning rate")
parser.add_argument("-e", "--number_epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("-m", "--model_dir", required=True, help="Path to save model")


class FCNNModel(ModelBase):
    """
    A Simple Fully Convolutional U-Net like model
    """

    def get_outputs(self, normalized_inputs):
        """
        This small model produces an output which has the same physical spacing as the input.
        The model generates [1 x 1 x N_CLASSES] output pixel for [32 x 32 x <nb channels>] input pixels.

        :param normalized_inputs: dict of normalized inputs`
        :return: activation values
        """

        # Model constants
        N_CLASSES = 6

        # Model input
        net = normalized_inputs["input_xs"]

        # Encoder
        convs_depth = {"conv1": 16, "conv2": 32, "conv3": 64, "conv4": 64}
        for name, depth in convs_depth.items():
            conv = layers.Conv2D(filters=depth, kernel_size=3, activation="relu", name=name)
            net = conv(net)

        # Decoder
        tconvs_depths = {"tconv1": 64, "tconv2": 32, "tconv3": 16, "tconv4": N_CLASSES}
        for name, depth in tconvs_depths.items():
            tconv = layers.Conv2DTranspose(filters=depth, kernel_size=3, activation="relu", name=name)
            net = tconv(net)

        return net


def preprocessing_fn(inputs, targets):
    """
    Preprocessing function for the training dataset.
    This function is only used at training time, to put the data in the expected format.
    DO NOT USE THIS FUNCTION TO NORMALIZE THE INPUTS ! (see `otbtf.ModelBase.normalize_fn` for that)

    :param inputs: dict for inputs
    :param targets: dict for targets
    :return: an output tuple (processed_inputs, processed_targets)
    """
    return inputs, {"label": tf.one_hot(tf.squeeze(targets["label"], axis=-1), depth=2)}


def normalize_fn(inputs):
    """
    The model will use this function internally to normalize its input, before applying the `get_outputs()` function
    that actually builds the operations graph (convolutions, etc).
    This function will hence work at training time and inference time.

    In this example, we assume that we have an input 12 bits multispectral image with values ranging from [0, 10 000],
    that we process using a simple stretch to roughly match the [0, 1] range.

    :param inputs: dict of inputs
    :return: dict of normalized inputs, ready to be used from the `get_outputs()` function of the model
    """
    return {"input_xs": inputs["input_xs"] * 0.0001}


if __name__ == "__main__":
    params = parser.parse_args()

    # Get datasets
    ds_train, ds_valid, ds_test = helper.get_datasets(params.dataset_format, params.dataset_dir)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        # Create and compile the model
        model = FCNNModel(normalize_fn=normalize_fn)
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate),
                      metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        # Summarize the model (in CLI)
        model.summary(line_length=120)

        # Summarize the model (in figure.png)
        pathlib.Path(params.model_dir).mkdir(exist_ok=True)
        tf.keras.utils.plot_model(model, os.path.join(params.model_dir, "figure.png"))

        # Train
        model.fit(ds_train, epochs=params.number_epochs, validation_data=ds_valid)

        # Evaluate against test data
        if ds_test is not None:
            model.evaluate(ds_test, batch_size=params.batch_size)

        # Save trained model
        model.save(params.model_dir)
