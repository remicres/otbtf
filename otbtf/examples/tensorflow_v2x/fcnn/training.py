from otbtf.model import ModelBase
import tensorflow as tf
import tensorflow.keras.layers as layers
import argparse
import pathlib
from otbtf import TFRecords
import os

# Application parameters
parser = argparse.ArgumentParser(description="Train a FCNN model")
parser.add_argument("-p", "--patches_dir", required=True, help="Directory of TFRecords dirs: train, valid(, test)")
parser.add_argument("-m", "--model_dir", required=True, help="Path to save model")
parser.add_argument("-e", "--number_epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("-b", "--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("-r", "--learning_rate", type=float, default=0.00001, help="Learning rate")
parser.add_argument('--dataset_mode', default='tfrecords', const='tfrecords', nargs='?',
                    choices=['tfrecords', 'patches_images'])

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
    Preprocessing function for the training dataset

    :param inputs: dict for inputs
    :param targets: dict for targets
    :return: an output tuple (processed_inputs, processed_targets) ready to feed the model
    """
    return inputs, {"label": tf.one_hot(tf.squeeze(targets["label"], axis=-1), depth=2)}


if __name__ == "__main__":
    params = parser.parse_args()

    # Patches directories must contain 'train' and 'valid' dirs, 'test' is not required
    patches = pathlib.Path(params.patches_dir)
    ds_test = None
    for d in patches.iterdir():
        if "train" in d.name.lower():
            ds_train = TFRecords(str(d)).read(shuffle_buffer_size=1000)
        elif "valid" in d.name.lower():
            ds_valid = TFRecords(str(d)).read()
        elif "test" in d.name.lower():
            ds_test = TFRecords(str(d)).read()

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        # Create and compile the model
        model = FCNNModel()
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
