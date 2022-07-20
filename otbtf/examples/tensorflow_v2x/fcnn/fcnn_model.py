"""
Implementation of a small U-Net like model
"""
from otbtf.model import ModelBase
import tensorflow as tf
import tensorflow.keras.layers as layers

N_CLASSES = 6


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

        # final layers
        net = tf.keras.activations.softmax(net)
        net = tf.keras.layers.Cropping2D(cropping=32)(net)

        return {"predictions": net}


def preprocessing_fn(examples):
    """
    Preprocessing function for the training dataset.
    This function is only used at training time, to put the data in the expected format.
    DO NOT USE THIS FUNCTION TO NORMALIZE THE INPUTS ! (see `otbtf.ModelBase.normalize_fn` for that).
    Note that this function is not called here, but in the code that prepares the datasets.

    :param examples: dict for examples (i.e. inputs and targets stored in a single dict)
    :return: preprocessed examples
    """
    def _to_categorical(x):
        return tf.one_hot(tf.squeeze(x, axis=-1), depth=N_CLASSES)
    return {"input_xs": examples["input_xs"],
            "predictions": _to_categorical(examples["labels"])}


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


def train(params, ds_train, ds_valid, ds_test, output_shapes):
    """
    Create, train, and save the model.

    :param params: contains batch_size, learning_rate, nb_epochs, and model_dir
    """

    # Model
    model = FCNNModel(dataset_input_keys=["input_xs"],
                      model_output_keys=["labels"],
                      dataset_shapes=output_shapes,
                      normalize_fn=normalize_fn)  # Note that the normalize_fn is now part of the model

    # strategy = tf.distribute.MirroredStrategy()  # For single or multi-GPUs
    strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    with strategy.scope():
        # Create and compile the model
        model.create_network()
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate),
                      metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

        # Summarize the model (in CLI)
        model.summary()

        # Train
        model.fit(ds_train, epochs=params.nb_epochs, validation_data=ds_valid)

        # Evaluate against test data
        if ds_test is not None:
            model.evaluate(ds_test, batch_size=params.batch_size)

        # Save trained model as SavedModel
        model.save(params.model_dir)
