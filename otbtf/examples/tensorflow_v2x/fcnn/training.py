from otbtf.model import ModelBase
import tensorflow as tf
import tensorflow.keras.layers as layers
import argparse
import pathlib
from otbtf import TFRecords

# Application parameters
parser = argparse.ArgumentParser(description="Train a FCNN model")
parser.add_argument("-p", "--patches_dir", required=True, help="Directory of TFRecords dirs: train, valid(, test)")
parser.add_argument("-m", "--model_dir", required=True, help="Path to save model")
parser.add_argument("-e", "--number_epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("-b", "--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("-r", "--learning_rate", type=float, default=0.00001, help="Learning rate")


class FCNNModel(ModelBase):
    """
    A Simple Fully Convolutional model
    """

    def get_outputs(self, normalized_inputs):
        """
        This small model produces an output which has the same physical spacing as the input.
        The model generates [1 x 1 x N_CLASSES] output pixel for [32 x 32 x <nb channels>] input pixels.

        #Conv  |  depth      | kernel size | out. size (in x/y dims)
        ------ | ----------- | ----------- | ------------------------
        1      | N_NEURONS   | 5           | 28
        2      | 2*N_NEURONS | 4           | 25
        3      | 4*N_NEURONS | 4           | 22
        4      | 4*N_NEURONS | 4           | 19
        5      | 4*N_NEURONS | 4           | 16
        6      | 4*N_NEURONS | 4           | 13
        7      | 4*N_NEURONS | 4           | 10
        8      | 4*N_NEURONS | 4           | 7
        9      | 4*N_NEURONS | 4           | 4
        10     | 4*N_NEURONS | 4           | 1
        11     | N_CLASSES   | 1           | 1

        :return: activation values
        """

        # Model constants
        N_NEURONS = 16
        N_CLASSES = 6

        # Convolutions
        depths = [N_NEURONS, 2 * N_NEURONS] + 8 * [4 * N_NEURONS]
        ksizes = [5] + 9 * [4]
        net = normalized_inputs["input_xs"]
        for i, (d, k) in enumerate(zip(depths, ksizes)):
            conv = layers.Conv2D(filters=d, kernel_size=k, activation="relu", name=f"conv{i}")
            bn = tf.keras.layers.BatchNormalization()
            net = bn(conv(net))

        # Classifier
        lastconv = layers.Conv2D(filters=N_CLASSES, kernel_size=1, name="conv_class")
        return lastconv(net)  # out size: 1x1xN_CLASSES


def preprocessing_fn(inputs, targets):
    """
    Preprocessing function for the training dataset

    This function returns an output tuple (processed_inputs, processed_targets) ready to feed the model
    """
    return inputs, {"label": tf.one_hot(tf.squeeze(targets["label"], axis=-1), depth=2)}

if __name__ == "__main__":
    params = parser.parse_args()

    # Patches directories must contain 'train' and 'valid' dirs, 'test' is not required
    patches = pathlib.Path(params.patches_dir)
    ds_test = None
    for d in patches.iterdir():
        if "train" in d.name.lower() :
            ds_train = TFRecords(str(d)).read(shuffle_buffer_size=1000)
        elif "valid" in d.name.lower():
            ds_valid = TFRecords(str(d)).read()
        elif "test" in d.name.lower():
            ds_test = TFRecords(str(d)).read()

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

    #     ### Input PXS image
    #     in_pxs = keras.Input(shape=[None, None, 4], name="pxs")  # 4 is the number of channels in the pxs
    #
    #     ### Create network
    #     # Normalize input roughly in the [0, 1] range
    #     if div:
    #         in_pxs = layers.Lambda(lambda t: t / div)(in_pxs)
    #
    #     if model_type == "cnn":
    #         pseudo_proba = cnn_patch32_out6m(in_pxs)
    #     elif model_type == "fcnn":
    #         pseudo_proba = fcnn_fullres(in_pxs)
    #     elif model_type == "resnet":
    #         pseudo_proba = resnet_model1(in_pxs, sliced=True, patch_size=patch_size)
    #     else:
    #         raise NotImplementedError
    #
    #     ### Callbacks
    #     callbacks = []
    #     # TensorBoard
    #     if logs:
    #         callbacks.append(keras.callbacks.TensorBoard(log_dir=logs + f"/{expe_name}"))
    #     # Save best checkpoint
    #     if save_best:
    #         ckpt_name = model_dir + f"/best_{save_best}.ckpt"
    #         callbacks.append(keras.callbacks.ModelCheckpoint(ckpt_name, save_best_only=True, monitor=save_best))
    #     # Rate scheduler with exponential decay after n epochs
    #     if schedule:
    #         def scheduler(epoch, lr, after=2):
    #             return lr if epoch < after else lr * tf.math.exp(-0.1)
    #         callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))
    #     # Early stop if loss does not increase after n consecutive epochs
    #     if early_stop:
    #         callbacks.append(keras.callbacks.EarlyStopping(monitor="loss", mode="min", patience=5))
    #
    #     ### Metrics
    #     metrics = [
    #         tf.keras.metrics.BinaryAccuracy(),
    #         keras.metrics.Precision(),
    #         keras.metrics.Recall()
    #     ]
    #     # Metrics for multiclass, not sure about this workaround https://github.com/tensorflow/addons/issues/746
    #     # import tensorflow_addons as tfa
    #     # tfa.metrics.F1Score(num_classes=1, name="f1_score", average='micro', threshold=0.5)
    #     # tfa.metrics.CohenKappa(num_classes=2, name="cohenn_kappa")
    #
    #     ### Compile the model
    #     model = keras.Model(inputs={"pxs": in_pxs}, outputs={"label": pseudo_proba})
    #     model.compile(
    #         loss=keras.losses.BinaryCrossentropy(),
    #         optimizer=keras.optimizers.Adam(learning_rate=rate),
    #         metrics=metrics
    #     )
    #     # Print network
    #     model.summary(line_length=120)
    #     Path(model_dir).mkdir(exist_ok=True)
    #     keras.utils.plot_model(model, model_dir + "/model.png")
    #
    #     ### Train
    #     model.fit(
    #         ds_train,
    #         epochs=epochs,
    #         validation_data=ds_valid,
    #         callbacks=callbacks,
    #         verbose=1
    #     )
    #     ### TODO : Test ?
    #     #if ds_test is not None:
    #     #    model.evaluate(ds_test, batch_size=batch)
    #     # Save full model
    #     model.save(model_dir)
    #
    # return 0
