"""
This example shows how to use the otbtf python API to train a deep net from TFRecords.

We expect that the files are stored in the following way, with m, n, and k denoting respectively
the number of TFRecords files in the training, validation, and test datasets:

/dataset_dir
    /train
        1.records
        2.records
        ...
        m.records
    /valid
        1.records
        2.records
        ...
        n.records
    /test
        1.records
        2.records
        ...
        k.records

"""
import os
from otbtf import TFRecords
from otbtf.examples.tensorflow_v2x.fcnn import helper
from otbtf.examples.tensorflow_v2x.fcnn import fcnn_model

parser = helper.base_parser()
parser.add_argument("--tfrecords_dir", required=True,
                    help="Directory containing train, valid(, test) folders of TFRecords files")


def train(params):
    # Patches directories must contain 'train' and 'valid' dirs ('test' is not required)
    train_dir = os.path.join(params.tfrecords_dir, "train")
    valid_dir = os.path.join(params.tfrecords_dir, "valid")
    test_dir = os.path.join(params.tfrecords_dir, "test")

    kwargs = {"batch_size": params.batch_size,
              "target_keys": [fcnn_model.TARGET_NAME],
              "preprocessing_fn": fcnn_model.dataset_preprocessing_fn}

    # Training dataset. Must be shuffled
    assert os.path.isdir(train_dir)
    ds_train = TFRecords(train_dir).read(shuffle_buffer_size=1000, **kwargs)

    # Validation dataset
    assert os.path.isdir(valid_dir)
    ds_valid = TFRecords(valid_dir).read(**kwargs)

    # Test dataset (optional)
    ds_test = TFRecords(test_dir).read(**kwargs) if os.path.isdir(test_dir) else None

    # Train the model
    fcnn_model.train(params, ds_train, ds_valid, ds_test)


if __name__ == "__main__":
    train(parser.parse_args())
