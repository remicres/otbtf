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
import helper
import os
from otbtf import TFRecords
import fcnn_model

parser = helper.base_parser()
parser.add_argument("--tfrecords_dir", required=True,
                    help="Directory of subdirs containing TFRecords files: train, valid(, test)")

if __name__ == "__main__":
    params = parser.parse_args()

    # Patches directories must contain 'train' and 'valid' dirs ('test' is not required)
    train_dir = os.path.join(params.tfrecords_dir, "train")
    valid_dir = os.path.join(params.tfrecords_dir, "valid")
    test_dir = os.path.join(params.tfrecords_dir, "test")

    # Training dataset. Must be shuffled!
    assert os.path.isdir(train_dir)
    ds = TFRecords(train_dir)
    ds_train = ds.read(batch_size=params.batch_size, target_keys=["label"],
                                         shuffle_buffer_size=1000)

    # Validation dataset
    assert os.path.isdir(valid_dir)
    ds_valid = TFRecords(valid_dir).read(batch_size=params.batch_size, target_keys=["label"])

    # Test dataset (optional)
    ds_test = TFRecords(test_dir).read(batch_size=params.batch_size, target_keys=["label"]) if os.path.isdir(
        test_dir) else None

    # Train the model
    fcnn_model.train(params, ds_train, ds_valid, ds_test, ds.output_shapes)
