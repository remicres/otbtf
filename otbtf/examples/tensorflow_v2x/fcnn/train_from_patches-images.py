"""
This example shows how to use the otbtf python API to train a deep net from patches-images.

We expect that the files are stored in the following way, with M, N and K denoting respectively
the number of patches-images in the training, validation, and test datasets:

/dataset_dir
    /train
        /image_1
            ..._xs.tif
            ..._labels.tif
        /image_2
            ..._xs.tif
            ..._labels.tif
        ...
        /image_M
            ..._xs.tif
            ..._labels.tif
    /valid
        /image_1
            ..._xs.tif
            ..._labels.tif
        ...
        /image_N
            ..._xs.tif
            ..._labels.tif
    /test
        /image_1
            ..._xs.tif
            ..._labels.tif
        ...
        /image_K
            ..._xs.tif
            ..._labels.tif

"""
import helper
from otbtf import DatasetFromPatchesImages
import fcnn_model

parser = helper.base_parser()
parser.add_argument("--train_xs", required=True, nargs="+", default=[],
                    help="A list of patches-images for the XS image (training dataset)")
parser.add_argument("--train_labels", required=True, nargs="+", default=[],
                    help="A list of patches-images for the labels (training dataset)")
parser.add_argument("--valid_xs", required=True, nargs="+", default=[],
                    help="A list of patches-images for the XS image (validation dataset)")
parser.add_argument("--valid_labels", required=True, nargs="+", default=[],
                    help="A list of patches-images for the labels (validation dataset)")
parser.add_argument("--test_xs", required=False, nargs="+", default=[],
                    help="A list of patches-images for the XS image (test dataset)")
parser.add_argument("--test_labels", required=False, nargs="+", default=[],
                    help="A list of patches-images for the labels (test dataset)")


def create_dataset(xs_filenames, labels_filenames, batch_size):
    """
    Create an otbtf.DatasetFromPatchesImages
    """
    # Sort patches and labels
    xs_filenames.sort()
    labels_filenames.sort()

    # Check patches and labels are correctly sorted
    helper.check_files_order(xs_filenames, labels_filenames)

    # Create dataset from the filename dict
    # You can add the use_streaming option here, is you want to lower the memory budget.
    # However, this can slow down your process since the patches are read on-the-fly on the filesystem.
    # Good when one batch computation is slower than one batch gathering.
    ds = DatasetFromPatchesImages(filenames_dict={"input_xs": xs_filenames, "labels": labels_filenames})
    return ds, ds.get_tf_dataset(batch_size=batch_size)


if __name__ == "__main__":
    params = parser.parse_args()

    _, ds_train = create_dataset(params.train_xs, params.train_labels, params.batchsize)
    _, ds_valid = create_dataset(params.valid_xs, params.valid_labels, params.batchsize)
    _, ds_test = create_dataset(params.valid_xs, params.valid_labels, params.batchsize) \
        if params.test_xs and params.test_labels else None

    # Train the model
    fcnn_model.train(params, ds_train, ds_valid, ds_test)
