from otbtf import TFRecords
import pathlib


def get_datasets(dataset_format, dataset_dir, batch_size, target_keys):
    """
    Function to use either TFRecords or patches images

    :param dataset_format: dataset format. Either ("tfrecords" or "patches_images")
    :param dataset_dir": dataset root directory. Must contain 2 (3) subdirectories (train, valid(, test))
    """
    assert dataset_format == "tfrecords" or dataset_format == "patches_images"

    # Patches directories must contain 'train' and 'valid' dirs, 'test' is not required
    patches = pathlib.Path(dataset_dir)
    datasets = {}
    for d in patches.iterdir():
        dir = d.name
        tag = dir.lower()
        assert tag in ["train", "valid", "test"], "Subfolders must be named train, valid (and test)"
        if dataset_format == "tfrecords":
            # When the dataset format is TFRecords, we expect that the files are stored in the following way, with
            # m, n, and k denoting respectively the number of TFRecords files in the training, validation, and test
            # datasets:
            #
            # /dataset_dir
            #     /train
            #         1.records
            #         2.records
            #         ...
            #         m.records
            #     /valid
            #         1.records
            #         2.records
            #         ...
            #         n.records
            #     /test
            #         1.records
            #         2.records
            #         ...
            #         k.records
            #
            tfrecords = TFRecords(dir)
            datasets[tag] = tfrecords.read(batch_size=batch_size, target_keys=target_keys,
                                           shuffle_buffer_size=1000) if tag == "train" else tfrecords.read(
                batch_size=batch_size, target_keys=target_keys)
        else:
            # When the dataset format is patches_images, we expect that the files are stored in the following way, with
            # M, N and K denoting respectively the number of patches-images in the training, validation, and test
            # datasets:
            #
            # /dataset_dir
            #     /train
            #         /image_1
            #             ..._xs.tif
            #             ..._labels.tif
            #         /image_2
            #             ..._xs.tif
            #             ..._labels.tif
            #         ...
            #         /image_M
            #             ..._xs.tif
            #             ..._labels.tif
            #     /valid
            #         /image_1
            #             ..._xs.tif
            #             ..._labels.tif
            #         ...
            #         /image_N
            #             ..._xs.tif
            #             ..._labels.tif
            #     /test
            #         /image_1
            #             ..._xs.tif
            #             ..._labels.tif
            #         ...
            #         /image_K
            #             ..._xs.tif
            #             ..._labels.tif
            for subd in d.iterdir():
                