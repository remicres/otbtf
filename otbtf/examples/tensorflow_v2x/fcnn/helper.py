from otbtf import TFRecords, dataset
import pathlib


def get_datasets(dataset_format, dataset_dir, batch_size, target_keys):
    """
    Function to use either TFRecords or patches images.
    Take a look in the comments below to see how files must be stored.

    :param dataset_format: dataset format. Either ("tfrecords" or "patches_images")
    :param dataset_dir": dataset root directory. Must contain 2 (3) subdirectories (train, valid(, test))
    """

    # Patches directories must contain 'train' and 'valid' dirs, 'test' is not required
    patches = pathlib.Path(dataset_dir)
    datasets = {}
    for d in patches.iterdir():
        if not d.is_dir():
            continue
        tag = d.name.lower()
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
            tfrecords = TFRecords(d)
            datasets[tag] = tfrecords.read(batch_size=batch_size, target_keys=target_keys,
                                           shuffle_buffer_size=1000) if tag == "train" else tfrecords.read(
                batch_size=batch_size, target_keys=target_keys)
        elif dataset_format == "patches_images":
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
            filenames_dict = {"input_xs": [],
                              "labels": []}
            for subd in d.iterdir():
                if not subd.is_dir():
                    continue
                for filename in subd.iterdir():
                    if filename.lower().endswith("_xs.tif"):
                        filenames_dict["input_xs"].append(filename)
                    if filename.lower().endswith("_labels.tif"):
                        filenames_dict["labels"].append(filename)

            # You can turn use_streaming=True to lower the memory footprint (patches are read on-the-fly on disk)
            datasets[tag] = dataset.DatasetFromPatchesImages(filenames_dict=filenames_dict)
        else:
            raise ValueError("dataset_format must be \"tfrecords\" or \"patches_images\"")

    return datasets
