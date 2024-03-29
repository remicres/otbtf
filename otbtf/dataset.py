# -*- coding: utf-8 -*-
# ==========================================================================
#
#   Copyright 2018-2019 IRSTEA
#   Copyright 2020-2023 INRAE
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0.txt
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ==========================================================================*/
"""
[Source code :fontawesome-brands-github:](https://github.com/remicres/otbtf/
tree/master/otbtf/dataset.py){ .md-button }

Contains stuff to help working with TensorFlow and geospatial data in the
OTBTF framework.
"""
import logging
import multiprocessing
import threading
import time
from abc import ABC, abstractmethod

from typing import Any, List, Dict, Type, Callable
import numpy as np
import tensorflow as tf

import otbtf.tfrecords
import otbtf.utils


class Buffer:
    """
    Used to store and access list of objects
    """

    def __init__(self, max_length: int):
        """
        Params:
            max_length: length of the buffer
        """
        self.max_length = max_length
        self.container = []

    def size(self) -> int:
        """
        Returns:
             the buffer size
        """
        return len(self.container)

    def add(self, new_element: Any):
        """
        Add an element in the buffer.

        Params:
            new_element: new element to add

        """
        self.container.append(new_element)
        assert self.size() <= self.max_length

    def is_complete(self) -> bool:
        """
        Returns:
             True if the buffer is full, False else
        """
        return self.size() == self.max_length


class PatchesReaderBase(ABC):
    """
    Base class for patches delivery
    """

    @abstractmethod
    def get_sample(self, index: int) -> Any:
        """
        Return one sample.

        Params:
            index: sample index

        Returns:
            One sample instance, whatever the sample structure is (dict, numpy
                array, ...)
        """

    @abstractmethod
    def get_stats(self) -> dict:
        """
        Compute some statistics for each source.
        Depending on if streaming is used, the statistics are computed
        directly in memory, or chunk-by-chunk.

        Returns:
            a dict having the following structure:
                {
                "src_key_0":
                    {"min": np.array([...]),
                    "max": np.array([...]),
                    "mean": np.array([...]),
                    "std": np.array([...])},
                ...,
                "src_key_M":
                    {"min": np.array([...]),
                    "max": np.array([...]),
                    "mean": np.array([...]),
                    "std": np.array([...])},
                }

        """

    @abstractmethod
    def get_size(self) -> int:
        """
        Returns the total number of samples

        Returns:
            number of samples (int)

        """


class PatchesImagesReader(PatchesReaderBase):
    """
    This class provides a read access to a set of patches images.

    A patches image is an image of patches stacked in rows, as produced from
    the OTBTF "PatchesExtraction" application, and is stored in a raster
    format (e.g. GeoTiff).
    A source can be a particular domain in which the patches are extracted
    (remember that in OTBTF applications, the number of sources is controlled
    by the `OTB_TF_NSOURCES` environment variable).

    This class enables to use:
     - multiple sources
     - multiple patches images per source

    Each patch can be independently accessed using the get_sample(index)
    function, with index in [0, self.size), self.size being the total number
    of patches (must be the same for each sources).

    See `PatchesReaderBase`.

    """

    def __init__(
            self,
            filenames_dict: Dict[str, List[str]],
            use_streaming: bool = False,
            scalar_dict: Dict[str, List[Any]] = None
    ):
        """
        Params:
            filenames_dict: A dict structured as follow:
                {
                    src_name1: [
                        src1_patches_image_1.tif, ..., src1_patches_image_N.tif
                    ],
                    src_name2: [
                        src2_patches_image_1.tif, ..., src2_patches_image_N.tif
                    ],
                    ...
                    src_nameM: [
                        srcM_patches_image_1.tif, ..., srcM_patches_image_N.tif
                    ]
                }
            use_streaming: if True, the patches are read on the fly from the
                disc, nothing is kept in memory. Else, everything is packed in
                memory.
            scalar_dict: (optional) a dict containing list of scalars (int,
            float, str) as follow:
                {
                    scalar_name1: ["value_1", ..., "value_N"],
                    scalar_name2: [value_1, ..., value_N],
                    ...
                    scalar_nameM: [value1, ..., valueN]
                }

        """

        assert len(filenames_dict.values()) > 0

        # gdal_ds dict
        self.gdal_ds = {
            key: [otbtf.utils.gdal_open(src_fn) for src_fn in src_fns]
            for key, src_fns in filenames_dict.items()
        }

        # streaming on/off
        self.use_streaming = use_streaming

        # Scalar dict (e.g. for metadata)
        # If the scalars are not numpy.ndarray, convert them
        self.scalar_dict = {
            key: [
                i if isinstance(i, np.ndarray) else np.asarray(i)
                for i in scalars
            ]
            for key, scalars in scalar_dict.items()
        } if scalar_dict else {}

        # check number of patches in each sources
        if len({
            len(ds_list)
            for ds_list in
            list(self.gdal_ds.values()) + list(self.scalar_dict.values())
        }) != 1:
            raise Exception(
                "Each source must have the same number of patches images"
            )

        # gdal_ds check
        nb_of_patches = {key: 0 for key in self.gdal_ds}
        self.nb_of_channels = {}
        for src_key, ds_list in self.gdal_ds.items():
            for gdal_ds in ds_list:
                nb_of_patches[src_key] += self._get_nb_of_patches(gdal_ds)
                if src_key not in self.nb_of_channels:
                    self.nb_of_channels[src_key] = gdal_ds.RasterCount
                else:
                    if self.nb_of_channels[src_key] != gdal_ds.RasterCount:
                        raise Exception(
                            "All patches images from one source must have the "
                            "same number of channels! "
                            f"Error happened for source: {src_key}"
                        )
        if len(set(nb_of_patches.values())) != 1:
            raise Exception(
                "Sources must have the same number of patches! "
                f"Number of patches: {nb_of_patches}"
            )

        # gdal_ds sizes
        src_key_0 = list(self.gdal_ds)[0]  # first key
        self.ds_sizes = [
            self._get_nb_of_patches(ds)
            for ds in self.gdal_ds[src_key_0]
        ]
        self.size = sum(self.ds_sizes)

        # if use_streaming is False, we store in memory all patches images
        if not self.use_streaming:
            self.patches_buffer = {
                src_key: np.concatenate([
                    otbtf.utils.read_as_np_arr(ds) for ds in src_ds
                ], axis=0)
                for src_key, src_ds in self.gdal_ds.items()
            }

    def _get_ds_and_offset_from_index(self, index):
        offset = index
        idx = None
        for idx, ds_size in enumerate(self.ds_sizes):
            if offset < ds_size:
                break
            offset -= ds_size

        return idx, offset

    @staticmethod
    def _get_nb_of_patches(gdal_ds):
        return int(gdal_ds.RasterYSize / gdal_ds.RasterXSize)

    @staticmethod
    def _read_extract_as_np_arr(gdal_ds, offset):
        assert gdal_ds is not None
        psz = gdal_ds.RasterXSize
        yoff = int(offset * psz)
        assert yoff + psz <= gdal_ds.RasterYSize
        buffer = gdal_ds.ReadAsArray(0, yoff, psz, psz)
        if len(buffer.shape) == 3:
            # multi-band raster
            return np.transpose(buffer, axes=(1, 2, 0))
        return np.expand_dims(buffer, axis=2)

    def get_sample(self, index: int) -> Dict[str, np.array]:
        """
        Return one sample of the dataset.

        Params:
            index: the sample index. Must be in the [0, self.size) range.

        Returns:
            The sample is stored in a dict with the following structure:
                {
                    "src_key_0": np.array((psz_y_0, psz_x_0, nb_ch_0)),
                    "src_key_1": np.array((psz_y_1, psz_x_1, nb_ch_1)),
                    ...
                    "src_key_M": np.array((psz_y_M, psz_x_M, nb_ch_M))
                }

        """
        assert index >= 0
        assert index < self.size

        i, offset = self._get_ds_and_offset_from_index(index)
        res = {
            src_key: scalar[i]
            for src_key, scalar in self.scalar_dict.items()
        }
        if not self.use_streaming:
            res.update({
                src_key: arr[index, :, :, :]
                for src_key, arr in self.patches_buffer.items()
            })
        else:
            res.update({
                src_key: self._read_extract_as_np_arr(
                    self.gdal_ds[src_key][i], offset
                )
                for src_key in self.gdal_ds
            })
        return res

    def get_stats(self) -> Dict[str, List[float]]:
        """
        Compute some statistics for each source.
        When streaming is used, chunk-by-chunk. Else, the statistics are
        computed directly in memory.

        Returns:
             statistics dict
        """
        logging.info("Computing stats")
        if not self.use_streaming:
            axis = (0, 1, 2)  # (row, col)
            stats = {
                src_key: {
                    "min": np.amin(patches_buffer, axis=axis),
                    "max": np.amax(patches_buffer, axis=axis),
                    "mean": np.mean(patches_buffer, axis=axis),
                    "std": np.std(patches_buffer, axis=axis)
                }
                for src_key, patches_buffer in self.patches_buffer.items()
            }
        else:
            axis = (0, 1)  # (row, col)

            def _filled(value):
                return {
                    src_key: value * np.ones((self.nb_of_channels[src_key]))
                    for src_key in self.gdal_ds}

            _maxs = _filled(0.0)
            _mins = _filled(float("inf"))
            _sums = _filled(0.0)
            _sqsums = _filled(0.0)
            for index in range(self.size):
                sample = self.get_sample(index=index)
                for src_key, np_arr in sample.items():
                    rnumel = 1.0 / float(np_arr.shape[0] * np_arr.shape[1])
                    _mins[src_key] = np.minimum(
                        np.amin(np_arr, axis=axis).flatten(), _mins[src_key]
                    )
                    _maxs[src_key] = np.maximum(
                        np.amax(np_arr, axis=axis).flatten(), _maxs[src_key]
                    )
                    _sums[src_key] += rnumel * np.sum(
                        np_arr, axis=axis
                    ).flatten()
                    _sqsums[src_key] += rnumel * np.sum(
                        np.square(np_arr), axis=axis
                    ).flatten()

            rsize = 1.0 / float(self.size)
            stats = {
                src_key: {
                    "min": _mins[src_key],
                    "max": _maxs[src_key],
                    "mean": rsize * _sums[src_key],
                    "std": np.sqrt(
                        rsize * _sqsums[src_key] - np.square(
                            rsize * _sums[src_key]
                        )
                    )
                }
                for src_key in self.gdal_ds
            }
        logging.info("Stats: %s", stats)
        return stats

    def get_size(self) -> int:
        """
        Returns:
            size
        """
        return self.size


class IteratorBase(ABC):
    """
    Base class for iterators
    """

    @abstractmethod
    def __init__(self, patches_reader: PatchesReaderBase):
        pass


class RandomIterator(IteratorBase):
    """
    Pick a random number in the [0, handler.size) range.
    """

    def __init__(self, patches_reader: PatchesReaderBase):
        """
        Params:
            patches_reader: patches reader
        """
        super().__init__(patches_reader=patches_reader)
        self.indices = np.arange(0, patches_reader.get_size())
        self._shuffle()
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        current_index = self.indices[self.count]
        if self.count < len(self.indices) - 1:
            self.count += 1
        else:
            self._shuffle()
            self.count = 0
        return current_index

    def _shuffle(self):
        np.random.shuffle(self.indices)


class Dataset:
    """
    Handles the "mining" of patches.
    This class has a thread that extract tuples from the readers, while
    ensuring the access of already gathered tuples.

    See `PatchesReaderBase` and `Buffer`

    """

    def __init__(
            self,
            patches_reader: PatchesReaderBase = None,
            buffer_length: int = 128,
            iterator_cls: Type[IteratorBase] = RandomIterator,
            max_nb_of_samples: int = None
    ):
        """
        Params:
            patches_reader: The patches reader instance
            buffer_length: The number of samples that are stored in the
                buffer
            iterator_cls: The iterator class used to generate the sequence of
                patches indices.
            max_nb_of_samples: Optional, max number of samples to consider

        """
        # patches reader
        self.patches_reader = patches_reader

        # If necessary, limit the nb of samples
        logging.info('Number of samples: %s', self.patches_reader.get_size())
        if max_nb_of_samples and \
                self.patches_reader.get_size() > max_nb_of_samples:
            logging.info('Reducing number of samples to %s', max_nb_of_samples)
            self.size = max_nb_of_samples
        else:
            self.size = self.patches_reader.get_size()

        # iterator
        self.iterator = iterator_cls(patches_reader=self.patches_reader)

        # Get patches sizes and type, of the first sample of the first tile
        self.output_types = {}
        self.output_shapes = {}
        one_sample = self.patches_reader.get_sample(index=0)
        for src_key, np_arr in one_sample.items():
            self.output_shapes[src_key] = np_arr.shape
            self.output_types[src_key] = tf.dtypes.as_dtype(np_arr.dtype)

        logging.info("output_types: %s", self.output_types)
        logging.info("output_shapes: %s", self.output_shapes)

        # buffers
        if self.size <= buffer_length:
            buffer_length = self.size
        self.miner_buffer = Buffer(buffer_length)
        self.mining_lock = multiprocessing.Lock()
        self.consumer_buffer = Buffer(buffer_length)
        self.consumer_buffer_pos = 0
        self.tot_wait = 0
        self.miner_thread = self._summon_miner_thread()
        self.read_lock = multiprocessing.Lock()
        self._dump()

        # Prepare tf dataset for one epoch
        self.tf_dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_types=self.output_types,
            output_shapes=self.output_shapes
        ).repeat(1)

    def to_tfrecords(
            self,
            output_dir: str,
            n_samples_per_shard: int = 100,
            drop_remainder: bool = True
    ):
        """
        Save the dataset into TFRecord files

        Params:
            output_dir: output directory
            n_samples_per_shard: number of samples per TFRecord file
            drop_remainder: drop remaining samples

        """
        tfrecord = otbtf.tfrecords.TFRecords(output_dir)
        tfrecord.ds2tfrecord(
            self,
            n_samples_per_shard=n_samples_per_shard,
            drop_remainder=drop_remainder
        )

    def get_stats(self) -> Dict[str, List[float]]:
        """
        Compute dataset statistics

        Return:
            the dataset statistics, computed by the patches reader

        """
        with self.mining_lock:
            return self.patches_reader.get_stats()

    def read_one_sample(self) -> Dict[str, Any]:
        """
        Read one element of the consumer_buffer
        The lock is used to prevent different threads to read and update the
        internal counter concurrently

        Return:
            one sample

        """
        with self.read_lock:
            output = None
            if self.consumer_buffer_pos < self.consumer_buffer.max_length:
                output = self.consumer_buffer.container[
                    self.consumer_buffer_pos]
                self.consumer_buffer_pos += 1
            if self.consumer_buffer_pos == self.consumer_buffer.max_length:
                self._dump()
                self.consumer_buffer_pos = 0
            return output

    def _dump(self):
        """
        This function dumps the miner_buffer into the consumer_buffer, and
        restarts the miner_thread

        """
        # Wait for miner to finish his job
        date_t = time.time()
        self.miner_thread.join()
        self.tot_wait += time.time() - date_t

        # Copy miner_buffer.container --> consumer_buffer.container
        self.consumer_buffer.container = self.miner_buffer.container.copy()

        # Clear miner_buffer.container
        self.miner_buffer.container.clear()

        # Restart miner_thread
        self.miner_thread = self._summon_miner_thread()

    def _collect(self):
        """
        This function collects samples.
        It is threaded by the miner_thread.

        """
        # Fill the miner_container until it's full
        while not self.miner_buffer.is_complete():
            index = next(self.iterator)
            with self.mining_lock:
                new_sample = self.patches_reader.get_sample(index=index)
                self.miner_buffer.add(new_sample)

    def _summon_miner_thread(self) -> threading.Thread:
        """
        Create and starts the thread for the data collect
        """
        new_thread = threading.Thread(target=self._collect)
        new_thread.start()
        return new_thread

    def _generator(self):
        """
        Generator function, used for the tf dataset
        """
        for _ in range(self.size):
            yield self.read_one_sample()

    def get_tf_dataset(
            self,
            batch_size: int,
            drop_remainder: bool = True,
            preprocessing_fn: Callable = None,
            targets_keys: List[str] = None
    ) -> tf.data.Dataset:
        """
        Returns a TF dataset, ready to be used with the provided batch size

        Params:
            batch_size: the batch size
            drop_remainder: drop incomplete batches
            preprocessing_fn: An optional preprocessing function that takes
                input examples as args and returns the preprocessed input
                examples. Typically, examples are composed of model inputs and
                targets. Model inputs and model targets must be computed
                accordingly to (1) what the model outputs and (2) what
                training loss needs. For instance, for a classification
                problem, the model will likely output the softmax, or
                activation neurons, for each class, and the cross entropy loss
                requires labels in one hot encoding. In this case, the
                preprocessing_fn has to transform the labels values (integer
                ranging from [0, n_classes]) in one hot encoding (vector of 0
                and 1 of length n_classes). The preprocessing_fn should not
                implement such things as radiometric transformations from
                input to input_preprocessed, because those are performed
                inside the model itself (see
                `otbtf.ModelBase.normalize_inputs()`).
            targets_keys: Optional. When provided, the dataset returns a tuple
                of dicts (inputs_dict, target_dict) so it can be
                straightforwardly used with keras models objects.

        Returns:
             The TF dataset

        """
        if 2 * batch_size >= self.miner_buffer.max_length:
            logging.warning(
                "Batch size is %s but dataset buffer has %s elements. "
                "Consider using a larger dataset buffer to avoid I/O "
                "bottleneck", batch_size, self.miner_buffer.max_length
            )
        tf_ds = self.tf_dataset.map(preprocessing_fn) \
            if preprocessing_fn else self.tf_dataset

        if targets_keys:
            def _split_input_and_target(example):
                # Differentiating inputs and outputs for keras
                inputs = {
                    key: value for (key, value) in example.items()
                    if key not in targets_keys
                }
                targets = {
                    key: value for (key, value) in example.items()
                    if key in targets_keys
                }
                return inputs, targets

            tf_ds = tf_ds.map(_split_input_and_target)

        return tf_ds.batch(batch_size, drop_remainder=drop_remainder)

    def get_total_wait_in_seconds(self) -> int:
        """
        Returns the number of seconds during which the data gathering was
        delayed because I/O bottleneck

        Returns:
            duration in seconds
        """
        return self.tot_wait


class DatasetFromPatchesImages(Dataset):
    """
    Handles the "mining" of a set of patches images.

    :see PatchesImagesReader
    :see Dataset
    """

    def __init__(
            self,
            filenames_dict: Dict[str, List[str]],
            use_streaming: bool = False,
            buffer_length: int = 128,
            iterator_cls=RandomIterator
    ):
        """
        Params:
            filenames_dict: A dict structured as follow:
            {
                src_name1: [src1_patches_image1, ..., src1_patches_imageN1],
                src_name2: [src2_patches_image2, ..., src2_patches_imageN2],
                ...
                src_nameM: [srcM_patches_image1, ..., srcM_patches_imageNM]
            }
        use_streaming: if True, the patches are read on the fly from the disc,
            nothing is kept in memory.
        buffer_length: The number of samples that are stored in the buffer
            (used when "use_streaming" is True).
        iterator_cls: The iterator class used to generate the sequence of
            patches indices.

        """
        # patches reader
        patches_reader = PatchesImagesReader(
            filenames_dict=filenames_dict,
            use_streaming=use_streaming
        )

        super().__init__(
            patches_reader=patches_reader,
            buffer_length=buffer_length,
            iterator_cls=iterator_cls
        )
