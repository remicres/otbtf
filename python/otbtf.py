import threading
import multiprocessing
import time
import numpy as np
import tensorflow as tf
import gdal
import logging
import math

"""
---------------------------------------------------- Buffer class ------------------------------------------------------
"""


class Buffer:
    """
    Used to store and access list of objects
    """

    def __init__(self, max_length):
        self.max_length = max_length
        self.container = []

    def size(self):
        return len(self.container)

    def add(self, x):
        self.container.append(x)
        assert (self.size() <= self.max_length)

    def is_complete(self):
        return self.size() == self.max_length


"""
------------------------------------------------ RandomIterator class --------------------------------------------------
"""


class RandomIterator:
    """
    Pick a random number in the [0, handler.size) range.
    """

    def __init__(self, handler):

        self.indices = np.arange(0, handler.size)
        self.shuffle()
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        current_index = self.indices[self.count]
        if self.count < len(self.indices) - 1:
            self.count += 1
        else:
            self.shuffle()
            self.count = 0
        return current_index

    def shuffle(self):
        np.random.shuffle(self.indices)


"""
-------------------------------------------------- Patches reader class ------------------------------------------------
"""


class PatchesReader:
    def __init__(self, filenames_dict, use_streaming=False):
        """
        :param filenames_dict: A dict() structured as follow:
            {src_name1: [src1_patches_image1.tif, ..., src1_patches_imageN1.tif],
             src_name2: [src2_patches_image2.tif, ..., src2_patches_imageN2.tif],
             ...
             src_nameM: [srcM_patches_image1.tif, ..., srcM_patches_imageNM.tif]}
        :param use_streaming: if True, the patches are read on the fly from the disc, nothing is kept in memory.
        """

        assert (len(filenames_dict.values()) > 0)

        # ds dict
        self.ds = dict()
        for src_key, src_filenames in filenames_dict.items():
            self.ds[src_key] = []
            for src_filename in src_filenames:
                self.ds[src_key].append(self.gdal_open(src_filename))

        if len(set([len(ds_list) for ds_list in self.ds.values()])) != 1:
            raise Exception("Each source must have the same number of patches images")

        # streaming on/off
        self.use_streaming = use_streaming

        # ds check
        nb_of_patches = {key: 0 for key in self.ds}
        self.nb_of_channels = dict()
        for src_key, ds_list in self.ds.items():
            for ds in ds_list:
                nb_of_patches[src_key] += self._get_nb_of_patches(ds)
                if src_key not in self.nb_of_channels:
                    self.nb_of_channels[src_key] = ds.RasterCount
                else:
                    if self.nb_of_channels[src_key] != ds.RasterCount:
                        raise Exception("All patches images from one source must have the same number of channels!"
                                        "Error happened for source: {}".format(src_key))
        if len(set(nb_of_patches.values())) != 1:
            raise Exception("Sources must have the same number of patches! Number of patches: {}".format(nb_of_patches))

        # ds sizes
        src_key_0 = list(self.ds)[0]  # first key
        self.ds_sizes = [self._get_nb_of_patches(ds) for ds in self.ds[src_key_0]]
        self.size = sum(self.ds_sizes)

        # if use_streaming is False, we store in memory all patches images
        if not self.use_streaming:
            patches_list = {src_key: [self.read_as_np_arr(ds) for ds in self.ds[src_key]] for src_key in self.ds}
            self.patches_buffer = {src_key: np.concatenate(patches_list[src_key], axis=-1) for src_key in self.ds}

    def _get_ds_and_offset_from_index(self, index):
        offset = index
        for i, ds_size in enumerate(self.ds_sizes):
            if offset < ds_size:
                break
            offset -= ds_size

        return i, offset

    @staticmethod
    def gdal_open(filename):
        ds = gdal.Open(filename)
        if ds is None:
            raise Exception("Unable to open file {}".format(filename))
        return ds

    @staticmethod
    def _get_nb_of_patches(ds):
        return int(ds.RasterYSize / ds.RasterXSize)

    @staticmethod
    def read_as_np_arr(ds, as_patches=True):
        buffer = ds.ReadAsArray()
        szx = ds.RasterXSize
        szy = ds.RasterYSize
        n = int(szy / szx)
        if len(buffer.shape) == 3:
            buffer = np.transpose(buffer, axes=(1, 2, 0))
        if not as_patches:
            n = 1
        return np.float32(buffer.reshape((n, szx, szx, ds.RasterCount)))

    @staticmethod
    def _read_extract_as_np_arr(ds, offset):
        assert (ds is not None)
        psz = ds.RasterXSize
        yoff = int(offset * psz)
        assert (yoff + psz <= ds.RasterYSize)
        buffer = ds.ReadAsArray(0, yoff, psz, psz)
        if len(buffer.shape) == 3:
            buffer = np.transpose(buffer, axes=(1, 2, 0))
        return np.float32(buffer)

    def get_sample(self, index):
        """ Return one sample of the dataset. index must be in the [0, self.size) range. """
        assert (0 <= index)
        assert (index < self.size)

        if not self.use_streaming:
            res = {src_key: self.patches_buffer[src_key][index, :, :, :] for src_key in self.ds}
        else:
            i, offset = self._get_ds_and_offset_from_index(index)
            res = {src_key: self._read_extract_as_np_arr(self.ds[src_key][i], offset) for src_key in self.ds}

        return res

    def get_stats(self):
        """ Return some statistics """
        logging.info("Computing stats")
        if not self.use_streaming:
            axis = (0, 1, 2)  # (row, col)
            stats = {src_key: {"min": np.amin(patches_buffer, axis=axis),
                               "max": np.amax(patches_buffer, axis=axis),
                               "mean": np.mean(patches_buffer, axis=axis),
                               "std": np.std(patches_buffer, axis=axis)} for src_key, patches_buffer in
                     self.patches_buffer.items()}
        else:
            axis = (0, 1)  # (row, col)
            def _filled(value):
                return {src_key: value * np.ones((self.nb_of_channels[src_key])) for src_key in self.ds}
            _maxs = _filled(0.0)
            _mins = _filled(100000.0)
            _sums = _filled(0.0)
            _sqsums = _filled(0.0)
            for index in range(self.size):
                sample = self.get_sample(index=index)
                for src_key, np_arr in sample.items():
                    _mins[src_key] = np.minimum(np.amin(np_arr, axis=axis).flatten(), _mins[src_key])
                    _maxs[src_key] = np.maximum(np.amax(np_arr, axis=axis).flatten(), _maxs[src_key])
                    _sums[src_key] += np.sum(np_arr, axis=axis).flatten()
                    _sqsums[src_key] += np.sum(np.square(np_arr), axis=axis).flatten()

            rsize = 1.0 / float(self.size)
            stats = {src_key: {"min": _mins[src_key],
                               "max": _maxs[src_key],
                               "mean": rsize * _sums[src_key],
                               "std": math.sqrt(rsize * _sqsums[src_key] - (rsize * _sums[src_key]) ** 2)
                               } for src_key, patches_buffer in self.patches_buffer.items()}
        logging.info("Stats: {}".format(stats))
        return stats


"""
--------------------------------------------------- Dataset class ------------------------------------------------------
"""


class Dataset:
    """
    Handles the "mining" of the patches.
    This class has a thread that extract tuples from the patches readers, while ensuring the access of already gathered
    tuples.
    """

    def __init__(self, filenames_dict, use_streaming=False, buffer_length=128, Iterator=RandomIterator):
        """
        :param filenames_dict: A dict() structured as follow:
            {src_name1: [src1_patches_image1, ..., src1_patches_imageN1],
             src_name2: [src2_patches_image2, ..., src2_patches_imageN2],
             ...
             src_nameM: [srcM_patches_image1, ..., srcM_patches_imageNM]}

        :param use_streaming: if True, the patches are read on the fly from the disc, nothing is kept in memory.
        :param buffer_length: The number of samples that are stored in the buffer when "use_streaming" is True.
        :param Iterator: The iterator class used to generate the sequence of patches indices.
        """

        # patches reader
        self.patches_reader = PatchesReader(filenames_dict=filenames_dict, use_streaming=use_streaming)
        self.size = self.patches_reader.size

        # iterator
        self.iterator = Iterator(handler=self.patches_reader)

        # Get patches sizes and type, of the first sample of the first tile
        self.output_types = dict()
        self.output_shapes = dict()
        one_sample = self.patches_reader.get_sample(index=0)
        for src_key, np_arr in one_sample.items():
            self.output_shapes[src_key] = np_arr.shape
            self.output_types[src_key] = tf.dtypes.as_dtype(np_arr.dtype)

        logging.info("output_types: {}".format(self.output_types))
        logging.info("output_shapes: {}".format(self.output_shapes))

        # buffers
        self.miner_buffer = Buffer(buffer_length)
        self.consumer_buffer = Buffer(buffer_length)
        self.consumer_buffer_pos = 0
        self.tot_wait = 0
        self.miner_thread = self._summon_miner_thread()
        self.read_lock = multiprocessing.Lock()
        self._dump()

        # Prepare tf dataset for one epoch
        self.tf_dataset = tf.data.Dataset.from_generator(self._generator,
                                                         output_types=self.output_types,
                                                         output_shapes=self.output_shapes).repeat(1)

    def get_stats(self):
        """
        @return: the dataset statistics, computed by the patches reader
        """
        return self.patches_reader.get_stats()

    def read_one_sample(self):
        """
        Read one element of the consumer_buffer
        The lock is used to prevent different threads to read and update the internal counter concurrently
        """
        with self.read_lock:
            output = None
            if self.consumer_buffer_pos < self.consumer_buffer.max_length:
                output = self.consumer_buffer.container[self.consumer_buffer_pos]
                self.consumer_buffer_pos += 1
            if self.consumer_buffer_pos == self.consumer_buffer.max_length:
                self._dump()
                self.consumer_buffer_pos = 0
            return output

    def _dump(self):
        """
        This function dumps the miner_buffer into the consumer_buffer, and restart the miner_thread
        """
        # Wait for miner to finish his job
        t = time.time()
        self.miner_thread.join()
        self.tot_wait += time.time() - t

        # Copy miner_buffer.container --> consumer_buffer.container
        self.consumer_buffer.container = [elem for elem in self.miner_buffer.container]

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
            try:
                index = next(self.iterator)
                new_sample = self.patches_reader.get_sample(index=index)
                self.miner_buffer.add(new_sample)
            except Exception as e:
                logging.warning("Error during collecting samples: {}".format(e))

    def _summon_miner_thread(self):
        """
        Create and starts the thread for the data collect
        """
        t = threading.Thread(target=self._collect)
        t.start()
        return t

    def _generator(self):
        """
        Generator function, used for the tf dataset
        """
        for elem in range(self.size):
            yield self.read_one_sample()

    def get_tf_dataset(self, batch_size, drop_remainder=True):
        """
        Returns a TF dataset, ready to be used with the provided batch size
        :param batch_size: the batch size
        :param drop_remainder: drop incomplete batches
        :return: The TF dataset
        """
        return self.tf_dataset.batch(batch_size, drop_remainder=drop_remainder)

    def get_total_wait_in_seconds(self):
        """
        Returns the number of seconds during which the data gathering was delayed because of I/O bottleneck
        :return: duration in seconds
        """
        return self.tot_wait
