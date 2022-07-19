# -*- coding: utf-8 -*-
# ==========================================================================
#
#   Copyright 2018-2019 IRSTEA
#   Copyright 2020-2022 INRAE
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
The tfrecords module provides an implementation for the TFRecords files read/write
"""
import glob
import json
import os
import logging
from functools import partial
import tensorflow as tf
from tqdm import tqdm


class TFRecords:
    """
    This class allows to convert Dataset objects to TFRecords and to load them in dataset tensorflows format.
    """

    def __init__(self, path):
        """
        :param path: Can be a directory where TFRecords must be saved/loaded
        """
        self.dirpath = path
        os.makedirs(self.dirpath, exist_ok=True)
        self.output_types_file = os.path.join(self.dirpath, "output_types.json")
        self.output_shape_file = os.path.join(self.dirpath, "output_shape.json")
        self.output_shape = self.load(self.output_shape_file) if os.path.exists(self.output_shape_file) else None
        self.output_types = self.load(self.output_types_file) if os.path.exists(self.output_types_file) else None

    @staticmethod
    def _bytes_feature(value):
        """
        Convert a value to a type compatible with tf.train.Example.
        :param value: value
        :return a bytes_list from a string / byte.
        """
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def ds2tfrecord(self, dataset, n_samples_per_shard=100, drop_remainder=True):
        """
        Convert and save samples from dataset object to tfrecord files.
        :param dataset: Dataset object to convert into a set of tfrecords
        :param n_samples_per_shard: Number of samples per shard
        :param drop_remainder: Whether additional samples should be dropped. Advisable if using multiworkers training.
                               If True, all TFRecords will have `n_samples_per_shard` samples
        """
        logging.info("%s samples", dataset.size)

        nb_shards = (dataset.size // n_samples_per_shard)
        if not drop_remainder and dataset.size % n_samples_per_shard > 0:
            nb_shards += 1

        output_shapes = {key: output_shape for key, output_shape in dataset.output_shapes.items()}
        self.save(output_shapes, self.output_shape_file)

        output_types = {key: output_type.name for key, output_type in dataset.output_types.items()}
        self.save(output_types, self.output_types_file)

        for i in tqdm(range(nb_shards)):

            if (i + 1) * n_samples_per_shard <= dataset.size:
                nb_sample = n_samples_per_shard
            else:
                nb_sample = dataset.size - i * n_samples_per_shard

            filepath = os.path.join(self.dirpath, f"{i}.records")
            with tf.io.TFRecordWriter(filepath) as writer:
                for _ in range(nb_sample):
                    sample = dataset.read_one_sample()
                    serialized_sample = {name: tf.io.serialize_tensor(fea) for name, fea in sample.items()}
                    features = {name: self._bytes_feature(serialized_tensor) for name, serialized_tensor in
                                serialized_sample.items()}
                    tf_features = tf.train.Features(feature=features)
                    example = tf.train.Example(features=tf_features)
                    writer.write(example.SerializeToString())

    @staticmethod
    def save(data, filepath):
        """
        Save data to JSON format.
        :param data: Data to save json format
        :param filepath: Output file name
        """

        with open(filepath, 'w') as file:
            json.dump(data, file, indent=4)

    @staticmethod
    def load(filepath):
        """
        Return data from JSON format.
        :param filepath: Input file name
        """
        with open(filepath, 'r') as file:
            return json.load(file)

    @staticmethod
    def parse_tfrecord(example, features_types, target_keys, preprocessing_fn=None, **kwargs):
        """
        Parse example object to sample dict.
        :param example: Example object to parse
        :param features_types: List of types for each feature
        :param target_keys: list of keys of the targets
        :param preprocessing_fn: Optional. A preprocessing function that takes input, target as args and returns
                                           a tuple (input_preprocessed, target_preprocessed)
        :param kwargs: some keywords arguments for preprocessing_fn
        """
        read_features = {key: tf.io.FixedLenFeature([], dtype=tf.string) for key in features_types}
        example_parsed = tf.io.parse_single_example(example, read_features)

        for key in read_features.keys():
            example_parsed[key] = tf.io.parse_tensor(example_parsed[key], out_type=features_types[key])

        # Differentiating inputs and outputs
        input_parsed = {key: value for (key, value) in example_parsed.items() if key not in target_keys}
        target_parsed = {key: value for (key, value) in example_parsed.items() if key in target_keys}

        if preprocessing_fn:
            input_parsed, target_parsed = preprocessing_fn(input_parsed, target_parsed, **kwargs)

        return input_parsed, target_parsed

    def read(self, batch_size, target_keys, n_workers=1, drop_remainder=True, shuffle_buffer_size=None,
             preprocessing_fn=None, **kwargs):
        """
        Read all tfrecord files matching with pattern and convert data to tensorflow dataset.
        :param batch_size: Size of tensorflow batch
        :param target_keys: Keys of the target, e.g. ['s2_out']
        :param n_workers: number of workers, e.g. 4 if using 4 GPUs
                                             e.g. 12 if using 3 nodes of 4 GPUs
        :param drop_remainder: whether the last batch should be dropped in the case it has fewer than
                               `batch_size` elements. True is advisable when training on multiworkers.
                               False is advisable when evaluating metrics so that all samples are used
        :param shuffle_buffer_size: if None, shuffle is not used. Else, blocks of shuffle_buffer_size
                                    elements are shuffled using uniform random.
        :param preprocessing_fn: Optional. A preprocessing function that takes (input, target) as args and returns
                                 a tuple (input_preprocessed, target_preprocessed). Typically, target_preprocessed
                                 must be computed accordingly to (1) what the model outputs and (2) what training loss
                                 needs. For instance, for a classification problem, the model will likely output the
                                 softmax, or activation neurons, for each class, and the cross entropy loss requires
                                 labels in one hot encoding. In this case, the preprocessing_fn has to transform the
                                 labels values (integer ranging from [0, n_classes]) in one hot encoding (vector of 0
                                 and 1 of length n_classes). The preprocessing_fn should not implement such things as
                                 radiometric transformations from input to input_preprocessed, because those are
                                 performed inside the model itself (see `model.normalize()`).
        :param kwargs: some keywords arguments for preprocessing_fn
        """
        options = tf.data.Options()
        if shuffle_buffer_size:
            options.experimental_deterministic = False  # disable order, increase speed
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO  # for multiworker
        parse = partial(self.parse_tfrecord, features_types=self.output_types, target_keys=target_keys,
                        preprocessing_fn=preprocessing_fn, **kwargs)

        # TODO: to be investigated :
        # 1/ num_parallel_reads useful ? I/O bottleneck of not ?
        # 2/ num_parallel_calls=tf.data.experimental.AUTOTUNE useful ?
        tfrecords_pattern_path = os.path.join(self.dirpath, "*.records")
        matching_files = glob.glob(tfrecords_pattern_path)
        logging.info('Searching TFRecords in %s...', tfrecords_pattern_path)
        logging.info('Number of matching TFRecords: %s', len(matching_files))
        matching_files = matching_files[:n_workers * (len(matching_files) // n_workers)]  # files multiple of workers
        nb_matching_files = len(matching_files)
        if nb_matching_files == 0:
            raise Exception(f"At least one worker has no TFRecord file in {tfrecords_pattern_path}. Please ensure that "
                            "the number of TFRecord files is greater or equal than the number of workers!")
        logging.info('Reducing number of records to : %s', nb_matching_files)
        dataset = tf.data.TFRecordDataset(matching_files)  # , num_parallel_reads=2)  # interleaves reads from xxx files
        dataset = dataset.with_options(options)  # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if shuffle_buffer_size:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset
