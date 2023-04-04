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
tree/master/otbtf/tfrecords.py){ .md-button }

The tfrecords module provides an implementation for the TFRecords files
read/write
"""
import glob
import json
import logging
import os
from functools import partial

from typing import Any, List, Dict, Callable
import tensorflow as tf
from tqdm import tqdm


class TFRecords:
    """
    This class allows to convert Dataset objects to TFRecords and to load them
    in dataset tensorflow format.
    """

    def __init__(self, path: str):
        """
        Params:
            path: Can be a directory where TFRecords must be saved/loaded
        """
        self.dirpath = path
        os.makedirs(self.dirpath, exist_ok=True)
        self.output_types_file = os.path.join(
            self.dirpath, "output_types.json"
        )
        self.output_shapes_file = os.path.join(
            self.dirpath, "output_shapes.json"
        )
        self.output_shapes = self.load(self.output_shapes_file) \
            if os.path.exists(self.output_shapes_file) else None
        self.output_types = self.load(self.output_types_file) \
            if os.path.exists(self.output_types_file) else None

    @staticmethod
    def _bytes_feature(value):
        """
        Convert a value to a type compatible with tf.train.Example.

        Params:
            value: value

        Returns:
            a bytes_list from a string / byte.
        """
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from
            # an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def ds2tfrecord(
            self,
            dataset: Any,
            n_samples_per_shard: int = 100,
            drop_remainder: bool = True
    ):
        """
        Convert and save samples from dataset object to tfrecord files.

        Params:
            dataset: Dataset object to convert into a set of tfrecords
            n_samples_per_shard: Number of samples per shard
            drop_remainder: Whether additional samples should be dropped.
                Advisable if using multiworkers training. If True, all
                TFRecords will have `n_samples_per_shard` samples

        """
        logging.info("%s samples", dataset.size)

        nb_shards = dataset.size // n_samples_per_shard
        if not drop_remainder and dataset.size % n_samples_per_shard > 0:
            nb_shards += 1

        output_shapes = dict(dataset.output_shapes.items())
        self.save(output_shapes, self.output_shapes_file)

        output_types = {
            key: output_type.name
            for key, output_type in dataset.output_types.items()
        }
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
                    serialized_sample = {
                        name: tf.io.serialize_tensor(fea)
                        for name, fea in sample.items()
                    }
                    features = {
                        name: self._bytes_feature(serialized_tensor)
                        for name, serialized_tensor in
                        serialized_sample.items()
                    }
                    tf_features = tf.train.Features(feature=features)
                    example = tf.train.Example(features=tf_features)
                    writer.write(example.SerializeToString())

    @staticmethod
    def save(data: Dict[str, Any], filepath: str):
        """
        Save data to JSON format.

        Params:
            data: Data to save json format
            filepath: Output file name

        """
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=4)

    @staticmethod
    def load(filepath: str):
        """
        Return data from JSON format.

        Args:
            filepath: Input file name

        """
        with open(filepath, 'r') as file:
            return json.load(file)

    def parse_tfrecord(
            self,
            example: Any,
            target_keys: List[str],
            preprocessing_fn: Callable = None,
            **kwargs
    ):
        """
        Parse example object to sample dict.

        Params:
            example: Example object to parse
            target_keys: list of keys of the targets
            preprocessing_fn: Optional. A preprocessing function that process
                the input example
            kwargs: some keywords arguments for preprocessing_fn

        """
        read_features = {
            key: tf.io.FixedLenFeature([], dtype=tf.string)
            for key in self.output_types
        }
        example_parsed = tf.io.parse_single_example(example, read_features)

        # Tensor with right data type
        for key, out_type in self.output_types.items():
            example_parsed[key] = tf.io.parse_tensor(
                example_parsed[key],
                out_type=out_type
            )

        # Ensure shape
        for key, shape in self.output_shapes.items():
            example_parsed[key] = tf.ensure_shape(example_parsed[key], shape)

        # Preprocessing
        example_parsed_prep = preprocessing_fn(
            example_parsed, **kwargs
        ) if preprocessing_fn else example_parsed

        # Differentiating inputs and targets
        input_parsed = {
            key: value for (key, value) in example_parsed_prep.items()
            if key not in target_keys
        }
        target_parsed = {
            key: value for (key, value) in example_parsed_prep.items()
            if key in target_keys
        }

        return input_parsed, target_parsed

    def read(
            self,
            batch_size: int,
            target_keys: List[str],
            n_workers: int = 1,
            drop_remainder: bool = True,
            shuffle_buffer_size: int = None,
            preprocessing_fn: Callable = None,
            shard_policy=tf.data.experimental.AutoShardPolicy.AUTO,
            prefetch_buffer_size: int = tf.data.experimental.AUTOTUNE,
            num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
            **kwargs
    ):
        """
        Read all tfrecord files matching with pattern and convert data to
        tensorflow dataset.

        Params:
            batch_size: Size of tensorflow batch
            target_keys: Keys of the target, e.g. ['s2_out']
            n_workers: number of workers, e.g. 4 if using 4 GPUs, e.g. 12 if
                using 3 nodes of 4 GPUs
            drop_remainder: whether the last batch should be dropped in the
                case it has fewer than `batch_size` elements. True is
                advisable when training on multiworkers. False is advisable
                when evaluating metrics so that all samples are used
            shuffle_buffer_size: if None, shuffle is not used. Else, blocks of
                shuffle_buffer_size elements are shuffled using uniform random.
            preprocessing_fn: Optional. A preprocessing function that takes
                input examples as args and returns the preprocessed input
                examples. Typically, examples are composed of model inputs and
                targets. Model inputs and model targets must be computed
                accordingly to (1) what the model outputs and (2) what
                training loss needs. For instance, for a classification
                problem, the model will likely output the softmax, or
                activation neurons, for each class, and the cross entropy loss
                requires labels in one hot encoding. In this case, the
                `preprocessing_fn` has to transform the labels values (integer
                ranging from [0, n_classes]) in one hot encoding (vector of 0
                and 1 of length n_classes). The `preprocessing_fn` should not
                implement such things as radiometric transformations from
                input to input_preprocessed, because those are performed
                inside the model itself (see
                `otbtf.ModelBase.normalize_inputs()`).
            shard_policy: sharding policy for the TFRecord dataset options
            prefetch_buffer_size: buffer size for the prefetch operation
            num_parallel_calls: number of parallel calls for the parsing +
                preprocessing step
            kwargs: some keywords arguments for `preprocessing_fn`

        """
        for dic, file in zip([self.output_types,
                              self.output_shapes],
                             [self.output_types_file,
                              self.output_shapes_file]):
            assert dic, f"The file {file} is missing!"

        options = tf.data.Options()
        if shuffle_buffer_size:
            # disable order, increase speed
            options.experimental_deterministic = False
        # for multiworker
        options.experimental_distribute.auto_shard_policy = shard_policy
        parse = partial(
            self.parse_tfrecord,
            target_keys=target_keys,
            preprocessing_fn=preprocessing_fn,
            **kwargs
        )

        # 1/ num_parallel_reads useful ? I/O bottleneck of not ?
        # 2/ num_parallel_calls=tf.data.experimental.AUTOTUNE useful ?
        tfrecords_pattern_path = os.path.join(self.dirpath, "*.records")
        matching_files = glob.glob(tfrecords_pattern_path)
        logging.info('Searching TFRecords in %s...', tfrecords_pattern_path)
        logging.info('Number of matching TFRecords: %s', len(matching_files))
        matching_files = matching_files[:n_workers * (
                len(matching_files) // n_workers)]  # files multiple of workers
        nb_matching_files = len(matching_files)
        if nb_matching_files == 0:
            raise Exception(
                "At least one worker has no TFRecord file in "
                f"{tfrecords_pattern_path}. Please ensure that the number of "
                "TFRecord files is greater or equal than the number of "
                "workers!"
            )
        logging.info('Reducing number of records to : %s', nb_matching_files)
        dataset = tf.data.TFRecordDataset(
            matching_files
        )  # , num_parallel_reads=2)  # interleaves reads from xxx files
        # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.with_options(options)
        dataset = dataset.map(parse, num_parallel_calls=num_parallel_calls)
        if shuffle_buffer_size:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

        return dataset
