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
tree/master/otbtf/ops.py){ .md-button }

The utils module provides some useful Tensorflow ad keras operators to build
and train deep nets.
"""
from typing import List, Tuple, Any
import tensorflow as tf


Tensor = Any
Scalars = List[float] | Tuple[Float]
def one_hot(labels: Tensor, nb_classes: int):
    """
    Converts labels values into one-hot vector.

    Params:
        labels: tensor of label values (shape [x, y, 1])
        nb_classes: number of classes

    Returns:
        one-hot encoded vector (shape [x, y, nb_classes])

    """
    labels_xy = tf.squeeze(tf.cast(labels, tf.int32), axis=-1)  # shape [x, y]
    return tf.one_hot(labels_xy, depth=nb_classes)  # shape [x, y, nb_classes]