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
tree/master/otbtf/model.py){ .md-button }

Base class for models.
"""
from typing import List, Dict, Any
import abc
import logging
import tensorflow as tf

TensorsDict = Dict[str, Any]


class ModelBase(abc.ABC):
    """
    Base class for all models
    """

    def __init__(
            self,
            dataset_element_spec: tf.TensorSpec,
            input_keys: List[str] = None,
            inference_cropping: List[int] = None
    ):
        """
        Model initializer, must be called **inside** the strategy.scope().

        Args:
            dataset_element_spec: the dataset elements specification (shape,
                dtype, etc). Can be retrieved from a dataset instance `ds`
                simply with `ds.element_spec`
            input_keys: Optional keys of the inputs used in the model. If not
                specified, all inputs from the dataset will be considered.
            inference_cropping: list of number of pixels to be removed on each
                side of the output for inference. Additional outputs are
                created in the model, not used during training, only during
                inference. Default [16, 32, 64, 96, 128]

        """
        # Retrieve dataset inputs shapes
        dataset_input_element_spec = dataset_element_spec[0]
        logging.info(
            "Dataset input element spec: %s", dataset_input_element_spec
        )

        if input_keys:
            self.dataset_input_keys = input_keys
            logging.info("Using input keys: %s", self.dataset_input_keys)
        else:
            self.dataset_input_keys = list(dataset_input_element_spec)
            logging.info(
                "Found dataset input keys: %s", self.dataset_input_keys
            )

        self.inputs_shapes = {
            key: dataset_input_element_spec[key].shape[1:]
            for key in self.dataset_input_keys
        }
        logging.info("Inputs shapes: %s", self.inputs_shapes)

        # Setup cropping, normalization function
        self.inference_cropping = inference_cropping or [16, 32, 64, 96, 128]
        logging.info("Inference cropping values: %s", self.inference_cropping)

        # Create model
        self.model = self.create_network()

    def __getattr__(self, name: str) -> Any:
        """
        This method is called when the default attribute access fails. We
        choose to try to access the attribute of self.model. Thus, any method
        of `keras.Model()` can be used transparently, e.g. `model.summary()`
        or model.fit()

        Args:
            name: name of the attribute

        Returns:
            attribute

        """
        return getattr(self.model, name)

    def get_inputs(self) -> TensorsDict:
        """
        This method returns the dict of keras.Input
        """
        # Create Keras inputs
        model_inputs = {}
        for key in self.dataset_input_keys:
            new_shape = list(self.inputs_shapes[key])
            logging.info("Original shape for input %s: %s", key, new_shape)
            # Here we modify the x and y dims of >2D tensors to enable any
            # image size at input
            if len(new_shape) > 2:
                new_shape[0] = None
                new_shape[1] = None
            placeholder = tf.keras.Input(shape=new_shape, name=key)
            logging.info("New shape for input %s: %s", key, new_shape)
            model_inputs.update({key: placeholder})
        return model_inputs

    @abc.abstractmethod
    def get_outputs(self, normalized_inputs: TensorsDict) -> TensorsDict:
        """
        Implementation of the model, from the normalized inputs.

        Params:
            normalized_inputs: normalized inputs, as generated from
                `self.normalize_inputs()`

        Returns:
            model outputs

        """
        raise NotImplementedError(
            "This method has to be implemented. Here you code the model :)"
        )

    def normalize_inputs(self, inputs: TensorsDict) -> TensorsDict:
        """
        Normalize the model inputs.
        Takes the dict of inputs and returns a dict of normalized inputs.

        Params:
            inputs: model inputs

        Returns:
            a dict of normalized model inputs

        """
        logging.warning(
            "normalize_input() undefined. No normalization of the model "
            "inputs will be performed. You can implement the function in your "
            "model class if you want."
        )
        return inputs

    def postprocess_outputs(
            self,
            outputs: TensorsDict,
            inputs: TensorsDict = None,
            normalized_inputs: TensorsDict = None
    ) -> TensorsDict:
        """
        Post-process the model outputs.
        Takes the dicts of inputs and outputs, and returns a dict of
        post-processed outputs.
        The default implementation provides a set of cropped output tensors.

        Params:
            outputs: dict of model outputs
            inputs: dict of model inputs (optional)
            normalized_inputs: dict of normalized model inputs (optional)

        Returns:
            a dict of post-processed model outputs

        """

        # Add extra outputs for inference
        extra_outputs = {}
        for out_key, out_tensor in outputs.items():
            for crop in self.inference_cropping:
                extra_output_key = cropped_tensor_name(out_key, crop)
                extra_output_name = cropped_tensor_name(
                    out_tensor._keras_history.layer.name, crop
                )
                logging.info(
                    "Adding extra output for tensor %s with crop %s (%s)",
                    out_key, crop, extra_output_name
                )
                cropped = out_tensor[:, crop:-crop, crop:-crop, :]
                identity = tf.keras.layers.Activation(
                    'linear', name=extra_output_name
                )
                extra_outputs[extra_output_key] = identity(cropped)

        return extra_outputs

    def create_network(self) -> tf.keras.Model:
        """
        This method returns the Keras model. This needs to be called
        **inside** the strategy.scope(). Can be reimplemented depending on the
        needs.

        Returns:
            the keras model
        """

        # Get the model inputs
        inputs = self.get_inputs()
        logging.info("Model inputs: %s", inputs)

        # Normalize the inputs
        normalized_inputs = self.normalize_inputs(inputs=inputs)
        logging.info("Normalized model inputs: %s", normalized_inputs)

        # Build the model
        outputs = self.get_outputs(normalized_inputs=normalized_inputs)
        logging.info("Model outputs: %s", outputs)

        # Post-processing for inference
        postprocessed_outputs = self.postprocess_outputs(
            outputs=outputs,
            inputs=inputs,
            normalized_inputs=normalized_inputs
        )
        outputs.update(postprocessed_outputs)

        # Return the keras model
        return tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name=self.__class__.__name__
        )

    def summary(self, strategy=None):
        """
        Wraps the summary printing of the model. When multiworker strategy,
        only prints if the worker is chief

        Params:
            strategy: strategy

        """
        if not strategy or _is_chief(strategy):
            self.model.summary(line_length=150)

    def plot(self, output_path: str, strategy=None, show_shapes: bool = False):
        """
        Enables to save a figure representing the architecture of the network.
        Needs pydot and graphviz to work (`pip install pydot` and
        https://graphviz.gitlab.io/download/)

        Params:
            output_path: output path for the schema
            strategy: strategy
            show_shapes: annotate with shapes values (True or False)

        """
        assert self.model, "Plot() only works if create_network() has been " \
                           "called beforehand"

        # When multiworker strategy, only plot if the worker is chief
        if not strategy or _is_chief(strategy):
            tf.keras.utils.plot_model(
                self.model, output_path, show_shapes=show_shapes
            )


def _is_chief(strategy):
    """
    Tell if the current worker is the chief.

    Params:
        strategy: strategy

    Returns:
        True if the current worker is the chief, False else

    """
    # Note: there are two possible `TF_CONFIG` configuration.
    #   1) In addition to `worker` tasks, a `chief` task type is use;
    #      in this case, this function should be modified to
    #      `return task_type == 'chief'`.
    #   2) Only `worker` task type is used; in this case, worker 0 is
    #      regarded as the chief. The implementation demonstrated here
    #      is for this case.
    # For the purpose of this Colab section, the `task_type is None` case
    # is added because it is effectively run with only a single worker.

    if strategy.cluster_resolver:  # this means MultiWorkerMirroredStrategy
        task_type = strategy.cluster_resolver.task_type
        task_id = strategy.cluster_resolver.task_id
        return (task_type == 'chief') \
            or (task_type == 'worker' and task_id == 0) \
            or task_type is None
    # strategy with only one worker
    return True


def cropped_tensor_name(tensor_name: str, crop: int):
    """
    A name for the padded tensor

    Params:
        tensor_name: tensor name
        crop: cropping value

    Returns:
        name for the cropped tensor

    """
    return f"{tensor_name}_crop{crop}"
