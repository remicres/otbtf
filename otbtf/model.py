# -*- coding: utf-8 -*-
""" Base class for models"""
import abc
import logging

import tensorflow as tf
from tensorflow import keras
from otbtf.utils import _is_chief


class ModelBase(abc.ABC):
    """
    Base class for all models
    """

    def __init__(self, dataset_input_keys, model_output_keys, dataset_shapes, target_cropping=None,
                 inference_cropping=None, normalize_fn=None):
        """
        Model base class

        :param dataset_input_keys: list of dataset keys used for the training
        :param model_output_keys: list of the model outputs keys
        :param dataset_shapes: a dict() of shapes
        :param target_cropping: Optional. Number of pixels to be removed on each side of the target. This is used when
                                training the model and can mitigate the effects of convolution
        :param inference_cropping: list of number of pixels to be removed on each side of the output during inference.
                                   This list creates some additional outputs in the model, not used during training,
                                   only during inference. Default [16, 32, 64, 96, 128]
        :param normalize_fn: a normalization function that can be added inside the Keras model. This function takes a
                             dict of inputs and returns a dict of normalized inputs. Optional
        """
        self.dataset_input_keys = dataset_input_keys
        self.model_output_keys = model_output_keys
        self.dataset_shapes = dataset_shapes
        self.model = None
        self.target_cropping = target_cropping
        if inference_cropping is None:
            inference_cropping = [16, 32, 64, 96, 128]
        self.inference_cropping = inference_cropping
        self.normalize_fn = normalize_fn

    def __getattr__(self, name):
        """This method is called when the default attribute access fails. We choose to try to access the attribute of
        self.model. Thus, any method of keras.Model() can be used transparently, e.g. model.summary() or model.fit()"""
        if not self.model:
            logging.warning("model is None. You should call `create_network()` before using it!")
            logging.warning("Creating the neural network. Note that training could fail if using keras distribution "
                            "strategy such as MirroredStrategy. Best practice is to call `create_network()` inside "
                            "`with strategy.scope():`")
            self.create_network()
        return getattr(self.model, name)

    def get_inputs(self):
        """
        This method returns the dict of keras.Input
        """
        # Create Keras inputs
        model_inputs = {}
        for key in self.dataset_input_keys:
            shape = self.dataset_shapes[key]
            new_shape = list(shape)
            if shape[0] is None or (len(shape) > 3):  # for backward comp (OTBTF<3.2.2), remove the potential batch dim
                new_shape = shape[1:]
            # Here we modify the x and y dims of >2D tensors to enable any image size at input
            if len(new_shape) > 2:
                new_shape[0] = None
                new_shape[1] = None
            placeholder = keras.Input(shape=new_shape, name=key)
            logging.info("New shape for input %s: %s", key, new_shape)
            model_inputs.update({key: placeholder})
        return model_inputs

    @abc.abstractmethod
    def get_outputs(self, inputs):
        """
        Implementation of the model
        :param inputs: inputs, either keras.Input or normalized_inputs
        :return: a dict of outputs tensors of the model
        """
        raise NotImplemented("This method has to be implemented. Here you code the model :)")

    def create_network(self):
        """
        This method returns the Keras model. This needs to be called **inside** the strategy.scope()
        :return: the keras model
        """

        # Get the model inputs
        model_inputs = self.get_inputs()

        # Normalize the inputs. If some input keys are not handled by normalized_fn, these inputs are not normalized
        normalized_inputs = model_inputs.copy()
        normalized_inputs.update(self.normalize_fn(model_inputs))

        # Build the model
        outputs = self.get_outputs(normalized_inputs)

        # Add extra outputs for inference
        extra_outputs = {}
        for out_key, out_tensor in outputs.items():
            for crop in self.inference_cropping:
                extra_output_key = cropped_tensor_name(out_key, crop)
                extra_output_name = cropped_tensor_name(out_tensor._keras_history.layer.name, crop)
                extra_output = tf.keras.layers.Cropping2D(cropping=crop, name=extra_output_name)(out_tensor)
                extra_outputs[extra_output_key] = extra_output
        outputs.update(extra_outputs)

        # Return the keras model
        self.model = keras.Model(inputs=model_inputs, outputs=outputs, name=self.__class__.__name__)

    def summary(self, strategy=None):
        """
        Wraps the summary printing of the model. When multiworker strategy, only prints if the worker is chief
        """
        if not strategy or _is_chief(strategy):
            self.model.summary(line_length=150)

    def plot(self, output_path, strategy=None):
        """
        Enables to save a figure representing the architecture of the network.
        //!\\ only works if create_network() has been called beforehand
        Needs pydot and graphviz to work (`pip install pydot` and https://graphviz.gitlab.io/download/)
        """
        assert self.model, "Plot() only works if create_network() has been called beforehand"

        # When multiworker strategy, only plot if the worker is chief
        if not strategy or _is_chief(strategy):
            # Build a simplified model, without normalization nor extra outputs.
            # This model is only used for plotting the architecture thanks to `keras.utils.plot_model`
            inputs = self.get_inputs()  # inputs without normalization
            outputs = self.get_outputs(inputs)  # raw model outputs
            model_simplified = keras.Model(inputs=inputs, outputs=outputs, name=self.__class__.__name__ + '_simplified')
            keras.utils.plot_model(model_simplified, output_path)


def cropped_tensor_name(tensor_name, crop):
    """
    A name for the padded tensor
    :param tensor_name: tensor name
    :param pad: pad value
    :return: name
    """
    return "{}_crop{}".format(tensor_name, crop)
