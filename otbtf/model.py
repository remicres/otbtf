# -*- coding: utf-8 -*-
""" Base class for models"""
import abc
import logging
import tensorflow
from otbtf.utils import _is_chief, cropped_tensor_name


class ModelBase(abc.ABC):
    """
    Base class for all models
    """

    def __init__(self, dataset_element_spec, input_keys=None, inference_cropping=None):
        """
        Model initializer, must be called **inside** the strategy.scope().

        :param dataset_element_spec: the dataset elements specification (shape, dtype, etc). Can be retrieved from the
                                     dataset instance simply with `ds.element_spec`
        :param input_keys: Optional. the keys of the inputs used in the model. If not specified, all inputs from the
                           dataset will be considered.
        :param inference_cropping: list of number of pixels to be removed on each side of the output during inference.
                                   This list creates some additional outputs in the model, not used during training,
                                   only during inference. Default [16, 32, 64, 96, 128]
        """
        # Retrieve dataset inputs shapes
        dataset_input_element_spec = dataset_element_spec[0]
        logging.info("Dataset input element spec: %s", dataset_input_element_spec)

        if input_keys:
            self.dataset_input_keys = input_keys
            logging.info("Using input keys: %s", self.dataset_input_keys)
        else:
            self.dataset_input_keys = list(dataset_input_element_spec)
            logging.info("Found dataset input keys: %s", self.dataset_input_keys)

        self.inputs_shapes = {key: dataset_input_element_spec[key].shape[1:] for key in self.dataset_input_keys}
        logging.info("Inputs shapes: %s", self.inputs_shapes)

        # Setup cropping, normalization function
        self.inference_cropping = [16, 32, 64, 96, 128] if not inference_cropping else inference_cropping
        logging.info("Inference cropping values: %s", self.inference_cropping)

        # Create model
        self.model = self.create_network()

    def __getattr__(self, name):
        """This method is called when the default attribute access fails. We choose to try to access the attribute of
        self.model. Thus, any method of keras.Model() can be used transparently, e.g. model.summary() or model.fit()"""
        return getattr(self.model, name)

    def get_inputs(self):
        """
        This method returns the dict of keras.Input
        """
        # Create Keras inputs
        model_inputs = {}
        for key in self.dataset_input_keys:
            new_shape = list(self.inputs_shapes[key])
            logging.info("Original shape for input %s: %s", key, new_shape)
            # Here we modify the x and y dims of >2D tensors to enable any image size at input
            if len(new_shape) > 2:
                new_shape[0] = None
                new_shape[1] = None
            placeholder = tensorflow.keras.Input(shape=new_shape, name=key)
            logging.info("New shape for input %s: %s", key, new_shape)
            model_inputs.update({key: placeholder})
        return model_inputs

    @abc.abstractmethod
    def get_outputs(self, normalized_inputs):
        """
        Implementation of the model, from the normalized inputs.

        :param normalized_inputs: normalized inputs, as generated from `self.normalize_inputs()`
        :return: dict of model outputs
        """
        raise NotImplementedError("This method has to be implemented. Here you code the model :)")

    def normalize_inputs(self, inputs):
        """
        Normalize the model inputs.
        Takes the dict of inputs and returns a dict of normalized inputs.

        :param inputs: model inputs
        :return: a dict of normalized model inputs
        """
        logging.warning("normalize_input() undefined. No normalization of the model inputs will be performed. "
                        "You can implement the function in your model class if you want.")
        return inputs

    def postprocess_outputs(self, outputs, inputs=None, normalized_inputs=None):
        """
        Post-process the model outputs.
        Takes the dicts of inputs and outputs, and returns a dict of post-processed outputs.
        The default implementation provides a set of cropped output tensors

        :param outputs: dict of model outputs
        :param inputs: dict of model inputs (optional)
        :param normalized_inputs: dict of normalized model inputs (optional)
        :return: a dict of post-processed model outputs
        """

        # Add extra outputs for inference
        extra_outputs = {}
        for out_key, out_tensor in outputs.items():
            for crop in self.inference_cropping:
                extra_output_key = cropped_tensor_name(out_key, crop)
                extra_output_name = cropped_tensor_name(out_tensor._keras_history.layer.name, crop)
                logging.info("Adding extra output for tensor %s with crop %s (%s)", out_key, crop, extra_output_name)
                cropped = out_tensor[:, crop:-crop, crop:-crop, :]
                identity = tensorflow.keras.layers.Activation('linear', name=extra_output_name)
                extra_outputs[extra_output_key] = identity(cropped)

        return extra_outputs

    def create_network(self):
        """
        This method returns the Keras model. This needs to be called **inside** the strategy.scope().
        Can be reimplemented depending on the needs.

        :return: the keras model
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
        postprocessed_outputs = self.postprocess_outputs(outputs=outputs, inputs=inputs,
                                                         normalized_inputs=normalized_inputs)
        outputs.update(postprocessed_outputs)

        # Return the keras model
        return tensorflow.keras.Model(inputs=inputs, outputs=outputs, name=self.__class__.__name__)

    def summary(self, strategy=None):
        """
        Wraps the summary printing of the model. When multiworker strategy, only prints if the worker is chief
        """
        if not strategy or _is_chief(strategy):
            self.model.summary(line_length=150)

    def plot(self, output_path, strategy=None):
        """
        Enables to save a figure representing the architecture of the network.
        Needs pydot and graphviz to work (`pip install pydot` and https://graphviz.gitlab.io/download/)
        """
        assert self.model, "Plot() only works if create_network() has been called beforehand"

        # When multiworker strategy, only plot if the worker is chief
        if not strategy or _is_chief(strategy):
            # Build a simplified model, without normalization nor extra outputs.
            # This model is only used for plotting the architecture thanks to `keras.utils.plot_model`
            inputs = self.get_inputs()  # inputs without normalization
            outputs = self.get_outputs(inputs)  # raw model outputs
            model_simplified = tensorflow.keras.Model(inputs=inputs, outputs=outputs,
                                                      name=self.__class__.__name__ + '_simplified')
            tensorflow.keras.utils.plot_model(model_simplified, output_path)
