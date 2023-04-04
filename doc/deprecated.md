!!! Warning

    The `tricks` module is deprecated since OTBTF 2.0

The Tensorflow python API has changed significantly (for the best) after the
Tensorflow 2.0 release. OTBTF used to provide the `tricks` module, providing 
useful methods to generate the SavedModels, or convert checkpoints into 
SavedModels.

[Source code :fontawesome-brands-github:](https://github.com/remicres/otbtf/tree/master/otbtf/tricks/){ .md-button }

## What is best in Tensorflow 2.X?

- Shorter and simpler code
- Easy to build and train a model with Keras, which has become the principal 
interface for Tensorflow
- More hardware-agnostic than ever: with the exact same code, you can run on 
a single-cpu, GPU, or a pool of GPU servers.

## Major changes between Tensorflow 1 and Tensorflow 2 APIs

Here are a few tips and tricks for people that want to move from 
Tensorflow 1 to Tensorflow 2 API.
Models built for OTBTF have to take in account the following changes:

- Models built with `otbtf.ModelBase` or `tensorflow.keras.model.Model` have 
no longer to use `tensorflow.compat.v1.placeholder` but 
`tensorflow.keras.Input` instead,
- Tensorflow variable scopes are no longer used when the training is done from 
Keras,
- SavedModel can be created directly from the model instance! (as simple as 
`mymodel.save("mymodel_v1_savedmode")`),
- Switching between single cpu/gpu or multiple computing nodes, distributed 
training, etc. is done using the so-called `tensorflow.Strategy`

!!! Note

    Read our [tutorial](#api_tutorial.html) to know more on working with Keras!