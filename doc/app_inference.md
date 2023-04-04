# Inference

In OTBTF, the `TensorflowModelServe` performs the inference.
The application can run models processing any kind or number of input images,
as soon as they have geographical information and can be read with GDAL, which 
is the underlying library for IO in OTB.

## Models information

Models can be built using Tensorflow/Keras.
They must be exported in **SavedModel** format.
When using a model in OTBTF for inference, the following parameters must be 
known:

- For each *input* (or *placeholder* for models built with Tensorflow API v1):
    - Name
    - Receptive field
- For each *output tensor*:
    - Name
    - Expression field
    - Scale factor

![Schema](images/schema.png)

The **scale factor** describes the physical change of spacing of the outputs,
typically introduced in the model by non unitary strides in pooling or
convolution operators.
For each output, it is expressed relatively to one single input of the model
called the *reference input source*.
Additionally, the names of the *target nodes* must be known (e.g. optimizers
for Tensorflow API v1).
Also, the names of *user placeholders*, typically scalars inputs that are
used to control some parameters of the model, must be know.
The **receptive field** corresponds to the input volume that "sees" the deep
net.
The **expression field** corresponds to the output volume that the deep net
will create.


## TensorflowModelServe

The `TensorflowModelServe` application performs the inference, it can be used
to produce an output raster with the specified tensors.
Thanks to the streaming mechanism, very large images can be produced.
The application uses the `TensorflowModelFilter` and a `StreamingFilter` to
force the streaming of output.
This last can be optionally disabled by the user, if he prefers using the
extended filenames to deal with chunk sizes.
However, it's still very useful when the application is used in other
composites applications, or just without extended filename magic.
Some models can consume a lot of memory.
In addition, the native tiling strategy of OTB consists in strips but this
might not always the best.
For Convolutional Neural Networks for instance, square tiles are more
interesting because the padding required to perform the computation of one
single strip of pixels induces to input a lot more pixels that to process the
computation of one single tile of pixels.
So, this application takes in input one or multiple _input sources_ (the number
of _input sources_ can be changed by setting the `OTB_TF_NSOURCES` to the
desired number) and produce one output of the specified tensors.
The user is responsible of giving the **receptive field** and **name** of
_input placeholders_, as well as the **expression field**, **scale factor** and
**name** of _output tensors_.
The first _input source_ (`source1.il`) corresponds to the _reference input
source_.
As explained, the **scale factor** provided for the
_output tensors_ is related to this _reference input source_.
The user can ask for multiple _output tensors_, that will be stack along the
channel dimension of the output raster.

!!! Warning

    Multiple outputs names can be provided which results in stacked tensors in 
    the output image along the channels dimension. In this case, tensors must 
    have the same size in spatial dimension: if the sizes of _output tensors_ 
    are not consistent (e.g. a different number of (x,y) elements), an 
    exception will be thrown.

!!! Warning

    If no output tensor name is specified, the application will try to grab 
    the first output tensor found in the SavedModel. This is okay with models
    having a single output (see 
    [deterministic models section](reference/otbtf/examples/tensorflow_v2x/deterministic/__init__.html)).

![Schema](images/classif_map.png)

The application description can be displayed using:

```commandline
otbcli_TensorflowModelServe --help
```

## Composite applications for classification

To use classic classifiers performing on a deep learning model features, one 
can use a traditional classifier generated from the 
`TrainClassifierFromDeepFeatures` application, in the
`ImageClassifierFromDeepFeatures` application, which implements the same 
approach with the official OTB `ImageClassifier` application.

The application description can be displayed using:

```commandline
otbcli_ImageClassifierFromDeepFeatures --help
```

!!! Note

    You can still set the `OTB_TF_NSOURCES` environment variable to change the
    number of sources.

## Example

We assume that we have already followed the 
[*training* section](app_training.html). We start from the files generated at 
the end of the training step.

After this step, we use the trained model to produce the entire map of forest
over the whole Spot-7 image.
For this, we use the `TensorflowModelServe` application to produce the *
*prediction** tensor output for the entire image.

```commandLine
otbcli_TensorflowModelServe -source1.il spot7.tif -source1.placeholder x1 \
-source1.rfieldx 16 -source1.rfieldy 16 \
-model.dir /path/to/oursavedmodel \
-output.names prediction -out map.tif uint8
```

