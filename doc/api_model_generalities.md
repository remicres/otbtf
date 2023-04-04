# Model generalities

This section gives a few tips to create your own models ready to be used in 
OTBTF.

## Inputs dimensions

All networks must input **4D tensors**.

- **dim 0** is for the batch dimension. It is used in the 
`TensorflowModelTrain` application during training, and in 
**patch-based mode** during inference: in this mode, `TensorflowModelServe` 
performs the inference of several patches simultaneously. In 
**fully-convolutional mode**, a single slice of the batch dimension is used.
- **dim 1** and **2** are for the spatial dimensions,
- **dim 3** is for the image channels. Even if your image have only 1 channel, 
you must set a shape value equals to 1 for the last dimension of the input 
placeholder.

## Inputs shapes

For networks intended to work in **patch-based** mode, you can stick with a 
placeholder having a patch size explicitly defined in **dims 1 and 2**.
However, for networks intended to work in **fully-convolutional** mode, 
you must set `None` in **dim 1** and **dim 2** (before Tensorflow 2.X, it was 
possible to feed placeholders with a tensor of different size where the dims 
were defined). For instance, let consider an input raster with 4 spectral 
bands: the input shape of the model input would be like `[None, None, None, 4]` 
to work in fully-convolutional mode. By doing so, the use of input images of 
any size is enabled (`TensorflowModelServe` will automatically compute the 
input/output regions sizes to process, given the **receptive field** and 
**expression field** of your network).

## Outputs dimensions

Supported tensors for the outputs must have **between 2 and 4 dimensions**.
OTBTF always consider that **the size of the last dimension is the number of 
channels in the output**.
For instance, you can have a model that outputs 8 channels with a tensor of 
shape `[None, 8]` or `[None, None, None, 8]`

## Outputs names

Always name explicitly your models outputs. You will need the output tensor 
name for performing the inference with `TensoflowModelServe`. If you forget to 
name them, use the graph viewer in `tensorboard` to get the names.

!!! note

    If you want to enable your network training with the `TensorflowModelTrain` 
    application, you can use the Tensorflow API v1. In this case, do not forget
    to name your optimizers/operators. You can build a single operator from 
    multiple ones using the `tf.group` command, which also enable you to name 
    your new operator. For sequential nodes trigger (e.g. GANs), you can build 
    an operator that do what you want is the desired order using  
    `tf.control_dependancies()`. 

