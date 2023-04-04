!!! Warning

	This section is for educational purposes. No coding skills are required,
	and it's easy to train an existing model built with the Tensorflow API
	v1. To have a full control over the model implementation and training 
	process, the Tensorflow API v2 with Keras is the way to go.
    If you are interested in more similar examples, please read the 
    [Tensorflow v1 models examples](reference/otbtf/examples/tensorflow_v1x/__init__.md)
	
## TensorflowModelTrain

Here we assume that you have produced patches using the `PatchesExtraction`
application, and that you have a **SavedModel** stored in a directory somewhere
on your filesystem.
The `TensorflowModelTrain` application performs the training, validation (
against test dataset, and against validation dataset) providing the usual
metrics that machine learning frameworks provide (confusion matrix, recall,
precision, f-score, ...).
You must provide the path of the **SavedModel** to the `model.dir` parameter.
The `model.restorefrom` and `model.saveto` corresponds to the variables of the
**SavedModel** used respectively for restoring and saving them.
Set you _input sources_ for training (`training` parameter group) and for
validation (`validation` parameter group): the evaluation is performed against
training data, and optionally also against the validation data (only if you
set `validation.mode` to "class").
For each _input sources_, the patch size and the placeholder name must be
provided.
Regarding validation, if a different name is found in a particular _input
source_ of the `validation` parameter group, the application knows that the
_input source_ is not fed to the model at inference, but is used as reference
to compute evaluation metrics of the validation dataset.
Batch size (`training.batchsize`) and number of epochs (`training.epochs`) can
be set.
_User placeholders_ can be set separately for
training (`training.userplaceholders`) and
validation (`validation.userplaceholders`).
The `validation.userplaceholders` can be useful if you have a model that
behaves differently depending on the given placeholder.
Let's take the example of dropout: it's nice for training, but you have to
disable it to use the model at inference time.
Hence you will pass a placeholder with "dropout\_rate=0.3" for training and "
dropout\_rate=0.0" for validation.
Of course, one can train models from handmade python code: to import the
patches images, a convenient method consist in reading patches images as numpy
arrays using OTB applications (e.g. `ExtractROI`) or GDAL, then do a
`numpy.reshape` to the dimensions wanted.

![Schema](https://gitlab.irstea.fr/remi.cresson/otbtf/-/raw/develop/doc/images/model_training.png)

The application description can be displayed using:

```commandLine
otbcli_TensorflowModelTrain --help
```

As you can note, there is `$OTB_TF_NSOURCES` + 1 sources because we often need
at least one more source for the reference data (e.g. terrain truth for land
cover mapping).

## Composite applications for classification

Who has never dreamed to use classic classifiers performing on deep learning
features?
This is possible thank to two new applications that uses the existing
training/classification applications of OTB:

`TrainClassifierFromDeepFeatures` is a composite application that wire 
`TensorflowModelServe` application output into the existing official 
`TrainImagesClassifier` application.

The application description can be displayed using:

```commandLine
otbcli_TrainClassifierFromDeepFeatures --help
```

## Example

We assume that we have already followed the 
[*sampling* section](app_sampling.html). We start from the files generated at 
the end of the patches extraction.

Now we have two images for patches and labels.
We can split them to distinguish test/validation groups (with the `ExtractROI`
 application for instance).
But here, we will just perform some fine-tuning of our model.
The **SavedModel** is located in the `outmodel` directory.
Our model is quite basic: it has two input placeholders, **x1** and **y1**
respectively for input patches (with size 16x16) and input reference labels (
with size 1x1).
We named **prediction** the tensor that predict the labels and the optimizer
that perform the stochastic gradient descent is an operator named **optimizer
**.
We perform the fine-tuning and we export the new model variables directly in
the _outmodel/variables_ folder, overwriting the existing variables of the
model.
We use the `TensorflowModelTrain` application to perform the training of this
existing model.

```
otbcli_TensorflowModelTrain -model.dir /path/to/oursavedmodel \
-training.targetnodesnames optimizer -training.source1.il samp_patches.tif \
-training.source1.patchsizex 16 -training.source1.patchsizey 16 \
-training.source1.placeholder x1 -training.source2.il samp_labels.tif \
-training.source2.patchsizex 1 -training.source2.patchsizey 1 \
-training.source2.placeholder y1 \
-model.saveto /path/to/oursavedmodel/variables/variables
```

Note that we could also have performed validation in this step. In this case,
the `validation.source2.placeholder` would be different than
the `training.source2.placeholder`, and would be **prediction**. This way, the
program know what is the target tensor to evaluate.

