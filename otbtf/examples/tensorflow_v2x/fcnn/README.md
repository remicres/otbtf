This example show how to train a small fully convolutional model using the
OTBTF python API. In particular, the example show how a model can be trained
(1) from **patches-images**, or (2) from **TFRecords** files.

# Files

- `fcnn_model.py` implements a small fully convolutional U-Net like model,
with the preprocessing and normalization functions
- `train_from_patches-images.py` shows how to train the model from a list of
patches-images
- `train_from_tfrecords.py` shows how to train the model from TFRecords files
- `create_tfrecords.py` shows how to convert patch-images into TFRecords files
- `helper.py` contains a few helping functions 
- 
# Patches-images vs TFRecords based datasets

TensorFlow datasets are the most practical way to feed a network data during 
training steps.
In particular, they are very useful to train models with data parallelism using
multiple workers (i.e. multiple GPU devices).
Since OTBTF 3, two kind of approaches are available to deliver the patches:
- Create TF datasets from **patches-images**: the first approach implemented in 
OTBTF, relying on geospatial raster formats supported by GDAL. Patches are simply 
stacked in rows. patches-images are friendly because they can be visualized 
like any other image. However this approach is **not very optimized**, since it
generates a lot of I/O and stresses the filesystem when iterating randomly over
patches.
- Create TF datasets from **TFRecords** files. The principle is that a number of
patches are stored in TFRecords files (google protubuf serialized data). This
approach provides the best performances, since it generates less I/Os since 
multiple patches are read simultaneously together. It is the recommended approach
to work on high end gear. It requires an additional step of converting the 
patches-images into TFRecords files.


# A quick overview

## Patches-images based datasets

**Patches-images** are generated from the `PatchesExtraction` application of OTBTF.
They consist in extracted patches stacked in rows into geospatial rasters. 
The `otbtf.DatasetFromPatchesImages` provides access to **patches-images** as a
TF dataset. It inherits from the `otbtf.Dataset` class, which can be a base class 
to develop other raster based datasets. 
The `use_streaming` option can be used to read the patches on-the-fly 
on the filesystem. However, this can cause I/O bottleneck when one training step 
is shorter that fetching one batch of data. Typically, this is very common with 
small networks trained over large amount of data using multiple GPUs, causing the 
filesystem read operation being the weak point (and the GPUs wait for the batches 
to be ready). The class offers other functionalities, for instance changing the 
iterator class with a custom one (can inherit from `otbtf.dataset.IteratorBase`) 
which is, by default, an `otbtf.dataset.RandomIterator`. This could enable to 
control how the patches are walked, from the multiple patches-images of the 
dataset.

## TFRecords batches datasets

**TFRecord** based datasets are implemented in the `otbtf.tfrecords` module.
They basically deliver patches from the TFRecords files, which can be created 
with the `to_tfrecords()` method of the `otbtf.Dataset` based classes.
Depending on the filesystem characteristics and the computational cost of one
training step, it can be good to select the number of samples per TFRecords file.
Another tweak is the shuffling: since one TFRecord file contains multiple patches, 
the way TFRecords files are accessed (sometimes, we need them to be randomly 
accessed), and the way patches are accessed (within a buffer, of size set with the 
`shuffle_buffer_size`), is crucial. 

