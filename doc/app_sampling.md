# Sampling applications

OTBTF sampling applications are OTB applications that focus on the extraction 
of samples in the remote sensing images.

Main OTBTF applications for sampling are:

- [`PatchesSelection`](#patchesselection)
- [`PatchesExtraction`](#patchesextraction)
 
Other applications were written for experimental and educational purposes, but 
could still fill some needs sometimes:

- [`DensePolygonClassStatistics`](#densepolygonclassstatistics) 
- [`LabelImageSampleSelection`](#labelimagesampleselection)

## PatchesSelection

This application generate points sampled at regular interval over the input 
image region. The selection strategy, patches grid size and step can be 
configured. The application produces a vector data containing a set of points 
centered on the patches after the selection process. 

The following strategies are implemented:

- Split: the classic training/validation/testing samples split,
- Chessboard: training/validation over the patches grid in a chessboard 
fashion
- All: all patches are selected
- Balanced: using an additional terrain truth labels map to select patches 
a random locations that try to balance the patches population distribution, 
based on the class value.

The application description can be displayed using:

```commandline
otbcli_PatchesSelection --help
```

## PatchesExtraction

The `PatchesExtraction` application performs the extraction of patches in
images from the following:

- a vector data containing points (mandatory)
- at least one imagery source (mandatory). To change the number of sources, 
set the environment variable `OTB_TF_NSOURCES`
- One exiting field name of the vector data to identify the different points.

Each point of the vector data locates the **center** of the **central pixel** 
of one patch.
For each source, the following parameters can be set:

- the patch size (x and y): for patches with even size *N*, the **central 
pixel** corresponds to the pixel index *N/2+1* (index starting at 0).
- a no-data value: If any pixel value inside the patch is equal to the 
provided value, the patch is rejected.
- an output file name for the *patches image* that the application exports at 
the end of the sampling. Patches are stacked in rows and exported as common 
raster files supported by GDAL, without any geographical information.

### Example with 2 sources

We denote one _input source_, either an input image, or a stack of input images
that will be concatenated (they must have the same size).
The user can set the `OTB_TF_NSOURCES` environment variable to select the
number of _input sources_ that he wants.
For example, for sampling a Time Series (TS) together with a single Very High
Resolution image (VHR), two sources are required:

- 1 input images list for time series,
- 1 input image for the VHR.

The sampled patches are extracted at each position designed by the input
vector data, only if a patch lies fully in all _input sources_ extents.
For each _input source_, patches sizes must be provided.
For each _input source_, the application export all sampled patches as a single
multiband raster, stacked in rows.
For instance, for *n* samples of size *16 x 16* from a *4* channels _input
source_, the output image will be a raster of size *16 x 16n* with *4*
channels.
An optional output is an image of size *1 x n* containing the value of one
specific field of the input vector data.
Typically, the *class* field can be used to generate a dataset suitable for a
model that performs pixel wise classification.

![Schema](https://gitlab.irstea.fr/remi.cresson/otbtf/-/raw/develop/doc/images/patches_extraction.png)

The application description can be displayed using:

```commandline
otbcli_PatchesExtraction --help
```


## DensePolygonClassStatistics

This application is a clone of the [`PolygonClassStatistics`](https://www.orfeo-toolbox.org/CookBook/Applications/app_PolygonClassStatistics.html)
application from OTB modified to use rasterization instead of vector based 
approach, making it faster.

The application description can be displayed using:

```commandline
otbcli_DensePolygonClassStatistics --help
```

## LabelImageSampleSelection

This application extracts points from an input label image. This application 
is like `SampleSelection`, but uses an input label image, rather than an input 
vector data. It produces a vector data containing a set of points centered on 
the pixels of the input label image. The user can control the number of 
points. The default strategy consists in producing the same number of points 
in each class. If one class has a smaller number of points than requested, 
this one is adjusted.

The application description can be displayed using:

```commandline
otbcli_LabelImageSampleSelection --help
```

## Example

Below is a minimal example that presents some steps to sample patches from a 
sparse annotated vector data as terrain truth.
Let's consider that our data set consists in one Spot-7 image, *spot7.tif*, 
and a training vector data, *terrain_truth.shp* that describes sparsely 
forest / non-forest polygons.

First, we compute statistics of the vector data : how many points can we sample
inside objects, and how many objects in each class.
We use the `PolygonClassStatistics` application of OTB.

```commandline
otbcli_PolygonClassStatistics -vec terrain_truth.shp -field class \
-in spot7.tif -out vec_stats.xml
```

Then, we will select some samples with the `SampleSelection` application of
the existing machine learning framework of OTB.
Since the terrain truth is sparse, we want to sample randomly points in
polygons with the default strategy of the `SampleSelection` OTB application.

```
otbcli_SampleSelection -in spot7.tif -vec terrain_truth.shp \
-instats vec_stats.xml -field class -out points.shp
```

Now we extract the patches with the `PatchesExtraction` application.
We want to produce one image of 16x16 patches, and one image for the
corresponding labels.

```
otbcli_PatchesExtraction -source1.il spot7.tif \
-source1.patchsizex 16 -source1.patchsizey 16 \
-vec points.shp -field class -source1.out samp_labels.tif \
-outpatches samp_patches.tif
```

Now we can use the generated *samp_patches.tif* and *samp_labels.tif* in the 
`TensorflowModelTrain` application, or using the python API to build and train 
models with Keras.