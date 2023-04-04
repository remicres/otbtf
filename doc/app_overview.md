# Applications overview

## Applications in OTB

In OTB, applications are processes working on geospatial images, with a 
standardized interface. This interface enables the applications to be fully 
interoperable, and operated from various ways: C++, python, command line 
interface. The cool thing is that most of the applications support the 
so-called *streaming* mechanism that enable to process very large images with 
a limited memory footprint. Thanks to the interface shared by the OTB
applications, we can use them as functional bricks to build large pipelines, 
that are memory and computationally efficient.

!!! Info

    As any OTB application, the new applications provided by OTBTF can be used 
    in command line interface, C++, or python.
    For the best experience in python, we recommend to use OTB applications 
    using the excellent 
    [PyOTB](https://pyotb.readthedocs.io/en/master/quickstart.html).

## New applications

Here are the new applications provided by OTBTF.

- **TensorflowModelServe**: Inference on real world remote sensing products
- **PatchesExtraction**: extract patches in images
- **PatchesSelection**: patches selection from rasters
- **LabelImageSampleSelection**: select patches from a label image
- **DensePolygonClassStatistics**: fast terrain truth polygons statistics
- **TensorflowModelTrain**: training/validation (educational purpose)
- **TrainClassifierFromDeepFeatures**: train traditional classifiers that use
  features from deep nets (educational/experimental)
- **ImageClassifierFromDeepFeatures**: use traditional classifiers with
  features from deep nets (educational/experimental)

Typically, you could build a pipeline like that without coding a single 
image process, only by using existing OTB applications, and bringing your own 
Tensorflow model inside (with the `TensorflowModelServe` application). 

![Schema](images/pipeline.png)

The entire pipeline would be fully streamable, with a minimal memory footprint.
Also, it should be noted that most OTB applications are multithreaded and 
benefit from multiple cores. Read more about streaming in OTB 
[here](https://www.orfeo-toolbox.org/CookBook/C++/StreamingAndThreading.html).