# OTBTF: Orfeo ToolBox meets TensorFlow

<p align="center">
<img src="images/logo.png" width="160px">
<br>
<a href="https://gitlab.irstea.fr/remi.cresson/otbtf/-/releases">
<img src="https://gitlab.irstea.fr/remi.cresson/otbtf/-/badges/release.svg">
</a>
<a href="https://gitlab.irstea.fr/remi.cresson/otbtf/-/commits/master">
<img src="https://gitlab.irstea.fr/remi.cresson/otbtf/badges/master/pipeline.svg">
</a>
<a href="LICENSE">
<img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
</a>
</p>


This remote module of the [Orfeo ToolBox](https://www.orfeo-toolbox.org) 
provides a generic, multi-purpose deep learning framework, targeting remote 
sensing images processing. It contains a set of new process objects for OTB 
that internally invoke [Tensorflow](https://www.tensorflow.org/), and new [OTB 
applications](#otb-applications) to perform deep learning with real-world 
remote sensing images. Applications can be used to build OTB pipelines from 
Python or C++ APIs. OTBTF also includes a [python API](#python-api) to build 
Keras compliant models, easy to train in distributed environments. 


## Features

### OTB Applications

- Sample patches in remote sensing images with `PatchesExtraction`,
- Inference with support of OTB streaming mechanism with 
`TensorflowModelServe`: this means that inference is not limited by images 
number, size, of channels depths, and can be used as a "lego" in any pipeline 
composed of OTB applications and preserving streaming.
- Model training, supporting save/restore/import operations (a model can be 
trained from scratch or fine-tuned) with `TensorflowModelTrain`. This 
application targets mostly newcomers and is nice for educational purpose, but 
deep learning practitioners will for sure prefer the Python API of OTBTF.  

### Python API

The `otbtf` module targets python developers that want to train their own 
model from python with TensorFlow or Keras.
It provides various classes for datasets and iterators to handle the 
_patches images_ generated from the `PatchesExtraction` OTB application.
For instance, the `otbtf.DatasetFromPatchesImages` can be instantiated from a 
set of _patches images_ and delivering samples as `tf.dataset` that can be 
used in your favorite TensorFlow pipelines, or convert your patches into 
TFRecords. The `otbtf.TFRecords` enables you train networks from TFRecords 
files, which is quite suited for distributed training. Read more in the 
[tutorial for keras](#api_tutorial.html).

## Examples

Below are some screen captures of deep learning applications performed at 
large scale with OTBTF.

 - Landcover mapping (Spot-7 images --> Building map using semantic 
segmentation)

![Landcover mapping](https://gitlab.irstea.fr/remi.cresson/otbtf/-/raw/develop/doc/images/landcover.png)

 - Super resolution (Sentinel-2 images upsampled with the 
[SR4RS software](https://github.com/remicres/sr4rs), which is based on OTBTF)
 
![Super resolution](https://gitlab.irstea.fr/remi.cresson/otbtf/-/raw/develop/doc/images/supresol.png)

 - Sentinel-2 reconstruction with Sentinel-1 VV/VH with the 
[Decloud software](https://github.com/CNES/decloud), which is based on OTBTF

![Decloud](https://github.com/CNES/decloud/raw/master/doc/images/cap2.jpg)
 
 - Image to image translation (Spot-7 image --> Wikimedia Map using CGAN. 
So unnecessary but fun!)

![Pix2pix](https://gitlab.irstea.fr/remi.cresson/otbtf/-/raw/develop/doc/images/pix2pix.png)

## Contribute

Every one can **contribute** to OTBTF. Just open a PR :)

## Cite

```
@article{cresson2018framework,
  title={A framework for remote sensing images processing using deep learning techniques},
  author={Cresson, R{\'e}mi},
  journal={IEEE Geoscience and Remote Sensing Letters},
  volume={16},
  number={1},
  pages={25--29},
  year={2018},
  publisher={IEEE}
}
```

## Additional resources

- The [*test* folder](https://github.com/remicres/otbtf/tree/master/test/) 
of this repository contains various use-cases with commands, python codes, and 
input/baseline data,
- This [book](https://doi.org/10.1201/9781003020851) contains 130 pages to 
learn how to use OTBTF with OTB and QGIS to perform various kind of deep 
learning sorcery on remote sensing images (patch-based classification for 
landcover mapping, semantic segmentation of buildings, optical image 
restoration from joint SAR/Optical time series): *Cresson, R. (2020). Deep 
Learning for Remote Sensing Images with Open Source Software. CRC Press.*
- A small [tutorial](https://mdl4eo.irstea.fr/2019/01/04/an-introduction-to-deep-learning-on-remote-sensing-images-tutorial/) on MDL4EO's blog
- Check [our repository](https://github.com/remicres/otbtf_tutorials_resources) 
containing stuff (data and models) to begin with!
