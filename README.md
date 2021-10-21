![OTBTF](doc/images/logo.png)

# OTBTF: Orfeo ToolBox meets TensorFlow

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This remote module of the [Orfeo ToolBox](https://www.orfeo-toolbox.org) provides a generic, multi purpose deep learning framework, targeting remote sensing images processing.
It contains a set of new process objects that internally invoke [Tensorflow](https://www.tensorflow.org/), and a bunch of user-oriented applications to perform deep learning with real-world remote sensing images.
Applications can be used to build OTB pipelines from Python or C++ APIs. 

## Features

### OTB Applications

- Sample patches in remote sensing images with `PatchesExtraction`,
- Model training, supporting save/restore/import operations (a model can be trained from scratch or fine-tuned) with `TensorflowModelTrain`,
- Inference with support of OTB streaming mechanism with `TensorflowModelServe`. The streaming mechanism means (1) no limitation with images sizes, (2) inference can be used as a "lego" in any OTB pipeline (using C++ or Python APIs) and preserving streaming, (3) MPI support available (use multiple processing unit to generate one single output image)

### Python

This is a work in progress. For now, `tricks.py` provides a set of helpers to build deep nets, and `otbtf.py` provides datasets which can be used in Tensorflow pipelines to train networks from python.

## Portfolio

Below are some screen captures of deep learning applications performed at large scale with OTBTF.
 - Image to image translation (Spot-7 image --> Wikimedia Map using CGAN)
<img src ="doc/images/pix2pix.png" />

 - Landcover mapping (Spot-7 images --> Building map using semantic segmentation)
<img src ="doc/images/landcover.png" />

 - Image enhancement (Enhancement of Sentinel-2 images at 1.5m  using SRGAN)
<img src ="doc/images/supresol.png" />

You can read more details about these applications on [this blog](https://mdl4eo.irstea.fr/2019/)

## How to install

For now you have two options: either use the existing **docker image**, or build everything **from source**.

### Docker

Use the latest image from dockerhub:
```
docker pull mdl4eo/otbtf2.5:cpu
docker run -u otbuser -v $(pwd):/home/otbuser mdl4eo/otbtf2.5:cpu otbcli_PatchesExtraction -help
```

Read more in the [docker use documentation](doc/DOCKERUSE.md).

### Build from sources

Read more in the [build from sources documentation](doc/HOWTOBUILD.md).

## How to use

- Reading [the applications documentation](doc/APPLICATIONS.md) will help, of course 😉
- A small [tutorial](https://mdl4eo.irstea.fr/2019/01/04/an-introduction-to-deep-learning-on-remote-sensing-images-tutorial/) on MDL4EO's blog
- in the `python` folder are provided some [ready-to-use deep networks, with documentation and scientific references](doc/EXAMPLES.md).
- A book: *Cresson, R. (2020). Deep Learning for Remote Sensing Images with Open Source Software. CRC Press.* Use QGIS, OTB and Tensorflow to perform various kind of deep learning sorcery on remote sensing images (patch-based classification for landcover mapping, semantic segmentation of buildings, optical image restoration from joint SAR/Optical time series).
- Check [our repository](https://github.com/remicres/otbtf_tutorials_resources) containing stuff (data and models) to begin with with!

## Contribute

Every one can **contribute** to OTBTF! Don't be shy.

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
