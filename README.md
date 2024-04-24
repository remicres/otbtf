# OTBTF: Orfeo ToolBox meets TensorFlow

<p align="center">
<img src="doc/images/logo.png" width="160px">
<br>
<a href="https://forgemia.inra.fr/orfeo-toolbox/otbtf/-/releases">
<img src="https://forgemia.inra.fr/orfeo-toolbox/otbtf/-/badges/release.svg">
</a>
<a href="https://forgemia.inra.fr/orfeo-toolbox/otbtf/-/commits/master">
<img src="https://forgemia.inra.fr/orfeo-toolbox/otbtf/badges/master/pipeline.svg">
</a>
<img src='https://readthedocs.org/projects/otbtf/badge/?version=latest' alt='Documentation Status' />
<a href="LICENSE">
<img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
</a>
<img src="https://img.shields.io/badge/dynamic/json?formatter=metric&color=blue&label=Docker%20pull&query=%24.pull_count&url=https://hub.docker.com/v2/repositories/mdl4eo/otbtf">
</p>

OTBTF is a remote module of the [Orfeo ToolBox](https://www.orfeo-toolbox.org). 
It provides a generic, multi-purpose deep learning framework, targeting remote 
sensing images processing. It contains a set of new process objects for OTB 
that internally invoke [Tensorflow](https://www.tensorflow.org/), and new OTB 
applications to perform deep learning with real-world remote sensing images. 
Applications can be used to build OTB pipelines from Python or C++ APIs. OTBTF 
also includes a python API to build quickly Keras compliant models suited for 
remote sensing imagery, easy to train in distributed environments. 

## Documentation

The documentation is available on [otbtf.readthedocs.io](https://otbtf.readthedocs.io).

## Use

You can use our latest GPU enabled docker images.

```bash
docker run --runtime=nvidia -ti mdl4eo/otbtf:latest-gpu otbcli_PatchesExtraction
docker run --runtime=nvidia -ti mdl4eo/otbtf:latest-gpu python -c "import otbtf"
```

You can also build OTBTF from sources (see the documentation)

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
