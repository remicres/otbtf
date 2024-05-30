# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="otbtf",
    version="4.3.1",
    author="Remi Cresson",
    author_email="remi.cresson@inrae.fr",
    description="OTBTF: Orfeo ToolBox meets TensorFlow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.irstea.fr/remi.cresson/otbtf",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    keywords=["remote sensing",
              "otb",
              "orfeotoolbox",
              "orfeo toolbox",
              "tensorflow",
              "deep learning",
              "machine learning"
              ],
    install_requires=[
        "deprecated",
        "tensorflow",
        "numpy",
        "tqdm",
    ]
)
