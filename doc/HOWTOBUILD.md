# How to build OTBTF from sources

This remote module has been tested successfully on Ubuntu 18 with last CUDA drivers, TensorFlow r2.1 and OTB 7.1.0.

## Build OTB
First, **build the *release-7.1* branch of OTB from sources**. You can check the [OTB documentation](https://www.orfeo-toolbox.org/SoftwareGuide/SoftwareGuidech2.html) which details all the steps. It is quite easy thank to the SuperBuild, a cmake script that automates the build.

Create a folder for OTB, clone sources, configure OTB SuperBuild, and build it.

Install required packages:

```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install sudo ca-certificates curl make cmake g++ gcc git libtool swig xvfb wget autoconf automake pkg-config zip zlib1g-dev unzip freeglut3-dev libboost-date-time-dev libboost-filesystem-dev libboost-graph-dev libboost-program-options-dev libboost-system-dev libboost-thread-dev libcurl4-gnutls-dev libexpat1-dev libfftw3-dev libgdal-dev libgeotiff-dev libglew-dev libglfw3-dev libgsl-dev libinsighttoolkit4-dev libkml-dev libmuparser-dev libmuparserx-dev libopencv-core-dev libopencv-ml-dev libopenthreads-dev libossim-dev libpng-dev libqt5opengl5-dev libqwt-qt5-dev libsvm-dev libtinyxml-dev qtbase5-dev qttools5-dev default-jdk python3-pip python3.6-dev python3.6-gdal python3-setuptools libxmu-dev libxi-dev qttools5-dev-tools bison software-properties-common dirmngr apt-transport-https lsb-release gdal-bin
```

Build OTB from sources:

```
sudo mkdir /work
sudo chown $USER /work
mkdir /work/otb
cd /work/otb
mkdir build
git clone -b release-7.1 https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb.git OTB
cd build
```

From here you can tell the interactively SuperBuild to use system boost, curl, zlib, libkml for instance.

```
ccmake /work/otb/OTB/SuperBuild
```

If you don't know how to configure options, you can use the following:

```
cmake /work/otb/OTB/SuperBuild -DUSE_SYSTEM_BOOST=ON -DUSE_SYSTEM_CURL=ON -DUSE_SYSTEM_EXPAT=ON -DUSE_SYSTEM_FFTW=ON -DUSE_SYSTEM_FREETYPE=ON -DUSE_SYSTEM_GDAL=ON -DUSE_SYSTEM_GEOS=ON -DUSE_SYSTEM_GEOTIFF=ON -DUSE_SYSTEM_GLEW=ON -DUSE_SYSTEM_GLFW=ON -DUSE_SYSTEM_GLUT=ON -DUSE_SYSTEM_GSL=ON -DUSE_SYSTEM_ITK=ON -DUSE_SYSTEM_LIBKML=ON -DUSE_SYSTEM_LIBSVM=ON -DUSE_SYSTEM_MUPARSER=ON -DUSE_SYSTEM_MUPARSERX=ON -DUSE_SYSTEM_OPENCV=ON -DUSE_SYSTEM_OPENTHREADS=ON -DUSE_SYSTEM_OSSIM=ON -DUSE_SYSTEM_PNG=ON -DUSE_SYSTEM_QT5=ON -DUSE_SYSTEM_QWT=ON -DUSE_SYSTEM_TINYXML=ON -DUSE_SYSTEM_ZLIB=ON -DUSE_SYSTEM_SWIG=OFF -DOTB_WRAP_PYTHON=OFF
```

Then you can build OTB:
```
make -j $(grep -c ^processor /proc/cpuinfo)
```

## Build TensorFlow with shared libraries
During this step, you have to **build Tensorflow from source** except if you want to use only the sampling applications of OTBTensorflow (in this case, skip this section).

### Bazel
First, install Bazel.
```
wget https://github.com/bazelbuild/bazel/releases/download/0.29.1/bazel-0.29.1-installer-linux-x86_64.sh
chmod +x bazel-0.29.1-installer-linux-x86_64.sh
./bazel-0.29.1-installer-linux-x86_64.sh --user
export PATH="$PATH:$HOME/bin"
```

If you fail to install properly Bazel, you can read the beginning of [the instructions](https://www.tensorflow.org/install/install_sources) that present alternative methods for this.

### Required packages
There is a few required packages that you need to install:
```
sudo python3 -m pip install --upgrade pip
sudo python3 -m pip install pip six numpy wheel mock keras future setuptools
```

For a pure python3 install, you might need to workaround a bazel bug the following way:
```
sudo ln -s /usr/bin/python3 /usr/bin/python
```

### Build TensorFlow
Create a directory for TensorFlow.
For instance `mkdir /work/tf`.

Clone TensorFlow.
```
cd /work/tf
git clone https://github.com/tensorflow/tensorflow.git
```

Now configure the project. If you have CUDA and other NVIDIA stuff installed in your system, remember that you have to tell the script that it is in `/usr/` (no symlink required!). If you have CPU-only hardware, building Intel MKL is a good choice since it provides a significant speedup in computations.

```
cd tensorflow
./configure
```

Then, you have to build TensorFlow with the instructions sets supported by your CPU (For instance here is AVX, AVX2, FMA, SSE4.1, SSE4.2 that play fine on a modern intel CPU). You have to tell Bazel to build:

 1. The TensorFlow python pip package
 2. The libtensorflow_cc.so library
 3. The libtensorflow_framework.so library

```
bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.1 --copt=-msse4.2 //tensorflow:libtensorflow_framework.so //tensorflow:libtensorflow_cc.so //tensorflow:libtensorflow.so //tensorflow/tools/pip_package:build_pip_package --noincompatible_do_not_split_linking_cmdline
```

*You might fail this step (e.g. missing packages). In this case, it's recommended to clear the bazel cache, using something like `rm $HOME/.cache/bazel/* -rf` before configuring and building everything!*

### Pip package
Build and deploy the pip package.

```
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip3 install $(find /tmp/tensorflow_pkg/ -type f -iname "tensorflow*.whl")
```

### C++ API
First, download and build TensorFlow dependencies.

```
/work/tf/tensorflow/tensorflow/lite/tools/make/download_dependencies.sh
/work/tf/tensorflow/tensorflow/lite/tools/make/build_lib.sh
```

Then, build Google Protobuf

```
mkdir -p /work/tf/installdir
cd /work/tf/
wget https://github.com/google/protobuf/releases/download/v3.8.0/protobuf-cpp-3.8.0.tar.gz
tar -xvf protobuf-cpp-3.8.0.tar.gz
cd protobuf-3.8.0
./configure --prefix=/work/tf/installdir/
make install -j $(grep -c ^processor /proc/cpuinfo)
```

Then, prepare a folder with everything (include, libs)

```
mkdir -p /work/tf/installdir/lib
mkdir -p /work/tf/installdir/include
cp bazel-bin/tensorflow/libtensorflow_cc.so* /work/tf/installdir/lib
cp bazel-bin/tensorflow/libtensorflow_framework.so* /work/tf/installdir/lib
cp -r bazel-genfiles/* /work/tf/installdir/include
cp -r tensorflow/cc /work/tf/installdir/include/tensorflow
cp -r tensorflow/core /work/tf/installdir/include/tensorflow
cp -r third_party /work/tf/installdir/include
cp -r bazel-tensorflow/external/eigen_archive/unsupported /work/tf/installdir/include
cp -r bazel-tensorflow/external/eigen_archive/Eigen /work/tf/installdir/include
cp -r tensorflow/lite/tools/make/downloads/absl/absl /work/tf/installdir/include
```

Now you have a working copy of TensorFlow located in `/work/tf/installdir` that is ready to use in external C++ cmake projects :)

## Build the OTBTF remote module
Finally, we can build the OTBTF module.
Clone the repository inside the OTB sources directory for remote modules: `/work/otb/OTB/Modules/Remote/`.
Re configure OTB with cmake of ccmake, and set the following variables

 - **Module_OTBTensorflow** to **ON**
 - **OTB_USE_TENSORFLOW** to **ON** (if you set to OFF, you will have only the sampling applications)
 - **TENSORFLOW_CC_LIB** to `/work/tf/installdir/lib/libtensorflow_cc.so`
 - **TENSORFLOW_FRAMEWORK_LIB** to `/work/tf/installdir/lib/libtensorflow_framework.so`
 - **tensorflow_include_dir** to `/work/tf/installdir/include`

Re build and re install OTB.

```
cd /work/otb/build/OTB/build
ccmake
make -j $(grep -c ^processor /proc/cpuinfo)
make install
```
Done !

Don't forget to add some important environment variables, and this is finished.

```
export PATH="$PATH:/work/otb/superbuild_install/bin/"
export PYTHONPATH="$PYTHONPATH:/work/otb/superbuild_install/lib/otb/python:/work/otb/otb/Modules/Remote/otbtf/python"
export OTB_APPLICATION_PATH="$OTB_APPLICATION_PATH:/work/otb/superbuild_install/lib/otb/applications"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/work/otb/superbuild_install/lib/:/work/tf/installdir/lib/"
```

Check that the applications run properly from command line.

```
otbcli_TensorflowModelServe --help
```

The following output should be displayed:

```
Multisource deep learning classifier using TensorFlow. Change the OTB_TF_NSOURCES environment variable to set the number of sources.
Parameters: 
        -source1                <group>          Parameters for source #1 
MISSING -source1.il             <string list>    Input image (or list to stack) for source #1  (mandatory)
MISSING -source1.rfieldx        <int32>          Input receptive field (width) for source #1  (mandatory)
MISSING -source1.rfieldy        <int32>          Input receptive field (height) for source #1  (mandatory)
MISSING -source1.placeholder    <string>         Name of the input placeholder for source #1  (mandatory)
        -model                  <group>          model parameters 
MISSING -model.dir              <string>         TensorFlow model_save directory  (mandatory)
        -model.userplaceholders <string list>    Additional single-valued placeholders. Supported types: int, float, bool.  (optional, off by default)
        -model.fullyconv        <boolean>        Fully convolutional  (optional, off by default, default value is false)
        -output                 <group>          Output tensors parameters 
        -output.spcscale        <float>          The output spacing scale, related to the first input  (mandatory, default value is 1)
MISSING -output.names           <string list>    Names of the output tensors  (mandatory)
        -output.efieldx         <int32>          The output expression field (width)  (mandatory, default value is 1)
        -output.efieldy         <int32>          The output expression field (height)  (mandatory, default value is 1)
        -optim                  <group>          This group of parameters allows optimization of processing time 
        -optim.disabletiling    <boolean>        Disable tiling  (optional, off by default, default value is false)
        -optim.tilesizex        <int32>          Tile width used to stream the filter output  (mandatory, default value is 16)
        -optim.tilesizey        <int32>          Tile height used to stream the filter output  (mandatory, default value is 16)
MISSING -out                    <string> [pixel] output image  [pixel=uint8/uint16/int16/uint32/int32/float/double/cint16/cint32/cfloat/cdouble] (default value is float) (mandatory)
        -inxml                  <string>         Load otb application from xml file  (optional, off by default)
        -progress               <boolean>        Report progress 
        -help                   <string list>    Display long help (empty list), or help for given parameters keys

Use -help param1 [... paramN] to see detailed documentation of those parameters.

Examples: 
otbcli_TensorflowModelServe -source1.il spot6pms.tif -source1.placeholder x1 -source1.rfieldx 16 -source1.rfieldy 16 -model.dir /tmp/my_saved_model/ -model.userplaceholders is_training=false dropout=0.0 -output.names out_predict1 out_proba1 -out "classif128tgt.tif?&streaming:type=tiled&streaming:sizemode=height&streaming:sizevalue=256"
```

