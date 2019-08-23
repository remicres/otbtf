# How to build OTBTF from sources

This remote module has been tested successfully on Ubuntu 18 and CentOs 7 with last CUDA drivers, TensorFlow r1.14 and OTB develop (0df44b312d64d6c3890b65d3790d4a17d0fd5f23).

## Build OTB
First, **build the latest *develop* branch of OTB from sources**. You can check the [OTB documentation](https://www.orfeo-toolbox.org/SoftwareGuide/SoftwareGuidech2.html) which details all the steps, if fact it is quite easy thank to the SuperBuild.

Basically, you have to create a folder for OTB, clone sources, configure OTB SuperBuild, and build it.
The following has been validated with an OTB 6.7.0.
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install aptitude
sudo aptitude install make cmake-curses-gui build-essential libtool automake git libbz2-dev python-dev libboost-dev libboost-filesystem-dev libboost-serialization-dev libboost-system-dev zlib1g-dev libcurl4-gnutls-dev swig libkml-dev
sudo mkdir /work
sudo chown $USER /work
mkdir /work/otb
cd /work/otb
mkdir build
git clone https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb.git OTB
cd build
ccmake /work/otb/OTB/SuperBuild
```
From here you can tell the SuperBuild to use system boost, curl, zlib, libkml for instance.

Then you can build it:
```
make -j $(grep -c ^processor /proc/cpuinfo)
```

## Build TensorFlow with shared libraries
During this step, you have to **build Tensorflow from source** except if you want to use only the sampling applications of OTBTensorflow (in this case, skip this section).
The following has been validated with TensorFlow r1.14 and gcc 5.3.1.

### Bazel
First, install Bazel.
```
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python
wget https://github.com/bazelbuild/bazel/releases/download/0.20.0/bazel-0.20.0-installer-linux-x86_64.sh
chmod +x bazel-0.20.0-installer-linux-x86_64.sh
./bazel-0.20.0-installer-linux-x86_64.sh --user
export PATH="$PATH:$HOME/bin"
```

If you fail to install properly Bazel, you can read the beginning of [the instructions](https://www.tensorflow.org/install/install_sources) that present alternative methods for this.

### Required packages
There is a few required packages that you need to install:
```
sudo apt install python-dev python-pip python3-dev python3-pip python3-mock
sudo pip install pip six numpy wheel mock keras
sudo pip3 install pip six numpy wheel mock keras
```

For a pure python3 install, you might need to workaround a bazel bug the following way:
```
sudo ln -s /usr/bin/python3 /usr/bin/python
```

### Build TensorFlow the right way
Now, let's build TensorFlow with all the stuff required by OTBTF.
Make a directory for TensorFlow.
For instance `mkdir /work/tf`.

Clone TensorFlow.
```
cd /work/tf
git clone https://github.com/tensorflow/tensorflow.git
```

Now configure the project. If you have CUDA and other NVIDIA stuff installed in your system, remember that you have to tell the script that it is in `/usr/` (no symlink required!).

```
cd tensorflow
./configure
```
Then, you have to build TensorFlow with the instructions sets supported by your CPU (For instance here is AVX, AVX2, FMA, SSE4.1, SSE4.2 that play fine on a modern intel CPU). You have to tell Bazel to build:

 1. The TensorFlow python pip package
 2. The libtensorflow_cc.so library
 3. The libtensorflow_framework.so library
```
bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.1 --copt=-msse4.2 //tensorflow:libtensorflow_framework.so //tensorflow:libtensorflow_cc.so //tensorflow:libtensorflow.so //tensorflow/tools/pip_package:build_pip_package
```

*You might fail this step (e.g. missing packages). In this case, it's recommended to clear the bazel cache, using something like `rm $HOME/.cache/bazel/* -rf` before configuring and building everything!*

### Prepare the right stuff to use TensorFlow in external (cmake) projects
This is the most important!
First, build and deploy the pip package.
```
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install /tmp/tensorflow_pkg/tensorflow-1.12.0rc0-cp27-cp27mu-linux_x86_64.whl
```
For the C++ API, it's a bit more tricky.
Let's begin.
First, download dependencies.
```
/work/tf/tensorflow/tensorflow/contrib/makefile/download_dependencies.sh
```
Then, build Google Protobuf
```
mkdir /tmp/proto
cd /work/tf/tensorflow/tensorflow/contrib/makefile/downloads/protobuf/
./autogen.sh
./configure --prefix=/tmp/proto/
make -j $(grep -c ^processor /proc/cpuinfo)
make install
```
Then, "build" eigen (header only...)
```
mkdir /tmp/eigen
cd ../eigen
mkdir build_dir
cd build_dir
cmake -DCMAKE_INSTALL_PREFIX=/tmp/eigen/ ../
make install -j $(grep -c ^processor /proc/cpuinfo)
```
Then, build NSync
```
/work/tf/tensorflow/tensorflow/contrib/makefile/compile_nsync.sh
```
Then, build absl
```
mkdir /tmp/absl
cd /work/tf/tensorflow/tensorflow/contrib/makefile/downloads/absl/
mkdir build_dir
cd build_dir
cmake -DCMAKE_INSTALL_PREFIX=/tmp/absl ../
make -j $(grep -c ^processor /proc/cpuinfo)
```
Now, you have to copy the useful stuff in a directory

```
# Create folders
mkdir /work/tf/installdir
mkdir /work/tf/installdir/lib
mkdir /work/tf/installdir/include

# Copy libs
cp /work/tf/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so /work/tf/installdir/lib/
cp /work/tf/tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so /work/tf/installdir/lib/
cp /tmp/proto/lib/libprotobuf.a /work/tf/installdir/lib/
cp /work/tf/tensorflow/tensorflow/contrib/makefile/downloads/nsync/builds/default.linux.c++11/*.a /work/tf/installdir/lib/
ln -s /work/tf/installdir/lib/libtensorflow_framework.so /work/tf/installdir/lib/libtensorflow_framework.so.1
ln -s /work/tf/installdir/lib/libtensorflow_cc.so /work/tf/installdir/lib/libtensorflow_cc.so.1

# Copy headers
mkdir /work/tf/installdir/include/tensorflow
cp -r /work/tf/tensorflow/bazel-genfiles/* /work/tf/installdir/include
cp -r /work/tf/tensorflow/tensorflow/cc /work/tf/installdir/include/tensorflow
cp -r /work/tf/tensorflow/tensorflow/core /work/tf/installdir/include/tensorflow
cp -r /work/tf/tensorflow/third_party /work/tf/installdir/include
cp -r /tmp/proto/include/* /work/tf/installdir/include
cp -r /tmp/eigen/include/eigen3/* /work/tf/installdir/include
cp /work/tf/tensorflow/tensorflow/contrib/makefile/downloads/nsync/public/* /work/tf/installdir/include/
cd /work/tf/tensorflow/tensorflow/contrib/makefile/downloads/absl
find absl/ -name '*.h' -exec cp --parents \{\} /work/tf/installdir/include/ \; 
find absl/ -name '*.inc' -exec cp --parents \{\} /work/tf/installdir/include/ \; 

# Cleaning
find /work/tf/installdir/ -name "*.cc" -type f -delete
```
Well done. Now you have a working copy of TensorFlow located in `/work/tf/installdir` that is ready to use in external C++ cmake projects :)

## Build this remote module
Finally, we can build this module.
Clone the repository in your the OTB sources directory for remote modules (something like `/work/otb/OTB/Modules/Remote/`).
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
```
Done !

Don't forget to add some important environment variables, and this is finished.

```
export PATH="$PATH:/work/otb/superbuild_install/bin/"
export PYTHONPATH="$PYTHONPATH:/work/otb/superbuild_install/lib/otb/python"
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

