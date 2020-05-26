FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

MAINTAINER Remi Cresson <remi.cresson[at]irstea[dot]fr>

RUN apt-get update -y \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends \
        sudo \
        ca-certificates \
        curl \
        make \
        cmake \
        g++ \
        gcc \
        git \
        libtool \
        swig \
        xvfb \
        wget \
        autoconf \
        automake \
        pkg-config \
        zip \
        zlib1g-dev \
        unzip \
 && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# OTB and TensorFlow dependencies
# ----------------------------------------------------------------------------
RUN apt-get update -y \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends \
        freeglut3-dev \
        libboost-date-time-dev \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-program-options-dev \
        libboost-system-dev \
        libboost-thread-dev \
        libcurl4-gnutls-dev \
        libexpat1-dev \
        libfftw3-dev \
        libgdal-dev \
        libgeotiff-dev \
        libglew-dev \
        libglfw3-dev \
        libgsl-dev \
        libinsighttoolkit4-dev \
        libkml-dev \
        libmuparser-dev \
        libmuparserx-dev \
        libopencv-core-dev \
        libopencv-ml-dev \
        libopenthreads-dev \
        libossim-dev \
        libpng-dev \
        libqt5opengl5-dev \
        libqwt-qt5-dev \
        libsvm-dev \
        libtinyxml-dev \
        qtbase5-dev \
        qttools5-dev \
        default-jdk \
        python3-pip \
        python3.6-dev \
        python3.6-gdal \
        python3-setuptools \
        libxmu-dev \
        libxi-dev \
        qttools5-dev-tools \
        bison \
        software-properties-common \
        dirmngr \
        apt-transport-https \
        lsb-release \
        gdal-bin \
 && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# Python packages
# ----------------------------------------------------------------------------
RUN ln -s /usr/bin/python3 /usr/bin/python \
 && python3 -m pip install --upgrade pip \
 && python3 -m pip install pip six numpy wheel mock keras future setuptools

# ----------------------------------------------------------------------------
# Build TensorFlow
# ----------------------------------------------------------------------------
RUN export TF_ROOT=/work/tf \
 && mkdir -p ${TF_ROOT}/bazel \
 && cd ${TF_ROOT}/bazel \
 && wget https://github.com/bazelbuild/bazel/releases/download/0.29.1/bazel-0.29.1-installer-linux-x86_64.sh \
 && chmod +x bazel-0.29.1-installer-linux-x86_64.sh \
 && ./bazel-0.29.1-installer-linux-x86_64.sh

RUN export TF_ROOT=/work/tf \
 && export PATH="$PATH:$HOME/bin" \
 && cd $TF_ROOT \
 && git clone https://github.com/tensorflow/tensorflow.git \
 && cd tensorflow \
 && git checkout r2.1 \
 && export PYTHON_BIN_PATH=$(which python) \
 && export PYTHON_LIB_PATH="$($PYTHON_BIN_PATH -c 'import site; print(site.getsitepackages()[0])')" \
 && export PYTHONPATH=/usr/lib/ \
 && export PYTHON_ARG=/usr/lib/ \
 && export CUDA_TOOLKIT_PATH=/usr/local/cuda-10.1/ \
 && export CUDNN_INSTALL_PATH=/usr/ \
 && export TF_NEED_GCP=0 \
 && export TF_NEED_CUDA=1 \
 && export TF_CUDA_VERSION="$($CUDA_TOOLKIT_PATH/bin/nvcc --version | sed -n 's/^.*release \(.*\),.*/\1/p')" \
 && export TF_CUDA_COMPUTE_CAPABILITIES=6.1,5.2,3.5 \
 && export TF_NEED_HDFS=0 \
 && export TF_NEED_OPENCL=0 \
 && export TF_NEED_JEMALLOC=1 \
 && export TF_ENABLE_XLA=0 \
 && export TF_NEED_VERBS=0 \
 && export TF_CUDA_CLANG=0 \
 && export TF_CUDNN_VERSION="$(sed -n 's/^#define CUDNN_MAJOR\s*\(.*\).*/\1/p' $CUDNN_INSTALL_PATH/include/cudnn.h)" \
 && export TF_NEED_MKL=0 \
 && export TF_DOWNLOAD_MKL=0 \
 && export TF_NEED_AWS=0 \
 && export TF_NEED_MPI=0 \
 && export TF_NEED_GDR=0 \
 && export TF_NEED_S3=0 \
 && export TF_NEED_OPENCL_SYCL=0 \
 && export TF_SET_ANDROID_WORKSPACE=0 \
 && export TF_NEED_COMPUTECPP=0 \
 && export GCC_HOST_COMPILER_PATH=$(which gcc) \
 && export CC_OPT_FLAGS="-march=native" \
 && export TF_NEED_KAFKA=0 \
 && export TF_NEED_TENSORRT=0 \
 && export TF_NCCL_VERSION=2.4 \
 && export GCC_HOST_COMPILER_PATH=$(which gcc) \
 && export CC_OPT_FLAGS="-march=native" \
 && ./configure \
 && bazel build //tensorflow:libtensorflow_framework.so //tensorflow:libtensorflow_cc.so //tensorflow/tools/pip_package:build_pip_package --noincompatible_do_not_split_linking_cmdline

# ----------------------------------------------------------------------------
# Build protobuf
# ----------------------------------------------------------------------------
RUN export TF_ROOT=/work/tf \
 && mkdir -p $TF_ROOT/installdir \
 && cd $TF_ROOT \
 && wget https://github.com/google/protobuf/releases/download/v3.8.0/protobuf-cpp-3.8.0.tar.gz \
 && tar -xvf protobuf-cpp-3.8.0.tar.gz \
 && cd protobuf-3.8.0 \
 && ./configure --prefix=/work/tf/installdir/ \
 && make install -j $(grep -c ^processor /proc/cpuinfo)

# ----------------------------------------------------------------------------
# Prepare TF dependencies
# ----------------------------------------------------------------------------
RUN export TF_ROOT=/work/tf \
 && cd $TF_ROOT/tensorflow \
 && bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg \
 && pip3 install $(find /tmp/tensorflow_pkg/ -type f -iname "tensorflow*.whl") \
 && ./tensorflow/lite/tools/make/download_dependencies.sh \
 && ./tensorflow/lite/tools/make/build_lib.sh \
 && mkdir -p /work/tf/installdir/lib \
 && mkdir -p /work/tf/installdir/include \
 && cp bazel-bin/tensorflow/libtensorflow_cc.so* /work/tf/installdir/lib \
 && cp bazel-bin/tensorflow/libtensorflow_framework.so* /work/tf/installdir/lib \
 && cp -r bazel-genfiles/* /work/tf/installdir/include \
 && cp -r tensorflow/cc /work/tf/installdir/include/tensorflow \
 && cp -r tensorflow/core /work/tf/installdir/include/tensorflow \
 && cp -r third_party /work/tf/installdir/include \
 && cp -r bazel-tensorflow/external/eigen_archive/unsupported /work/tf/installdir/include \
 && cp -r bazel-tensorflow/external/eigen_archive/Eigen /work/tf/installdir/include \
 && cp -r tensorflow/lite/tools/make/downloads/absl/absl /work/tf/installdir/include 

# ----------------------------------------------------------------------------
# Build OTB: Stage 1 (clone)
# ----------------------------------------------------------------------------
RUN mkdir -p /work/otb \
 && cd /work/otb \
 && git clone https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb.git otb \
 && cd otb \
 && git checkout release-7.1

# ----------------------------------------------------------------------------
# Build OTB: Stage 2 (superbuild)
# ----------------------------------------------------------------------------
RUN mkdir -p /work/otb/build \
 && cd /work/otb/build \
 && cmake /work/otb/otb/SuperBuild \
        -DUSE_SYSTEM_BOOST=ON \
        -DUSE_SYSTEM_CURL=ON \
        -DUSE_SYSTEM_EXPAT=ON \
        -DUSE_SYSTEM_FFTW=ON \
        -DUSE_SYSTEM_FREETYPE=ON \
        -DUSE_SYSTEM_GDAL=ON \
        -DUSE_SYSTEM_GEOS=ON \
        -DUSE_SYSTEM_GEOTIFF=ON \
        -DUSE_SYSTEM_GLEW=ON \
        -DUSE_SYSTEM_GLFW=ON \
        -DUSE_SYSTEM_GLUT=ON \
        -DUSE_SYSTEM_GSL=ON \
        -DUSE_SYSTEM_ITK=ON \
        -DUSE_SYSTEM_LIBKML=ON \
        -DUSE_SYSTEM_LIBSVM=ON \
        -DUSE_SYSTEM_MUPARSER=ON \
        -DUSE_SYSTEM_MUPARSERX=ON \
        -DUSE_SYSTEM_OPENCV=ON \
        -DUSE_SYSTEM_OPENTHREADS=ON \
        -DUSE_SYSTEM_OSSIM=ON \
        -DUSE_SYSTEM_PNG=ON \
        -DUSE_SYSTEM_QT5=ON \
        -DUSE_SYSTEM_QWT=ON \
        -DUSE_SYSTEM_TINYXML=ON \
        -DUSE_SYSTEM_ZLIB=ON \
        -DUSE_SYSTEM_SWIG=OFF \
        -DOTB_WRAP_PYTHON=OFF \
 && make -j $(grep -c ^processor /proc/cpuinfo)

# ----------------------------------------------------------------------------
# Build OTB: Stage 3 (bindings)
# ----------------------------------------------------------------------------
RUN cd /work/otb/otb/Modules/Remote \
 && git clone -b v2beta https://github.com/remicres/otbtf.git \
 && cd /work/otb/build/OTB/build \
 && cmake /work/otb/otb \
        -DOTB_WRAP_PYTHON=ON \
        -DPYTHON_EXECUTABLE=/usr/bin/python3.6 \
        -Dopencv_INCLUDE_DIR=/usr/include \
        -DModule_OTBTensorflow=ON \
        -DOTB_USE_TENSORFLOW=ON \
        -DTENSORFLOW_CC_LIB=/work/tf/installdir/lib/libtensorflow_cc.so \
        -DTENSORFLOW_FRAMEWORK_LIB=/work/tf/installdir/lib/libtensorflow_framework.so \
        -Dtensorflow_include_dir=/work/tf/installdir/include/ \
 && cd /work/otb/build/ \
 && make -j $(grep -c ^processor /proc/cpuinfo)

# ----------------------------------------------------------------------------
# Add important environment variables
# ----------------------------------------------------------------------------
ENV PATH="$PATH:/work/otb/superbuild_install/bin/"
ENV PYTHONPATH="/work/otb/superbuild_install/lib/otb/python:/work/otb/otb/Modules/Remote/otbtf/python:$PYTHONPATH"
ENV OTB_APPLICATION_PATH="/work/otb/superbuild_install/lib/otb/applications"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/work/otb/superbuild_install/lib/:/work/tf/installdir/lib/"

# ----------------------------------------------------------------------------
# Clean
# ----------------------------------------------------------------------------
RUN rm -rf /tmp/* /root/.cache && apt-get clean

# ----------------------------------------------------------------------------
# Add one user
# ----------------------------------------------------------------------------
RUN useradd -s /bin/bash -m otbuser
USER otbuser
WORKDIR /home/otbuser

# ----------------------------------------------------------------------------
# Exemple :
# docker build --tag otbtf_image
# docker run -v /path/to/host/:/path/to/mount/ otbtf_image otbcli_ExtractROI
# docker run -ti -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY otbtf_image otbgui_ExtractROI
# ----------------------------------------------------------------------------

