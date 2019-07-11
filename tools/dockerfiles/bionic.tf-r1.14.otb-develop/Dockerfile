FROM ubuntu:18.04

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
 && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# Python packages
# ----------------------------------------------------------------------------
RUN ln -s /usr/bin/python3 /usr/bin/python \
 && python3 -m pip install --upgrade pip \
 && python3 -m pip install pip six numpy wheel mock keras future

# ----------------------------------------------------------------------------
# Build TensorFlow
# ----------------------------------------------------------------------------
RUN export TF_ROOT=/work/tf \
 && mkdir -p ${TF_ROOT}/bazel \
 && cd ${TF_ROOT}/bazel \
 && wget https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel-0.24.1-installer-linux-x86_64.sh \
 && chmod +x bazel-0.24.1-installer-linux-x86_64.sh \
 && ./bazel-0.24.1-installer-linux-x86_64.sh

RUN export TF_ROOT=/work/tf \
 && export PATH="$PATH:$HOME/bin" \
 && cd $TF_ROOT \
 && git clone https://github.com/tensorflow/tensorflow.git \
 && cd tensorflow \
 && git checkout r1.14 \
 && echo "\n\n\n\n\n\n\n\n\n" | ./configure \
 && bazel build //tensorflow:libtensorflow_framework.so //tensorflow:libtensorflow_cc.so //tensorflow/tools/pip_package:build_pip_package

RUN export TF_ROOT=/work/tf \
 && cd $TF_ROOT/tensorflow \
 && bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg \
 && pip3 install $(find /tmp/tensorflow_pkg/ -type f -iname "tensorflow*.whl") \
 && ./tensorflow/contrib/makefile/build_all_linux.sh \
 && mkdir -p /work/tf/installdir/lib \
 && mkdir -p /work/tf/installdir/include \
 && cp bazel-bin/tensorflow/libtensorflow_cc.so                                          /work/tf/installdir/lib \
 && cp bazel-bin/tensorflow/libtensorflow_framework.so                                   /work/tf/installdir/lib \
 && cp tensorflow/contrib/makefile/gen/protobuf/lib/libprotobuf.a                        /work/tf/installdir/lib \
 && cp tensorflow/contrib/makefile/downloads/nsync/builds/default.linux.c++11/*.a        /work/tf/installdir/lib \
 && cp -r bazel-genfiles/*                                                               /work/tf/installdir/include/ \
 && cp -r tensorflow/cc                                                                  /work/tf/installdir/include/tensorflow/ \
 && cp -r tensorflow/core                                                                /work/tf/installdir/include/tensorflow/ \
 && cp -r third_party                                                                    /work/tf/installdir/include/ \
 && cp -r tensorflow/contrib/makefile/gen/protobuf/include/*                             /work/tf/installdir/include/ \
 && cp -r tensorflow/contrib/makefile/downloads/eigen/Eigen                              /work/tf/installdir/include/ \
 && cp -r tensorflow/contrib/makefile/downloads/eigen/unsupported                        /work/tf/installdir/include/ \
 && cp -r tensorflow/contrib/makefile/downloads/eigen/signature_of_eigen3_matrix_library /work/tf/installdir/include/ \
 && cd ${TF_ROOT}/tensorflow/tensorflow/contrib/makefile/downloads/absl \
 && find absl/ -name '*.h' -exec cp --parents \{\}                                       /work/tf/installdir/include/ \; \
 && find absl/ -name '*.inc' -exec cp --parents \{\}                                     /work/tf/installdir/include/ \; \
 && find /work/tf/installdir/ -name "*.cc" -type f -delete 

RUN echo "Create symlinks for tensorflow libs" \
 && ln -s /work/tf/installdir/lib/libtensorflow_cc.so /work/tf/installdir/lib/libtensorflow_cc.so.1 \
 && ln -s /work/tf/installdir/lib/libtensorflow_framework.so /work/tf/installdir/lib/libtensorflow_framework.so.1

# ----------------------------------------------------------------------------
# Build OTB
# ----------------------------------------------------------------------------
RUN mkdir -p /work/otb/build \
 && cd /work/otb \
 && git clone https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb.git otb \
 && cd otb \
 && git checkout 0df44b312d64d6c3890b65d3790d4a17d0fd5f23 \
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
 && cd /work/otb/otb/Modules/Remote \
 && git clone https://github.com/remicres/otbtf.git \
 && cd /work/otb/build/OTB/build \
 && cmake /work/otb/otb \
        -DOTB_WRAP_PYTHON=ON \
        -DModule_Mosaic=ON \
        -DModule_OTBTensorflow=ON \
        -DOTB_USE_TENSORFLOW=ON \
        -Dopencv_INCLUDE_DIR=/usr/include \
        -DTENSORFLOW_CC_LIB=/work/tf/installdir/lib/libtensorflow_cc.so \
        -DTENSORFLOW_FRAMEWORK_LIB=/work/tf/installdir/lib/libtensorflow_framework.so \
        -Dtensorflow_include_dir=/work/tf/installdir/include/ \
 && cd /work/otb/build/ \
 && make -j $(grep -c ^processor /proc/cpuinfo)

# ----------------------------------------------------------------------------
# Install GDAL
# ----------------------------------------------------------------------------
RUN apt-get update -y \
#  && apt-get upgrade -y \
  && apt -y install \
	software-properties-common \
	dirmngr \
	apt-transport-https \
	lsb-release \
  && add-apt-repository ppa:ubuntugis/ppa \
  && apt-get update \
  && apt-get install -y gdal-bin \
  && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# Add user otbuser # Replace 1001 with your user / group id
# ----------------------------------------------------------------------------

RUN export uid=1001 gid=1001 && \
    mkdir -p /home/otbuser && \
    echo "otbuser:x:${uid}:${gid}:otbuser,,,:/home/otbuser:/bin/bash" >> /etc/passwd && \
    echo "otbuser:x:${uid}:" >> /etc/group && \
    echo "otbuser ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/otbuser && \
    chmod 0440 /etc/sudoers.d/otbuser && \
    chown ${uid}:${gid} -R /home/otbuser

USER /home/otbuser
ENV HOME /home/otbuser
ENV OTB_MAX_RAM_HINT=512
ENV PATH "$PATH:/work/otb/superbuild_install/bin/"
ENV PYTHONPATH="/work/otb/superbuild_install/lib/otb/python:$PYTHONPATH"
ENV OTB_APPLICATION_PATH="/work/otb/superbuild_install/lib/otb/applications"
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/work/otb/superbuild_install/lib/:/work/tf/installdir/lib/"
WORKDIR /home/otbuser

# ----------------------------------------------------------------------------
# Exemple :
# docker run -v $(pwd):/home/otbuser otbtf otbcli_ExtractROI -h
# docker run -ti -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v $(pwd):/home/otbuser otbtf otbgui_ExtractROI
# ----------------------------------------------------------------------------
