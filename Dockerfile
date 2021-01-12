##### Configurable Dockerfile with multi-stage build - Author: Vincent Delbar
##############################################################################
ARG BASE_IMG
# Mandatory, ubuntu:20.04 and nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04

# ----------------------------------------------------------------------------
# Init base stage - will be cloned as intermediate env
# ----------------------------------------------------------------------------
FROM $BASE_IMG AS otbtf-base
WORKDIR /tmp

### Sys packages
COPY tools/docker/build-deps-*.txt ./
ARG DEBIAN_FRONTEND=noninteractive

# CLI
RUN apt-get update -y \
 && cat build-deps-cli.txt | xargs apt-get install --no-install-recommends -y \
 && apt-get clean && rm -rf /var/lib/apt/lists/*
# GUI
ARG GUI=false
RUN if $GUI ; then \
        apt-get update -y \
        && cat build-deps-gui.txt | xargs apt-get install --no-install-recommends -y \
        && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* ; fi

### Python3 link
RUN ln -s /usr/bin/python3 /usr/local/bin/python && ln -s /usr/bin/pip3 /usr/local/bin/pip

# NumPy version is a problem with system's gdal dep - venv could be a better option than just being first in PYTHONPATH
RUN pip install --no-cache-dir -U numpy future pip six mock wheel \
 && pip install --no-cache-dir keras_applications --no-deps \
 && pip install --no-cache-dir keras_preprocessing --no-deps

# ----------------------------------------------------------------------------
# Tmp builder stage - dangling cache should persist until "docker system prune"
# ----------------------------------------------------------------------------
FROM otbtf-base AS builder

# May be required to avoid OOM errors (0.95 eq to n-1 if n>=10)
ARG CPU_RATIO=0.95

RUN mkdir -p /src /opt/otbtf
WORKDIR /tmp

### Get bazel bin
ARG BAZEL=3.1.0
RUN wget https://github.com/bazelbuild/bazel/releases/download/$BAZEL/bazel-$BAZEL-installer-linux-x86_64.sh \
 && bash bazel-$BAZEL-installer-linux-x86_64.sh --prefix=/opt/otbtf \
 && rm -f /tmp/*

### TF
ARG TF=r2.4
ARG BZL_TARGETS="//tensorflow:libtensorflow_cc.so //tensorflow:libtensorflow_framework.so //tensorflow/tools/pip_package:build_pip_package"
ARG BZL_CONFIG="--config=opt --config=nogcp --config=noaws --config=nohdfs"
ARG BZL_OPTIONS="--compilation_mode opt --verbose_failures --remote_cache=http://localhost:9090"

RUN mkdir /src/tf
WORKDIR /src/tf
COPY tools/docker/build-env-tf.sh ./

# Build
ARG KEEP_SRC_TF=false
RUN git clone https://github.com/tensorflow/tensorflow.git -b $TF \
 && cd tensorflow \
 && export PATH=$PATH:/opt/otbtf/bin \
 && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/otbtf/lib \
 && bash -c " \
       source ../build-env-tf.sh \
       && export TMP=/tmp/bazel \
       && ./configure \
       && bazel build $BZL_TARGETS $BZL_CONFIG $BZL_OPTIONS --jobs=\"HOST_CPUS*$CPU_RATIO\"" \
# In order to debug you may need to split here, but bazel cache is huge - we can shrink cached docker layers
#RUN cd tensorflow \
 && ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg \
 && pip3 install --no-cache-dir --prefix=/opt/otbtf $(find /tmp/tensorflow_pkg/ -type f -iname "tensorflow*.whl") \
 && ./tensorflow/lite/tools/make/download_dependencies.sh \
 && ./tensorflow/lite/tools/make/build_lib.sh \
 && mkdir -p /opt/otbtf/lib /opt/otbtf/include/tensorflow \
 && cp bazel-bin/tensorflow/libtensorflow_cc.so* /opt/otbtf/lib \
 && cp bazel-bin/tensorflow/libtensorflow_framework.so* /opt/otbtf/lib \
 && cp -r tensorflow/cc /opt/otbtf/include/tensorflow \
 && cp -r tensorflow/core /opt/otbtf/include/tensorflow \
 && cp -r third_party /opt/otbtf/include \
 && cp -r bazel-tensorflow/external/eigen_archive/unsupported /opt/otbtf/include \
 && cp -r bazel-tensorflow/external/eigen_archive/Eigen /opt/otbtf/include \
 && cp -r tensorflow/lite/tools/make/downloads/absl/absl /opt/otbtf/include \
 # Cleaning
 && ( $KEEP_SRC_TF || rm -rf /src/tf ) \
 && rm -rf /root/.cache/ /tmp/*

### Build protobuf
ARG PROTOBUF=3.9.2
RUN cd /tmp \
 && wget https://github.com/protocolbuffers/protobuf/releases/download/v$PROTOBUF/protobuf-cpp-$PROTOBUF.tar.gz \
 && tar -xvf protobuf-cpp-$PROTOBUF.tar.gz \
 && cd protobuf-$PROTOBUF \
 && ./configure --prefix=/opt/otbtf \
 && make install -j $(python -c "import os; print(round( os.cpu_count() * $CPU_RATIO ))") \
 && rm -rf /root/.cache/ /tmp/*

# Link python site-packages in order to build with any ubuntu distribution
RUN cd /opt/otbtf/lib/ && ln -s $(find . -maxdepth 1 -type d -name "python3.*") python3

### OTB
ARG GUI=false
ARG OTB=release-7.2

RUN mkdir /src/otb
WORKDIR /src/otb
COPY tools/docker/build-flags-otb.txt ./

# SuperBuild OTB deps
RUN git clone https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb.git -b $OTB \
 && mkdir -p build \
 && cd build \
 # Set GL/Qt build flags
 && if $GUI; then \
        sed -i -r 's/-DOTB_USE_(QT|OPENGL|GL[UFE][WT])=OFF/-DOTB_USE_\1=ON/' ../build-flags-otb.txt; fi \
 && OTB_FLAGS=$(cat "../build-flags-otb.txt") \
 && cmake ../otb/SuperBuild \
        -DCMAKE_INSTALL_PREFIX=/opt/otbtf \
        $OTB_FLAGS \
 && make OTB_DEPENDS -j $(python -c "import os; print(round( os.cpu_count() * $CPU_RATIO ))")

### OTBTF
RUN mkdir /src/otb/otb/Modules/Remote/otbtf
# Copy local dir or pull files from repo
COPY . /src/otb/otb/Modules/Remote/otbtf
#RUN cd /src/otb/otb/Modules/Remote && git clone https://github.com/remicres/otbtf.git

# Build OTB
ARG KEEP_SRC_OTB=false
RUN cd /src/otb/build/OTB/build \
 && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/otbtf/lib \
 && export PATH=$PATH:/opt/otbtf/bin \
 && cmake /src/otb/otb \
        -DCMAKE_INSTALL_PREFIX=/opt/otbtf \
        -DOTB_WRAP_PYTHON=ON \
        -DPYTHON_EXECUTABLE=/usr/bin/python3 \
        -DModule_OTBTensorflow=ON \
        -DOTB_USE_TENSORFLOW=ON \
        -DTENSORFLOW_CC_LIB=/opt/otbtf/lib/libtensorflow_cc.so \
        -DTENSORFLOW_FRAMEWORK_LIB=/opt/otbtf/lib/libtensorflow_framework.so \
        -Dtensorflow_include_dir=/opt/otbtf/include/ \
        -DCMAKE_CXX_FLAGS="-I /opt/otbtf/lib/python3/site-packages/tensorflow/include/" \
 && cd /src/otb/build/ \
 && make -j $(python -c "import os; print(round( os.cpu_count() * $CPU_RATIO ))") \
 # Cleaning
 && ( $GUI || rm -rf /opt/otbtf/bin/otbgui* ) \
 && ( $KEEP_SRC_OTB || rm -rf /src/otb ) \
 && rm -rf /root/.cache /tmp/*

# ----------------------------------------------------------------------------
# Final stage
# ----------------------------------------------------------------------------
FROM otbtf-base
MAINTAINER Remi Cresson <remi.cresson[at]inrae[dot]fr>

# Copy installation from intermediate layer
COPY --from=builder /opt/otbtf /opt/otbtf
# /src will be empty, except with true KEEP_*_SRC variables
COPY --from=builder /src /src

# Persistent environment variables (all users)
ENV PYTHONPATH="/opt/otbtf/lib/python3/site-packages:/opt/otbtf/lib/otb/python:/home/otbuser/pyotbtf:$PYTHONPATH"
ENV PATH="/opt/otbtf/bin:$PATH"
ENV OTB_APPLICATION_PATH="/opt/otbtf/lib/otb/applications"
ENV LD_LIBRARY_PATH="/opt/otbtf/lib:$LD_LIBRARY_PATH"

# Required with TF<2.4
#ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Enable auto XLA JIT - will cause cublas errors with CUDA 11.1 (core dump)
#ENV TF_XLA_FLAGS="--tf_xla_auto_jit=2"

# Create default user
RUN useradd -s /bin/bash -m otbuser
WORKDIR /home/otbuser

# Copy python files
COPY --chown=otbuser python pyotbtf

# Give admin rights without password
ARG SUDO=true
RUN if $SUDO; then \
        usermod -a -G sudo otbuser \
        && echo "otbuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers; fi

# This won't prevent ownership problems when using volume - if you're not UID 1000, see "docker [run|create] -u $UID:$GID"
USER otbuser
# User variables goes here

# Test python imports
RUN python -c 'import tensorflow, otbtf, tricks, otbApplication'
