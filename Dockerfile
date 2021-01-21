##### Configurable Dockerfile with multi-stage build - Author: Vincent Delbar
## Mandatory
ARG BASE_IMG

# ----------------------------------------------------------------------------
# Init base stage - will be cloned as intermediate build env
FROM $BASE_IMG AS otbtf-base
WORKDIR /tmp

### System packages
COPY tools/docker/build-deps-*.txt ./
ARG DEBIAN_FRONTEND=noninteractive
# CLI
RUN apt-get update -y && apt-get upgrade -y \
 && cat build-deps-cli.txt | xargs apt-get install --no-install-recommends -y \
 && apt-get clean && rm -rf /var/lib/apt/lists/*
# Optional GUI
ARG GUI=false
RUN if $GUI; then \
      apt-get update -y \
      && cat build-deps-gui.txt | xargs apt-get install --no-install-recommends -y \
      && apt-get clean && rm -rf /var/lib/apt/lists/* ; fi

### Python3 links and pip packages
RUN ln -s /usr/bin/python3 /usr/local/bin/python && ln -s /usr/bin/pip3 /usr/local/bin/pip
# NumPy version is conflicting with system's gdal dep, may require venv
ARG NUMPY_SPEC="~=1.19"
RUN pip install --no-cache-dir -U pip wheel mock six future "numpy$NUMPY_SPEC" \
 && pip install --no-cache-dir --no-deps keras_applications keras_preprocessing

# ----------------------------------------------------------------------------
# Tmp builder stage - dangling cache should persist until "docker builder prune"
FROM otbtf-base AS builder
# 0.75 may be required to avoid OOM errors (especially for OTB GUI)
ARG CPU_RATIO=1

RUN mkdir -p /src/tf /opt/otbtf/bin /opt/otbtf/include /opt/otbtf/lib
WORKDIR /src/tf

### TF
ARG TF=v2.4.0
# Install bazelisk (will read .bazelrc and download the right bazel version - latest by default)
RUN wget -O /opt/otbtf/bin/bazelisk https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
 && chmod +x /opt/otbtf/bin/bazelisk-linux-amd64 \
 && ln -s /opt/otbtf/bin/bazelisk-linux-amd64 /opt/otbtf/bin/bazel

ARG BZL_TARGETS="//tensorflow:libtensorflow_cc.so //tensorflow:libtensorflow_framework.so //tensorflow/tools/pip_package:build_pip_package"
# --config=opt with bazel's default optimizations (otherwise edit CC_OPT_FLAGS in build-env-tf.sh)
ARG BZL_CONFIGS="--config=nogcp --config=noaws --config=nohdfs --config=opt"
# --compilation_mode opt is already enabled by default (see tf repo /.bazelrc and /configure.py)
ARG BZL_OPTIONS="--verbose_failures --remote_cache=http://localhost:9090"

# Build
ARG KEEP_SRC_TF=false
COPY tools/docker/build-env-tf.sh ./
RUN git clone --single-branch -b $TF https://github.com/tensorflow/tensorflow.git \
 && cd tensorflow \
 && export PATH=$PATH:/opt/otbtf/bin \
 && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/otbtf/lib \
 && bash -c '\
      source ../build-env-tf.sh \
      && ./configure \
      && export TMP=/tmp/bazel \
      && BZL_CMD="build $BZL_TARGETS $BZL_CONFIGS $BZL_OPTIONS"
      && bazel $BZL_CMD --jobs="HOST_CPUS*$CPU_RATIO" ' \
# Installation - split here in order to check logs      ^
#RUN cd tensorflow \
 && ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg \
 && pip3 install --no-cache-dir --prefix=/opt/otbtf /tmp/tensorflow_pkg/tensorflow*.whl \
 # Libraries
 && mkdir -p /opt/otbtf/lib \
 && cp -P bazel-bin/tensorflow/libtensorflow_cc.so* /opt/otbtf/lib/ \
 && cp -P bazel-bin/tensorflow/libtensorflow_framework.so* /opt/otbtf/lib/ \
 # => symlink external libs (required for MKL - libiomp5)
 && for f in $(find -L /opt/otbtf/include/tf -wholename "*/external/*/*.so"); do ln -s $f /opt/otbtf/lib/; done \
 # Headers
 && mkdir /opt/otbtf/include \
 && ln -s $(find /opt/otbtf -type d -wholename "*/site-packages/tensorflow/include") /opt/otbtf/include/tf \
 # => the only missing header in the wheel
 && cp tensorflow/cc/saved_model/tag_constants.h /opt/otbtf/include/tf/tensorflow/cc/saved_model/ \
 # Cleaning
 && mv /root/.cache/bazel* /src/tf/ \
 && ( $KEEP_SRC_TF || rm -rf /src/tf ) \
 && rm -rf /root/.cache/ /tmp/*
# Link wheel's site-packages in order to find any python3 version
RUN cd /opt/otbtf/lib && ln -s python3.* python3

### OTB
ARG GUI=false
ARG OTB=7.2.0

RUN mkdir /src/otb
WORKDIR /src/otb

# SuperBuild OTB
COPY tools/docker/build-flags-otb.txt ./
RUN git clone --single-branch -b $OTB https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb.git \
 && mkdir -p build \
 && cd build \
 # Set GL/Qt build flags
 && if $GUI; then \
      sed -i -r "s/-DOTB_USE_(QT|OPENGL|GL[UFE][WT])=OFF/-DOTB_USE_\1=ON/" ../build-flags-otb.txt; fi \
 && OTB_FLAGS=$(cat "../build-flags-otb.txt") \
 && cmake ../otb/SuperBuild -DCMAKE_INSTALL_PREFIX=/opt/otbtf $OTB_FLAGS \
 && make -j $(python -c "import os; print(round( os.cpu_count() * $CPU_RATIO ))")

### OTBTF - copy (without .git) or clone repo
COPY . /src/otbtf
#RUN git clone https://github.com/remicres/otbtf.git /src/otbtf
RUN ln -s /src/otbtf /src/otb/otb/Modules/Remote/otbtf

# Rebuild OTB with module
ARG KEEP_SRC_OTB=false
RUN cd /src/otb/build/OTB/build \
 && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/otbtf/lib \
 && export PATH=$PATH:/opt/otbtf/bin \
 && cmake /src/otb/otb \
      -DCMAKE_INSTALL_PREFIX=/opt/otbtf \
      -DOTB_WRAP_PYTHON=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 \
      -DOTB_USE_TENSORFLOW=ON -DModule_OTBTensorflow=ON \
      -Dtensorflow_include_dir=/opt/otbtf/include/tf \
      -DTENSORFLOW_CC_LIB=/opt/otbtf/lib/libtensorflow_cc.so \
      -DTENSORFLOW_FRAMEWORK_LIB=/opt/otbtf/lib/libtensorflow_framework.so \
 && make install -j $(python -c "import os; print(round( os.cpu_count() * $CPU_RATIO ))") \
 # Cleaning
 && ( $GUI || rm -rf /opt/otbtf/bin/otbgui* ) \
 && ( $KEEP_SRC_OTB || rm -rf /src/otb ) \
 && rm -rf /root/.cache /tmp/*

# Symlink executable python files in PATH
RUN for f in /src/otbtf/python/*.py; do if [ -x $f ]; then ln -s $f /opt/otbtf/bin/; fi; done

# ----------------------------------------------------------------------------
# Final stage
FROM otbtf-base
MAINTAINER Remi Cresson <remi.cresson[at]inrae[dot]fr>

COPY --from=builder /opt/otbtf /opt/
COPY --from=builder /src /
# Relocate ~/.cache/bazel and ~/.cache/bazelisk
RUN if [ -d /src/tf/bazel ]; then \
      mkdir -p /root/.cache && mv /src/tf/bazel* /root/.cache/

# System-wide ENV
ENV PATH="/opt/otbtf/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/otbtf/lib:$LD_LIBRARY_PATH"
ENV PYTHONPATH="/opt/otbtf/lib/python3/site-packages:/opt/otbtf/lib/otb/python:/src/otbtf/python"
ENV OTB_APPLICATION_PATH="/opt/otbtf/lib/otb/applications"

# Default user, directory and command (bash = 'docker create' entrypoint)
RUN useradd -s /bin/bash -m otbuser
WORKDIR /home/otbuser

# Admin rights without password
ARG SUDO=true
RUN if $SUDO; then \
      usermod -a -G sudo otbuser \
      && echo "otbuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers; fi

# Set /src/otbtf ownership to otbuser (but you will need 'sudo -i' in order to rebuild TF or OTB)
RUN chown -R otbuser:otbuser /src/otbtf

# This won't prevent ownership problems with volumes if you're not UID 1000
USER otbuser
# User-only ENV

# Test python imports
RUN python -c "import numpy, tensorflow, otbtf, tricks, otbApplication"
