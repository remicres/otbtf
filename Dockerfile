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
RUN apt-get update -y && apt-get upgrade -y \
 && cat build-deps-cli.txt | xargs apt-get install --no-install-recommends -y \
 && apt-get clean && rm -rf /var/lib/apt/lists/*
### Python3 links and pip packages
RUN ln -s /usr/bin/python3 /usr/local/bin/python && ln -s /usr/bin/pip3 /usr/local/bin/pip
# Upgrade pip
RUN pip install --no-cache-dir pip --upgrade
# In case NumPy version is conflicting with system's gdal dep and may require venv
ARG NUMPY_SPEC=""
# This is to avoid https://github.com/tensorflow/tensorflow/issues/61551
ARG PROTO_SPEC="==4.23.*"
RUN pip install --no-cache-dir -U wheel mock six future tqdm deprecated "numpy$NUMPY_SPEC" "protobuf$PROTO_SPEC" packaging requests \
 && pip install --no-cache-dir --no-deps keras_applications keras_preprocessing

# ----------------------------------------------------------------------------
# Tmp builder stage - dangling cache should persist until "docker builder prune"
FROM otbtf-base AS builder
# A smaller value may be required to avoid OOM errors when building OTB
ARG CPU_RATIO=1

RUN mkdir -p /src/tf /opt/otbtf/bin /opt/otbtf/include /opt/otbtf/lib/python3
WORKDIR /src/tf

RUN git config --global advice.detachedHead false

### TF

ARG TF=v2.14.0
ARG TENSORRT

# Install bazelisk (will read .bazelversion and download the right bazel binary - latest by default)
RUN wget -qO /opt/otbtf/bin/bazelisk https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
 && chmod +x /opt/otbtf/bin/bazelisk \
 && ln -s /opt/otbtf/bin/bazelisk /opt/otbtf/bin/bazel

ARG BZL_TARGETS="//tensorflow:libtensorflow_cc.so //tensorflow/tools/pip_package:build_pip_package"

# "--config=opt" will enable 'march=native'
# (otherwise read comments about CPU compatibility and edit CC_OPT_FLAGS in
# build-env-tf.sh)
ARG BZL_CONFIGS="--config=nogcp --config=noaws --config=nohdfs --config=opt"

# "--compilation_mode opt" is already enabled by default (see tf repo .bazelrc
# and configure.py)
ARG BZL_OPTIONS="--verbose_failures --remote_cache=http://localhost:9090"

# Build
ARG ZIP_TF_BIN=false
COPY tools/docker/build-env-tf.sh ./
RUN git clone --single-branch -b $TF https://github.com/tensorflow/tensorflow.git
RUN cd tensorflow \
 && export PATH=$PATH:/opt/otbtf/bin \
 && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/otbtf/lib \
 && bash -c '\
      source ../build-env-tf.sh \
      && ./configure \
      && export TMP=/tmp/bazel \
      && BZL_CMD="build $BZL_TARGETS $BZL_CONFIGS $BZL_OPTIONS" \
      && bazel $BZL_CMD --jobs="HOST_CPUS*$CPU_RATIO" '

# Installation
RUN apt update && apt install -y patchelf
RUN cd tensorflow \
 && ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg \
 && pip3 install --no-cache-dir --prefix=/opt/otbtf /tmp/tensorflow_pkg/tensorflow*.whl \
 && ln -s /opt/otbtf/local/lib/python3.*/* /opt/otbtf/lib/python3 \
 && ln -s /opt/otbtf/local/bin/* /opt/otbtf/bin \
 && ln -s $(find /opt/otbtf -type d -wholename "*/dist-packages/tensorflow/include") /opt/otbtf/include/tf \
 # The only missing header in the wheel
 && cp tensorflow/cc/saved_model/tag_constants.h /opt/otbtf/include/tf/tensorflow/cc/saved_model/ \
 && cp tensorflow/cc/saved_model/signature_constants.h /opt/otbtf/include/tf/tensorflow/cc/saved_model/ \
 # Symlink external libs (required for MKL - libiomp5)
 && for f in $(find -L /opt/otbtf/include/tf -wholename "*/external/*/*.so"); do ln -s $f /opt/otbtf/lib/; done \
 # Compress and save TF binaries
 && ( ! $ZIP_TF_BIN || zip -9 -j --symlinks /opt/otbtf/tf-$TF.zip tensorflow/cc/saved_model/tag_constants.h tensorflow/cc/saved_model/signature_constants.h bazel-bin/tensorflow/libtensorflow_cc.so* /tmp/tensorflow_pkg/tensorflow*.whl ) \
 # Cleaning
 && rm -rf bazel-* /src/tf /root/.cache/ /tmp/*

### OTB

ARG OTB=release-9.0
ARG OTBTESTS=false

RUN mkdir /src/otb
WORKDIR /src/otb

# SuperBuild OTB
COPY tools/docker/build-flags-otb.txt ./
RUN apt-get update -y \
 && apt-get install --reinstall ca-certificates -y \
 && update-ca-certificates \
 && git clone https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb.git \
 && cd otb && git checkout $OTB

RUN cd /src/otb/ \
 && mkdir -p build \
 && cd build \
 && if $OTBTESTS; then \
      echo "-DBUILD_TESTING=ON" >> ../build-flags-otb.txt; fi \
 # Possible ENH: superbuild-all-dependencies switch, with separated build-deps-minimal.txt and build-deps-otbcli.txt)
 #&& if $OTB_SUPERBUILD_ALL; then sed -i -r "s/-DUSE_SYSTEM_([A-Z0-9]*)=ON/-DUSE_SYSTEM_\1=OFF/ " ../build-flags-otb.txt; fi \
 && OTB_FLAGS=$(cat "../build-flags-otb.txt") \
 && cmake ../otb/SuperBuild -DCMAKE_INSTALL_PREFIX=/opt/otbtf $OTB_FLAGS \
 && make -j $(python -c "import os; print(round( os.cpu_count() * $CPU_RATIO ))")

### OTBTF - copy (without .git/) or clone repository
COPY . /src/otbtf
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
      -DTENSORFLOW_CC_LIB=/opt/otbtf/local/lib/python3.10/dist-packages/tensorflow/libtensorflow_cc.so.2 \
      -DTENSORFLOW_FRAMEWORK_LIB=/opt/otbtf/local/lib/python3.10/dist-packages/tensorflow/libtensorflow_framework.so.2 \
 && make install -j $(python -c "import os; print(round( os.cpu_count() * $CPU_RATIO ))") \
 # Cleaning
 && ( $KEEP_SRC_OTB || rm -rf /src/otb ) \
 && rm -rf /root/.cache /tmp/*

# Symlink executable python files in PATH
RUN for f in /src/otbtf/python/*.py; do if [ -x $f ]; then ln -s $f /opt/otbtf/bin/; fi; done

# ----------------------------------------------------------------------------
# Final stage
FROM otbtf-base
LABEL maintainer="Remi Cresson <remi.cresson[at]inrae[dot]fr>"

# Copy files from intermediate stage
COPY --from=builder /opt/otbtf /opt/otbtf
COPY --from=builder /src /src

# System-wide ENV
ENV PATH="/opt/otbtf/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/otbtf/lib:$LD_LIBRARY_PATH"
ENV PYTHONPATH="/opt/otbtf/lib/python3/dist-packages:/opt/otbtf/lib/otb/python"
ENV OTB_APPLICATION_PATH="/opt/otbtf/lib/otb/applications"
RUN pip install -e /src/otbtf

# Default user, directory and command (bash is the entrypoint when using
# 'docker create')
RUN useradd -s /bin/bash -m otbuser
WORKDIR /home/otbuser

# Admin rights without password
ARG SUDO=true
RUN if $SUDO; then \
      usermod -a -G sudo otbuser \
      && echo "otbuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers; fi

# Set /src/otbtf ownership to otbuser (but you still need 'sudo -i' in order
# to rebuild TF or OTB)
RUN chown -R otbuser:otbuser /src/otbtf

# This won't prevent ownership problems with volumes if you're not UID 1000
USER otbuser

# User-only ENV
ENV PATH="/home/otbuser/.local/bin:$PATH"

# Test python imports
RUN python -c "import tensorflow"
RUN python -c "import otbtf, tricks"
RUN python -c "import otbApplication as otb; otb.Registry.CreateApplication('ImageClassifierFromDeepFeatures')"
RUN python -c "from osgeo import gdal"
