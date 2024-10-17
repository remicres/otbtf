##### OTBTF configurable Dockerfile with multi-stage build
# ----------------------------------------------------------------------------
# Init base stage - used for intermediate build env and final image

# Freeze ubuntu version to avoid suprise rebuild
FROM ubuntu:noble-20241011 AS base-stage

WORKDIR /tmp

### System packages
ARG DEBIAN_FRONTEND=noninteractive
COPY system-dependencies.txt ./
RUN apt-get update -y && apt-get upgrade -y \
 && cat system-dependencies.txt | xargs apt-get install --no-install-recommends -y \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PYTHON_VERSION=3.12
ENV VIRTUAL_ENV=/opt/otbtf/venv
ENV PATH="$VIRTUAL_ENV/bin:/opt/otbtf/bin:$PATH"
ENV PYTHON_SITE_PACKAGES="$VIRTUAL_ENV/lib/python$PYTHON_VERSION/site-packages"
ENV LD_LIBRARY_PATH=/opt/otbtf/lib

# ----------------------------------------------------------------------------
# Tmp builder stage - dangling cache should persist until "docker builder prune"
FROM base-stage AS build-stage
# A smaller value may be used to limit bazel or to avoid OOM errors while building OTB
ARG CPU_RATIO=1

### Python venv and packages
RUN virtualenv $VIRTUAL_ENV
RUN pip install --no-cache-dir -U pip wheel
# Numpy 2 support in TF is planned for 2.18, but isn't supported by most libraries for now
ARG NUMPY="1.26.4"
RUN pip install --no-cache-dir -U mock six future tqdm deprecated numpy==$NUMPY packaging requests \
 && pip install --no-cache-dir --no-deps keras_applications keras_preprocessing

### TensorFlow
WORKDIR /src/tf

# Clang + LLVM
RUN apt-get update \
 && apt-get install --no-install-recommends -y clang-18 llvm-18 libomp-18-dev lld-18 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

ARG TF=v2.18.0-rc2
ARG WITH_CUDA=false
ARG WITH_MKL=false
ARG WITH_XLA=true

RUN mkdir -p /opt/otbtf/bin /opt/otbtf/lib /opt/otbtf/include

# Install bazelisk (will read .bazelversion and download the right bazel binary - latest by default)
RUN wget -qO /opt/otbtf/bin/bazelisk https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
 && chmod +x /opt/otbtf/bin/bazelisk \
 && ln -s /opt/otbtf/bin/bazelisk /opt/otbtf/bin/bazel

ARG BZL_TARGETS="//tensorflow:libtensorflow_cc.so //tensorflow/tools/pip_package:wheel"
# You can use --build-arg BZL_OPTIONS="--remote_cache=http://..." at build time
ARG BZL_OPTIONS

# Build and install TF wheels
ARG ZIP_COMP_FILES=false
RUN git config --global advice.detachedHead false
RUN git clone --single-branch -b $TF https://github.com/tensorflow/tensorflow.git \
 && cd tensorflow \
 && export TMP=/tmp/bazel \
 && export PYTHON_BIN_PATH=$(which python) \
 && export PYTHON_LIB_PATH=$PYTHON_SITE_PACKAGES \
 && export TF_PYTHON_VERSION=$PYTHON_VERSION \
 && export BZL_CONFIGS="--config=release_cpu_linux" \
 && ( ! $WITH_CUDA || export BZL_CONFIGS="--config=release_gpu_linux --config=cuda_clang --config=cuda_wheel" ) \
 && ( ! $WITH_MKL || export BZL_CONFIGS="$BZL_CONFIGS --config=mkl" ) \
 && ( ! $WITH_XLA || export BZL_CONFIGS="$BZL_CONFIGS --config=xla" ) \
 && BZL_CMD="build $BZL_TARGETS $BZL_CONFIGS --verbose_failures $BZL_OPTIONS" \
 && echo "Build env:" && env \
 && echo "Starting build with cmd: \"bazel $BZL_CMD\"" \
 && bazel $BZL_CMD --jobs="HOST_CPUS*$CPU_RATIO" \
 && pip install --no-cache-dir ./bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow*.whl \
 && ln -s $(find /opt/otbtf/venv -type d -wholename "*/tensorflow/include") /opt/otbtf/include/tf \
 && cp tensorflow/cc/saved_model/tag_constants.h tensorflow/cc/saved_model/signature_constants.h /opt/otbtf/include/tf/tensorflow/cc/saved_model/ \
 && for f in $(find -L /opt/otbtf/include/tf -wholename "*/external/*/*.so"); do ln -s $f /opt/otbtf/lib/; done \
 && ( ! $ZIP_COMP_FILES || zip -9 -j --symlinks /opt/otbtf/tf-$TF.zip tensorflow/cc/saved_model/tag_constants.h tensorflow/cc/saved_model/signature_constants.h bazel-bin/tensorflow/libtensorflow_cc.so* bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow*.whl ) \
 && rm -rf bazel-* /src/tf /root/.cache/ /tmp/*

### OTB
WORKDIR /src/otb

ARG OTB=release-9.1
ARG OTBTESTS=false
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

# SuperBuild OTB
RUN apt-get update -y \
 && apt-get install --reinstall ca-certificates -y \
 && update-ca-certificates \
 && git clone https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb.git \
 && cd otb && git checkout $OTB

# This is a dirty hack for release 4.0.0alpha
# We have to wait that OTB moves from C++14 to C++17
# See https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/issues/2338
RUN cd /src/otb/otb \
 && sed -i 's/CMAKE_CXX_STANDARD 14/CMAKE_CXX_STANDARD 17/g' CMakeLists.txt \
 && echo "" > Modules/Core/ImageManipulation/test/CMakeLists.txt \
 && echo "" > Modules/Core/Conversion/test/CMakeLists.txt \
 && echo "" > Modules/Core/Indices/test/CMakeLists.txt \
 && echo "" > Modules/Core/Edge/test/CMakeLists.txt \
 && echo "" > Modules/Core/ImageBase/test/CMakeLists.txt \
 && echo "" > Modules/Learning/DempsterShafer/test/CMakeLists.txt \
 && cd .. \
 && mkdir -p build \
 && cd build \
 && cmake ../otb/SuperBuild \
     -DCMAKE_INSTALL_PREFIX=/opt/otbtf \
     -DOTB_BUILD_FeaturesExtraction=ON \
     -DOTB_BUILD_Hyperspectral=ON \
     -DOTB_BUILD_Learning=ON \
     -DOTB_BUILD_Miscellaneous=ON \
     -DOTB_BUILD_RemoteModules=ON \
     -DOTB_BUILD_SAR=ON \
     -DOTB_BUILD_Segmentation=ON \
     -DOTB_BUILD_StereoProcessing=ON \
     $($OTBTESTS && echo "-DBUILD_TESTING=ON") \
 && make -j $(python -c "import os; print(round( os.cpu_count() * $CPU_RATIO ))")

# Rebuild OTB with OTBTF module
COPY . /src/otbtf
RUN ln -s /src/otbtf /src/otb/otb/Modules/Remote/otbtf

ARG KEEP_SRC_OTB=false
RUN cd /src/otb/build/OTB/build \
 && cmake /src/otb/otb \
      -DCMAKE_INSTALL_PREFIX=/opt/otbtf \
      -DOTB_WRAP_PYTHON=ON \
      -DPYTHON_EXECUTABLE=$(wich python) \
      -DOTB_USE_TENSORFLOW=ON \
      -DModule_OTBTensorflow=ON \
      -Dtensorflow_include_dir=/opt/otbtf/include/tf \
      -DTENSORFLOW_CC_LIB=$PYTHON_SITE_PACKAGES/tensorflow/libtensorflow_cc.so.2 \
      -DTENSORFLOW_FRAMEWORK_LIB=$PYTHON_SITE_PACKAGES/tensorflow/libtensorflow_framework.so.2 \
 && make install -j $(python -c "import os; print(round( os.cpu_count() * $CPU_RATIO ))") \
 && ( $KEEP_SRC_OTB || rm -rf /src/otb ) \
 && rm -rf /root/.cache /tmp/*

# Install OTBTF python lib
RUN pip install -e /src/otbtf

# Symlink executable python files in PATH
RUN for f in /src/otbtf/python/*.py; do if [ -x $f ]; then ln -s $f /opt/otbtf/bin/; fi; done

# ----------------------------------------------------------------------------
# Final stage from a clean base
FROM base-stage AS final-stage
LABEL maintainer="Remi Cresson <remi.cresson[at]inrae[dot]fr>"

# System-wide ENV
ENV OTB_INSTALL_DIR=/opt/otbtf
ENV PYTHONPATH=/opt/otbtf/lib/otb/python
ENV OTB_APPLICATION_PATH=/opt/otbtf/lib/otb/applications

# Add a standard user - this won't prevent ownership issues with volumes if you're not UID 1000
RUN useradd -s /bin/bash -m otbuser

# Copy built files from intermediate stage
COPY --from=build-stage --chown=otbuser:otbuser /opt/otbtf /opt/otbtf
COPY --from=build-stage --chown=otbuser:otbuser /src /src

# Admin rights without password (not recommended, use `docker run -u root` instead)
ARG SUDO=false
RUN if $SUDO; then usermod -a -G sudo otbuser && echo "otbuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers; fi

# Default user, directory and command (bash will be the default entrypoint)
WORKDIR /home/otbuser
USER otbuser

# Test python imports
RUN python -c "import tensorflow"
RUN python -c "import otbtf, tricks"
RUN python -c "import otbApplication as otb; otb.Registry.CreateApplication('ImageClassifierFromDeepFeatures')"
