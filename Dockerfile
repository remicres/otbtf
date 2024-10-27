##### OTBTF configurable Dockerfile with multi-stage build
# ----------------------------------------------------------------------------
# Init base stage - used for intermediate build env and final image

# Freeze ubuntu version to avoid suprise rebuild
FROM ubuntu:jammy-20240911.1 AS base-stage

WORKDIR /tmp

### System packages
ARG DEBIAN_FRONTEND=noninteractive
COPY system-dependencies.txt .
RUN apt-get update -y && apt-get upgrade -y \
 && cat system-dependencies.txt | xargs apt-get install --no-install-recommends -y \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PY=3.10
ENV VIRTUAL_ENV=/opt/otbtf/venv
ENV PATH="$VIRTUAL_ENV/bin:/opt/otbtf/bin:$PATH"
ENV PYTHON_SITE_PACKAGES="$VIRTUAL_ENV/lib/python$PY/site-packages"
ENV LD_LIBRARY_PATH=/opt/otbtf/lib
# A smaller value may be used to limit bazel or to avoid OOM errors while building OTB
ARG CPU_RATIO=1

# ----------------------------------------------------------------------------
### Builder stage: bazel clang tensorflow
FROM base-stage AS tf-build
WORKDIR /src/tf
RUN mkdir -p /opt/otbtf/bin /opt/otbtf/lib /opt/otbtf/include

# Clang + LLVM
ARG LLVM=18

ADD https://apt.llvm.org/llvm.sh llvm.sh
RUN bash ./llvm.sh $LLVM
ENV CC=/usr/bin/clang-$LLVM
ENV CXX=/usr/bin/clang++-$LLVM
ENV BAZEL_COMPILER="/usr/bin/clang-$LLVM"
RUN apt-get update -y && apt-get upgrade -y \
 && apt-get install -y lld-$LLVM libomp-$LLVM-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

### Python venv and packages
RUN virtualenv $VIRTUAL_ENV
RUN pip install --no-cache-dir -U pip wheel
# Numpy 2 support in TF is planned for 2.18, but isn't supported by most libraries for now
ARG NUMPY="1.26.4"
RUN pip install --no-cache-dir -U mock six future tqdm deprecated numpy==$NUMPY packaging requests
 
# TF build arguments
ARG TF=v2.18.0
ARG WITH_CUDA=false
# Custom compute capabilities, else use default one from .bazelrc
ARG CUDA_CC
ARG WITH_XLA=true
ARG WITH_MKL=false

# Install bazelisk: will read .bazelversion and download the right bazel binary
ADD https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64  /opt/otbtf/bin/bazelisk
RUN chmod +x /opt/otbtf/bin/bazelisk && ln -s /opt/otbtf/bin/bazelisk /opt/otbtf/bin/bazel

# Build and install TF wheel
ADD https://github.com/tensorflow/tensorflow.git#$TF tensorflow
ARG BZL_TARGETS="//tensorflow:libtensorflow_cc.so //tensorflow/tools/pip_package:wheel"
# You can use --build-arg BZL_OPTIONS="--remote_cache=http://..." at build time
ARG BZL_OPTIONS
ARG TF_BUILD_ARTIFACTS

# Save local bazel cache with docker mount
RUN --mount=type=cache,target=/root/.cache/bazel \
 cd tensorflow \
 && BZL_CONFIGS="--repo_env=WHEEL_NAME=tensorflow_cpu --config=release_cpu_linux" \
 && if [ "$WITH_CUDA" = "true" ] ; then BZL_CONFIGS="--repo_env=WHEEL_NAME=tensorflow --config=release_gpu_linux" ; fi \
 && if [ -n "$CUDA_CC" ] ; then BZL_CONFIGS="$BZL_CONFIGS --repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES=$CUDA_CC"; fi \
 && if [ "$WITH_MKL" = "true" ] ; then BZL_CONFIGS="$BZL_CONFIGS --config=mkl" ; fi \
 && if [ "$WITH_XLA" = "true" ] ; then BZL_CONFIGS="$BZL_CONFIGS --config=xla" ; fi \
 && export HERMETIC_PYTHON_VERSION=$PY \
 && echo "Build env:" && env \
 && bazel cquery //tensorflow/tools/pip_package:wheel $BZL_CONFIGS $BZL_OPTIONS --output=build \
 && BZL_CMD="build $BZL_TARGETS $BZL_CONFIGS $BZL_OPTIONS --verbose_failures" \
 && echo "Starting build with cmd: \"bazel $BZL_CMD\"" \
 && bazel $BZL_CMD --jobs="HOST_CPUS*$CPU_RATIO" \
 && TF_WHEEL="bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow*.whl" \
 && pip install --no-cache-dir $TF_WHEEL \
 && ln -s $PYTHON_SITE_PACKAGES/tensorflow/include /opt/otbtf/include/tf \
 && for f in $(find -L /opt/otbtf/include/tf -wholename "*/external/*/*.so"); do ln -s $f /opt/otbtf/lib/; done \
 && TF_MISSING_HEADERS="tensorflow/cc/saved_model/tag_constants.h tensorflow/cc/saved_model/signature_constants.h" \
 && cp $TF_MISSING_HEADERS /opt/otbtf/include/tf/tensorflow/cc/saved_model/ \
 && ARTIFACTS="$TF_WHEEL $TF_MISSING_HEADERS bazel-bin/tensorflow/libtensorflow_cc.so*" \
 && if [ -n "$TF_BUILD_ARTIFACTS" ] ; then mkdir -p $TF_BUILD_ARTIFACTS && mv $ARTIFACTS $TF_BUILD_ARTIFACTS ; fi \
 && rm -rf bazel-* /src/tf

# ----------------------------------------------------------------------------
### Builder stage: cmake gcc otb
FROM base-stage AS otb-build
WORKDIR /src/otb

COPY --from=tf-build /opt/otbtf /opt/otbtf

# SuperBuild OTB
ARG OTB=release-9.1
ADD --keep-git-dir=true https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb.git#$OTB otb

# <------------------------------------------
# This is a dirty hack for release 4.0.0alpha
# We have to wait that OTB moves from C++14 to C++17
# See https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/issues/2338
RUN cd otb \
 && sed -i 's/CMAKE_CXX_STANDARD 14/CMAKE_CXX_STANDARD 17/g' CMakeLists.txt \
 && echo "" > Modules/Core/ImageManipulation/test/CMakeLists.txt \
 && echo "" > Modules/Core/Conversion/test/CMakeLists.txt \
 && echo "" > Modules/Core/Indices/test/CMakeLists.txt \
 && echo "" > Modules/Core/Edge/test/CMakeLists.txt \
 && echo "" > Modules/Core/ImageBase/test/CMakeLists.txt \
 && echo "" > Modules/Learning/DempsterShafer/test/CMakeLists.txt \
 && cd .. \
 && mkdir -p build /tmp/SuperBuild-downloads \
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
     -DDOWNLOAD_LOCATION=/tmp/SuperBuild-downloads \
 && make -j $(python -c "import os; print(round( os.cpu_count() * $CPU_RATIO ))") \
 && rm -rf /tmp/SuperBuild-downloads

# Copy cpp and cmake files from build context (TODO: use `COPY --parents` feature when released)
WORKDIR /src/otbtf
COPY app ./app
COPY include ./include
COPY CMakeLists.txt otb-module.cmake ./
RUN mkdir test
COPY test/CMakeLists.txt test/*.cxx test/

# Build OTBTF cpp
ARG DEV_IMAGE=false
RUN ln -s /src/otbtf /src/otb/otb/Modules/Remote/otbtf
RUN cd /src/otb/build/OTB/build \
 && cmake /src/otb/otb \
      -DCMAKE_INSTALL_PREFIX=/opt/otbtf \
      -DOTB_WRAP_PYTHON=ON \
      -DPython_EXECUTABLE=$(which python) \
      -DOTB_USE_TENSORFLOW=ON \
      -DModule_OTBTensorflow=ON \
      -Dtensorflow_include_dir=/opt/otbtf/include/tf \
      -DTENSORFLOW_CC_LIB=$PYTHON_SITE_PACKAGES/tensorflow/libtensorflow_cc.so.2 \
      -DTENSORFLOW_FRAMEWORK_LIB=$PYTHON_SITE_PACKAGES/tensorflow/libtensorflow_framework.so.2 \
      $( [ "$DEV_IMAGE" != "true" ] || echo "-DBUILD_TESTING=ON" ) \
 && make install -j $(python -c "import os; print(round( os.cpu_count() * $CPU_RATIO ))") \
 && ( [ "$DEV_IMAGE" = "true" ] || rm -rf /src/otb ) \
 && rm -rf /root/.cache /tmp/*

# ----------------------------------------------------------------------------
# Final stage: copy binaries from middle layers and install python module
FROM base-stage AS final-stage
LABEL maintainer="Remi Cresson <remi.cresson[at]inrae[dot]fr>"

# System-wide ENV
ENV OTB_INSTALL_DIR=/opt/otbtf
ENV OTB_APPLICATION_PATH=/opt/otbtf/lib/otb/applications
# For otbApplication and osgeo modules
ENV PYTHONPATH="/opt/otbtf/lib/otb/python:/opt/otbtf/lib/python$PY/site-packages"

# Add a standard user - this won't prevent ownership issues with volumes if you're not UID 1000
RUN useradd -s /bin/bash -m otbuser

# Admin rights without password (not recommended, use `docker run -u root` instead)
ARG SUDO=false
RUN ! $SUDO || usermod -a -G sudo otbuser && echo "otbuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Allow user to install packages in prefix /opt/otbtf and venv without being root
COPY --from=otb-build --chown=otbuser:otbuser /opt/otbtf /opt/otbtf
COPY --from=otb-build --chown=otbuser:otbuser /src /src
USER otbuser

# Install OTBTF python module
WORKDIR /src/otbtf
COPY otbtf ./otbtf
COPY tricks ./tricks
COPY README.md setup.py .
RUN pip install -e .

# Install test packages for dev image
RUN ! $DEV_IMAGE || pip install codespell flake8 pylint pytest pytest-cov pytest-order

WORKDIR /home/otbuser

# Test python imports
RUN python -c "import tensorflow ; import keras"
RUN python -c "import otbApplication as otb; otb.Registry.CreateApplication('ImageClassifierFromDeepFeatures')"
RUN python -c "import otbtf ; import tricks"
