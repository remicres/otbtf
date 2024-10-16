##### OTBTF configurable Dockerfile with multi-stage build

# ----------------------------------------------------------------------------
# Init base stage - will be cloned as intermediate build env
FROM ubuntu:22.04 AS base-stage
WORKDIR /tmp

### System packages
COPY apt-dependencies.txt ./
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get upgrade -y \
 && cat apt-dependencies.txt | xargs apt-get install --no-install-recommends -y \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

### Python3 environment
RUN ln -s /usr/bin/python3 /usr/local/bin/python && ln -s /usr/bin/pip3 /usr/local/bin/pip
# Upgrade pip
RUN pip install --no-cache-dir pip --upgrade
# Numpy 2 support in TF is planned for 2.18, but isn't supported by most libraries for now
ARG NUMPY_SPEC="<2"
RUN pip install --no-cache-dir -U wheel mock six future tqdm deprecated "numpy$NUMPY_SPEC" packaging requests \
 && pip install --no-cache-dir --no-deps keras_applications keras_preprocessing

# ----------------------------------------------------------------------------
# Tmp builder stage - dangling cache should persist until "docker builder prune"
FROM base-stage AS build-stage
# A smaller value may be used to limit bazel or to avoid OOM errors while building OTB
ARG CPU_RATIO=1

# Install Clang+LLVM 18
RUN wget -q https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 18
ENV CC=/usr/bin/clang-18
ENV CXX=/usr/bin/clang++-18
ENV BAZEL_COMPILER=/usr/bin/clang-18
RUN apt-get update -y && apt-get upgrade -y && apt-get install -y libomp-18-dev && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /src/tf /opt/otbtf/bin /opt/otbtf/include /opt/otbtf/lib/python3

### TF
ARG TF=v2.18.0-rc1
ARG WITH_CUDA=false
ARG WITH_MKL=false
ARG WITH_XLA=true

# Install bazelisk (will read .bazelversion and download the right bazel binary - latest by default)
RUN wget -qO /opt/otbtf/bin/bazelisk https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
 && chmod +x /opt/otbtf/bin/bazelisk \
 && ln -s /opt/otbtf/bin/bazelisk /opt/otbtf/bin/bazel

ARG BZL_TARGETS="//tensorflow:libtensorflow_cc.so //tensorflow/tools/pip_package:wheel"
# You could add --remote_cache=http://... here
ARG BZL_OPTIONS="--verbose_failures"

# Build and install TF wheels
WORKDIR /src/tf
ARG ZIP_COMP_FILES=false
RUN git config --global advice.detachedHead false
RUN git clone --single-branch -b $TF https://github.com/tensorflow/tensorflow.git \
 && cd tensorflow \
 && export PATH="$PATH:/opt/otbtf/bin" \
 && export LD_LIBRARY_PATH="/opt/otbtf/lib" \
 && export TMP=/tmp/bazel \
 && export PYTHON_BIN_PATH=$(which python3) \
 && export PYTHON_LIB_PATH=$($PYTHON_BIN_PATH -c 'import site; print(site.getsitepackages()[0])') \
 && export TF_PYTHON_VERSION=$($PYTHON_BIN_PATH -c 'import sys; print(sys.version[:4])') \
 && export BZL_CONFIGS="--config=release_cpu_linux" \
 && ( ! $WITH_CUDA || export BZL_CONFIGS="--config=release_gpu_linux --config=cuda_clang --config=cuda_wheel" ) \
 && ( ! $WITH_MKL || export BZL_CONFIGS="$BZL_CONFIGS --config=mkl" ) \
 && ( ! $WITH_XLA || export BZL_CONFIGS="$BZL_CONFIGS --config=xla" ) \
 && BZL_CMD="build $BZL_TARGETS $BZL_OPTIONS $BZL_CONFIGS" \
 && echo "Build env:" && env \
 && echo "Starting build with cmd: \"bazel $BZL_CMD\"" \
 && bazel $BZL_CMD --jobs="HOST_CPUS*$CPU_RATIO" \
 && pip3 install --no-cache-dir --prefix=/opt/otbtf ./bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow*.whl \
 && ln -s /opt/otbtf/local/lib/python3.*/* /opt/otbtf/lib/python3 \
 && ln -s /opt/otbtf/local/bin/* /opt/otbtf/bin \
 && ln -s $(find /opt/otbtf -type d -wholename "*/dist-packages/tensorflow/include") /opt/otbtf/include/tf \
 && cp tensorflow/cc/saved_model/tag_constants.h tensorflow/cc/saved_model/signature_constants.h /opt/otbtf/include/tf/tensorflow/cc/saved_model/ \
 && for f in $(find -L /opt/otbtf/include/tf -wholename "*/external/*/*.so"); do ln -s $f /opt/otbtf/lib/; done \
 && ( ! $ZIP_COMP_FILES || zip -9 -j --symlinks /opt/otbtf/tf-$TF.zip tensorflow/cc/saved_model/tag_constants.h tensorflow/cc/saved_model/signature_constants.h bazel-bin/tensorflow/libtensorflow_cc.so* bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow*.whl ) \
 && rm -rf bazel-* /src/tf /root/.cache/ /tmp/*

### OTB
ARG OTB=release-9.1
ARG OTBTESTS=false
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

RUN mkdir /src/otb
WORKDIR /src/otb

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
 && export LD_LIBRARY_PATH=/opt/otbtf/lib \
 && export PATH=$PATH:/opt/otbtf/bin \
 && cmake /src/otb/otb \
      -DCMAKE_INSTALL_PREFIX=/opt/otbtf \
      -DOTB_WRAP_PYTHON=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 \
      -DOTB_USE_TENSORFLOW=ON -DModule_OTBTensorflow=ON \
      -Dtensorflow_include_dir=/opt/otbtf/include/tf \
      -DTENSORFLOW_CC_LIB=/opt/otbtf/local/lib/python3.10/dist-packages/tensorflow/libtensorflow_cc.so.2 \
      -DTENSORFLOW_FRAMEWORK_LIB=/opt/otbtf/local/lib/python3.10/dist-packages/tensorflow/libtensorflow_framework.so.2 \
 && make install -j $(python -c "import os; print(round( os.cpu_count() * $CPU_RATIO ))") \
 && ( $KEEP_SRC_OTB || rm -rf /src/otb ) \
 && rm -rf /root/.cache /tmp/*

# Symlink executable python files in PATH
RUN for f in /src/otbtf/python/*.py; do if [ -x $f ]; then ln -s $f /opt/otbtf/bin/; fi; done

# ----------------------------------------------------------------------------
# Final stage
FROM build-stage AS final-stage
LABEL maintainer="Remi Cresson <remi.cresson[at]inrae[dot]fr>"

# Default user, directory and command (bash will be the default entrypoint)
RUN useradd -s /bin/bash -m otbuser
# Admin rights without password (potential security issue)
ARG SUDO=true
RUN if $SUDO; then usermod -a -G sudo otbuser && echo "otbuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers; fi

# Copy built files from intermediate stage
COPY --from=build-stage --chown=otbuser:otbuser /opt/otbtf /opt/otbtf
COPY --from=build-stage --chown=otbuser:otbuser /src /src

# System-wide ENV
ENV PATH="/opt/otbtf/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/otbtf/lib"
ENV PYTHONPATH="/opt/otbtf/lib/python3/dist-packages:/opt/otbtf/lib/otb/python"
ENV OTB_APPLICATION_PATH="/opt/otbtf/lib/otb/applications"
RUN pip install -e /src/otbtf
WORKDIR /home/otbuser

# Add a standard user - this won't prevent ownership issues with volumes if you're not UID 1000
USER otbuser
# User-only ENV
ENV PATH="/home/otbuser/.local/bin:$PATH"

# Test python imports
RUN python -c "import tensorflow"
RUN python -c "import otbtf, tricks"
RUN python -c "import otbApplication as otb; otb.Registry.CreateApplication('ImageClassifierFromDeepFeatures')"
