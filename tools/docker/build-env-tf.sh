#!/usr/bin/bash
# As in official TF wheels, we avoid "-march=native" to prevent MAVX512 compatibility issues
# Here we disable only AVX512 but enable commons optimizations like FMA, SSE4.2 and AVX2
export CC_OPT_FLAGS="--copt=-mfma --copt=-msse4.2 --copt=-mavx --copt=-mavx2"
export PYTHON_BIN_PATH=$(which python3)
export PYTHON_LIB_PATH="$($PYTHON_BIN_PATH -c 'import site; print(site.getsitepackages()[0])')"
# Required variable since TF 2.16
export TF_PYTHON_VERSION="$($PYTHON_BIN_PATH -c 'import sys; print(sys.version[:4])')"
export TF_DOWNLOAD_CLANG=0
export TF_ENABLE_XLA=1
export TF_NEED_COMPUTECPP=0
export TF_NEED_GDR=0
export TF_NEED_JEMALLOC=1
export TF_NEED_KAFKA=0
export TF_NEED_MPI=0
export TF_NEED_OPENCL=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_VERBS=0
export TF_SET_ANDROID_WORKSPACE=0
export TF_NEED_CLANG=1

## GPU
export TF_NEED_ROCM=0
export TF_NEED_CUDA=0
export TF_NEED_TENSORRT=0
export CUDA_TOOLKIT_PATH=$(find /usr/local -maxdepth 1 -type d -name 'cuda-*')
if  [ ! -z $CUDA_TOOLKIT_PATH ] ; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_TOOLKIT_PATH/lib64:$CUDA_TOOLKIT_PATH/lib64/stubs"
    export TF_CUDA_VERSION=$(echo $CUDA_TOOLKIT_PATH | sed -r 's/.*\/cuda-(.*)/\1/')
    # Let TF set compute capabilities
    #export TF_CUDA_COMPUTE_CAPABILITIES="5.2,6.1,7.0,7.5,8.0,8.6,8.9,9.0"
    export TF_NEED_CUDA=1
    export CUDNN_INSTALL_PATH="/usr/"
    export TF_CUDNN_VERSION=$(sed -n 's/^#define CUDNN_MAJOR\s*\(.*\).*/\1/p' $CUDNN_INSTALL_PATH/include/cudnn_version.h)
    export TF_NCCL_VERSION=2
fi

echo "Starting build with the following environment variables:"
env
