#!/usr/bin/bash
# As in official TF wheels, we avoid "-march=native" to prevent MAVX512 compatibility issues
# Here we disable only AVX512 but enable commons optimizations like FMA, SSE4.2 and AVX2
export CC_OPT_FLAGS="--copt=-mfma --copt=-msse4.2 --copt=-mavx --copt=-mavx2"
export PYTHON_BIN_PATH=$(which python3)
export PYTHON_LIB_PATH="$($PYTHON_BIN_PATH -c 'import site; print(site.getsitepackages()[0])')"
# Required variable since TF 2.16
export TF_PYTHON_VERSION="$($PYTHON_BIN_PATH -c 'import sys; print(sys.version[:4])')"

# Disabled features
export TF_NEED_COMPUTECPP=0
export TF_NEED_GDR=0
export TF_NEED_KAFKA=0
export TF_NEED_MPI=0
export TF_NEED_OPENCL=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_VERBS=0
export TF_SET_ANDROID_WORKSPACE=0

if $WITH_CUDA; then
    export BZL_CONFIGS="--config=release_gpu_linux --config=cuda_clang --config=cuda_wheel"
else
    export BZL_CONFIGS="--config=release_cpu_linux"
fi

# Enabled features
export TF_NEED_JEMALLOC=1
if $WITH_XLA; then
    export BZL_CONFIGS="$BZL_CONFIGS --config=mkl"
fi

if $WITH_MKL; then
    export BZL_CONFIGS="$BZL_CONFIGS --config=xla"
fi

echo"Starting build with the following environment variables:"
env
