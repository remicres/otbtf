#!/usr/bin/bash
export PYTHON_BIN_PATH=$(which python3)
export PYTHON_LIB_PATH="$($PYTHON_BIN_PATH -c 'import site; print(site.getsitepackages()[0])')"
# Required variable since TF 2.16
export TF_PYTHON_VERSION="$($PYTHON_BIN_PATH -c 'import sys; print(sys.version[:4])')"

if $WITH_CUDA; then
    export BZL_CONFIGS="--config=release_gpu_linux --config=cuda_clang --config=cuda_wheel"
else
    export BZL_CONFIGS="--config=release_cpu_linux"
fi

if $WITH_XLA; then
    export BZL_CONFIGS="$BZL_CONFIGS --config=xla"
fi

if $WITH_MKL; then
    export BZL_CONFIGS="$BZL_CONFIGS --config=mkl"
fi

echo "Starting build with the following environment variables:"
env
