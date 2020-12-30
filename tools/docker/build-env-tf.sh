# TF - bazel build env variables

export PYTHON_BIN_PATH=$(which python)
export PYTHON_LIB_PATH="$($PYTHON_BIN_PATH -c 'import site; print(site.getsitepackages()[0])')"
export TF_ENABLE_XLA=1
export TF_NEED_CUDA=0
export TF_NEED_MPI=0
export TF_NEED_GDR=0
export TF_NEED_S3=0
export TF_NEED_KAFKA=0
export TF_NEED_OPENCL=0
export TF_NEED_JEMALLOC=1
export TF_NEED_VERBS=0
export TF_NEED_OPENCL_SYCL=0
export TF_SET_ANDROID_WORKSPACE=0
export TF_NEED_COMPUTECPP=0
export GCC_HOST_COMPILER_PATH=$(which gcc)
# You could use BZL_CONFIG=" --config=nogcp --config=noaws --config=nohdfs"
export TF_NEED_AWS=0
export TF_NEED_GCP=0
export TF_NEED_HDFS=0
# For MKL support, edit BZL_CONFIG: --config=mkl --copt='-mfpmath=both'
#export TF_DOWNLOAD_MKL=1
export TF_NEED_MKL=0
# Optimization is controlled with bazel -c opt
#export CC_OPT_FLAGS="-march=native"

export CUDA_TOOLKIT_PATH=$(find /usr/local -maxdepth 1 -type d -name 'cuda-*')
if  [ ! -z $CUDA_TOOLKIT_PATH ] ; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_TOOLKIT_PATH/lib64:$CUDA_TOOLKIT_PATH/lib64/stubs"
    export TF_CUDA_VERSION=$(echo $CUDA_TOOLKIT_PATH | sed -r 's/.*\/cuda-([0-9]{2}\.[0-9])/\1/')
    export TF_CUDA_CLANG=0
    export TF_CUDA_COMPUTE_CAPABILITIES="8.6,7.5,6.1,5.2"
    export TF_NEED_CUDA=1
    export TF_NEED_TENSORRT=0
    export CUDNN_INSTALL_PATH="/usr/"
    export TF_CUDNN_VERSION=$(sed -n 's/^#define CUDNN_MAJOR\s*\(.*\).*/\1/p' $CUDNN_INSTALL_PATH/include/cudnn.h)
    export TF_NCCL_VERSION=2
fi
