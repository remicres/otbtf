#!/bin/bash
# Batch several docker
# See default args and more examples in tools/docker/README.md
RELEASE=2.1
UBUNTU=20.04
CUDA=11.1
CUDNN=8
IMG=ubuntu:$UBUNTU
GPU_IMG=nvidia/cuda:$CUDA-cudnn$CUDNN-devel-ubuntu$UBUNTU

# Bazel remote cache daemon
mkdir -p $HOME/.cache/bazel_remote
docker run -d -u 1000:1000 -v $HOME/.cache/bazel_remote:/data -p 9090:8080  buchgr/bazel-remote-cache --max_size=20

# GPU support is enabled if CUDA is found in /usr/local
docker build --network='host' -t mdl4eo/otbtf$RELEASE:gpu --build-arg BASE_IMG=$GPU_IMG .
# Keep OTB src and build files in order to rebuild with other modules
docker build --network='host' -t mdl4eo/otbtf$RELEASE:gpu-dev --build-arg BASE_IMG=$GPU_IMG --build-arg KEEP_SRC_OTB=true .

# CPU
docker build --network='host' -t mdl4eo/otbtf$RELEASE:cpu --build-arg BASE_IMG=$IMG .
docker build --network='host' -t mdl4eo/otbtf$RELEASE:cpu-dev --build-arg BASE_IMG=$IMG --build-arg KEEP_SRC_OTB=true .
# Enable MKL with bazel config flag
#BZL_CONFIG="--config=opt --config=nogcp --config=noaws --config=nohdfs --config=mkl --copt='-mfpmath=both'"
#docker build --network='host' -t mdl4eo/otbtf$RELEASE:cpu-mkl --build-arg BASE_IMG=$IMG --build-arg BZL_CONFIG=$CONF .

#docker login
docker push mdl4eo/otbtf$RELEASE:cpu
docker push mdl4eo/otbtf$RELEASE:cpu-dev
#docker push mdl4eo/otbtf$RELEASE:cpu-mkl
#docker push mdl4eo/otbtf$RELEASE:cpu-gui

docker push mdl4eo/otbtf$RELEASE:gpu
docker push mdl4eo/otbtf$RELEASE:gpu-dev
#docker push mdl4eo/otbtf$RELEASE:gpu-gui
