#!/bin/bash
# Batch several docker
# See default args and more examples in tools/docker/README.md
RELEASE=2.1
UBUNTU=18.04
CUDA=11.0
CUDNN=8
IMG=ubuntu:$UBUNTU
GPU_IMG=nvidia/cuda:$CUDA-cudnn$CUDNN-devel-ubuntu$UBUNTU
# ubuntu20.04 (python3.8) should work but 11.0-cudnn8-devel-ubuntu20.04 is missing from nvidia's docker repo

# Bazel remote cache daemon
mkdir -p $HOME/.cache/bazel-remote
docker run -d -u 1000:1000 -v $HOME/.cache/bazel-remote:/data -p 9090:8080  buchgr/bazel-remote-cache --max_size=20

# CPU (no MKL)
docker build --network='host' -t mdl4eo/otbtf$RELEASE:cpu --build-arg BASE_IMG=$IMG .
docker build --network='host' -t mdl4eo/otbtf$RELEASE:cpu-dev --build-arg BASE_IMG=$IMG --build-arg KEEP_SRC_OTB=true .
# Enable MKL with bazel config flag (tested on CNN : actually slower than a normal CPU build)
#MKL_CONF="--config=nogcp --config=noaws --config=nohdfs --config=mkl --config=opt --copt='-mfpmath=both' --copt='-march=native'"
#docker build --network='host' -t mdl4eo/otbtf$RELEASE:cpu --build-arg BASE_IMG=$IMG --build-arg BZL_CONFIG="$MKL_CONF" .
# Keep OTB src and build files in order to rebuild with other modules
#docker build --network='host' -t mdl4eo/otbtf$RELEASE:cpu-dev --build-arg BASE_IMG=$IMG --build-arg BZL_CONFIG="$MKL_CONF" --build-arg KEEP_SRC_OTB=true .

# GPU support is enabled if CUDA is found in /usr/local
docker build --network='host' -t mdl4eo/otbtf$RELEASE:gpu --build-arg BASE_IMG=$GPU_IMG .
docker build --network='host' -t mdl4eo/otbtf$RELEASE:gpu-dev --build-arg BASE_IMG=$GPU_IMG --build-arg KEEP_SRC_OTB=true .

#docker login
docker push mdl4eo/otbtf$RELEASE:cpu
docker push mdl4eo/otbtf$RELEASE:cpu-dev
#docker push mdl4eo/otbtf$RELEASE:cpu-gui

docker push mdl4eo/otbtf$RELEASE:gpu
docker push mdl4eo/otbtf$RELEASE:gpu-dev
#docker push mdl4eo/otbtf$RELEASE:gpu-gui
