#!/bin/bash
# Various docker builds using bazel cache
RELEASE=3.5
CPU_IMG=ubuntu:22.04
GPU_IMG=nvidia/cuda:12.1.0-devel-ubuntu22.04

## Bazel remote cache daemon
mkdir -p $HOME/.cache/bazel-remote
docker run -d -u 1000:1000 \
-v $HOME/.cache/bazel-remote:/data \
-p 9090:8080 \
buchgr/bazel-remote-cache --max_size=20

### CPU images

# CPU-Dev
docker build . \
--network='host' \
-t mdl4eo/otbtf:$RELEASE-cpu-dev \
--build-arg BASE_IMG=$CPU_IMG \
--build-arg KEEP_SRC_OTB=true

# CPU
docker build . \
--network='host' \
-t mdl4eo/otbtf:$RELEASE-cpu \
--build-arg BASE_IMG=$CPU_IMG

# CPU-GUI
docker build . \
--network='host' \
-t mdl4eo/otbtf:$RELEASE-cpu-gui \
--build-arg BASE_IMG=$CPU_IMG \
--build-arg GUI=true

### CPU images with Intel MKL support
MKL_CONF="--config=nogcp --config=noaws --config=nohdfs --config=mkl --config=opt"

# CPU-MKL
docker build . \
--network='host' \
-t mdl4eo/otbtf:$RELEASE-cpu-mkl \
--build-arg BASE_IMG=$CPU_IMG \
--build-arg BZL_CONFIGS="$MKL_CONF"

# CPU-MKL-Dev
docker build . \
--network='host' \
-t mdl4eo/otbtf:$RELEASE-cpu-mkl-dev \
--build-arg BASE_IMG=$CPU_IMG \
--build-arg BZL_CONFIGS="$MKL_CONF" \
--build-arg KEEP_SRC_OTB=true

### GPU enabled images
# Support is enabled if CUDA is found in /usr/local

# GPU
docker build . \
--network='host' \
-t mdl4eo/otbtf:$RELEASE-gpu-dev \
--build-arg BASE_IMG=$GPU_IMG \
--build-arg KEEP_SRC_OTB=true

# GPU-Dev
docker build . \
--network='host' \
-t mdl4eo/otbtf:$RELEASE-gpu \
--build-arg BASE_IMG=$GPU_IMG

# GPU-GUI
docker build . \
--network='host' \
-t mdl4eo/otbtf:$RELEASE-gpu-gui \
--build-arg BASE_IMG=$GPU_IMG \
--build-arg GUI=true
