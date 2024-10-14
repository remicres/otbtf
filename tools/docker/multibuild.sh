#!/bin/bash
# Various docker builds using bazel cache
RELEASE=4.3.2
CPU_IMG=ubuntu:22.04
GPU_IMG=nvidia/cuda:12.1.0-devel-ubuntu22.04

## Bazel remote cache daemon
mkdir -p $HOME/.cache/bazel-remote
docker run -d -u 1000:1000 \
-v $HOME/.cache/bazel-remote:/data \
-p 9090:8080 \
buchgr/bazel-remote-cache --max_size=20

### CPU images

# CPU
docker build . \
--network='host' \
-t mdl4eo/otbtf:$RELEASE-cpu \
--build-arg BASE_IMG=$CPU_IMG \
--build-arg BZL_OPTIONS="--verbose_failures --remote_cache=http://localhost:9090" \

# CPU-Dev
docker build . \
--network='host' \
-t mdl4eo/otbtf:$RELEASE-cpu-dev \
--build-arg BASE_IMG=$CPU_IMG \
--build-arg BZL_OPTIONS="--verbose_failures --remote_cache=http://localhost:9090" \
--build-arg KEEP_SRC_OTB=true

### GPU enabled images
# Support is enabled if CUDA is found in /usr/local

# GPU
docker build . \
--network='host' \
-t mdl4eo/otbtf:$RELEASE-gpu-dev \
--build-arg BASE_IMG=$GPU_IMG \
--build-arg BZL_OPTIONS="--verbose_failures --remote_cache=http://localhost:9090" \
--build-arg KEEP_SRC_OTB=true

# GPU-Dev
docker build . \
--network='host' \
-t mdl4eo/otbtf:$RELEASE-gpu \
--build-arg BZL_OPTIONS="--verbose_failures --remote_cache=http://localhost:9090" \
--build-arg BASE_IMG=$GPU_IMG \
