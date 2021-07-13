#!/bin/bash
### Docker multibuild and push, see default args and more examples in tools/docker/README.md
RELEASE=2.5
UBUNTU=20.04
CUDA=11.2.2
CUDNN=8
IMG=ubuntu:$UBUNTU
GPU_IMG=nvidia/cuda:$CUDA-cudnn$CUDNN-devel-ubuntu$UBUNTU

## Bazel remote cache daemon
mkdir -p $HOME/.cache/bazel-remote
docker run -d -u 1000:1000 -v $HOME/.cache/bazel-remote:/data -p 9090:8080  buchgr/bazel-remote-cache --max_size=20

### CPU (no MKL)
docker build --network='host' -t mdl4eo/otbtf$RELEASE:cpu-dev --build-arg BASE_IMG=$IMG --build-arg KEEP_SRC_OTB=true .
docker build --network='host' -t mdl4eo/otbtf$RELEASE:cpu --build-arg BASE_IMG=$IMG .
#docker build --network='host' -t mdl4eo/otbtf$RELEASE:-cpu-gui --build-arg BASE_IMG=$IMG --build-arg GUI=true .

### MKL is enabled with bazel config flag
#MKL_CONF="--config=nogcp --config=noaws --config=nohdfs --config=mkl --config=opt"
#docker build --network='host' -t mdl4eo/otbtf$RELEASE:-cpu-mkl --build-arg BASE_IMG=$IMG --build-arg BZL_CONFIGS="$MKL_CONF" .
#docker build --network='host' -t mdl4eo/otbtf$RELEASE:-cpu-mkl-dev --build-arg BASE_IMG=$IMG --build-arg BZL_CONFIGS="$MKL_CONF" --build-arg KEEP_SRC_OTB=true .

### GPU support is enabled if CUDA is found in /usr/local
docker build --network='host' -t mdl4eo/otbtf$RELEASE:gpu-dev --build-arg BASE_IMG=$GPU_IMG --build-arg KEEP_SRC_OTB=true .
docker build --network='host' -t mdl4eo/otbtf$RELEASE:gpu --build-arg BASE_IMG=$GPU_IMG .
#docker build --network='host' -t mdl4eo/otbtf$RELEASE:-gpu-gui --build-arg BASE_IMG=$GPU_IMG --build-arg GUI=true .

#docker login
docker push mdl4eo/otbtf$RELEASE:-cpu-dev
docker push mdl4eo/otbtf$RELEASE:-cpu
#docker push mdl4eo/otbtf$RELEASE:-cpu-gui
#docker push mdl4eo/otbtf$RELEASE:-cpu-mkl

docker push mdl4eo/otbtf$RELEASE:-gpu-dev
docker push mdl4eo/otbtf$RELEASE:-gpu
#docker push mdl4eo/otbtf$RELEASE:-gpu-gui
