# Docker multi-stage build with external bazel cache
Docker build has to be called from the root of the repository (i.e. `docker build .` or `bash tools/docker/multibuild.sh`).  
You may build a custom docker image using `--build-arg` and [build-env-tf.sh](build-env-tf.sh).  
Regarding OTB, you can edit cmake flags in [build-flags-otb.txt](build-flags-otb.txt) and the `OTB` argument for the git branch to clone.  
If you need additional Ubuntu packages see [build-deps-cli.txt](build-deps-cli.txt) and [build-deps-gui.txt](build-deps-gui.txt) for GUI related packages, it is disabled by default in order to save space, and because docker xvfb isn't working properly with opengl.

## Default arguments
```
BASE_IMG        (mandatory)
CPU_RATIO=0.95
GUI=false
BAZEL=3.1.0
TF=r2.4
OTB=release-7.2
BZL_TARGETS="//tensorflow:libtensorflow_cc.so //tensorflow:libtensorflow_framework.so //tensorflow/tools/pip_package:build_pip_package"
BZL_CONFIG="--config=nogcp --config=noaws --config=nohdfs --config=opt"
BZL_OPTIONS="--verbose_failures --remote_cache=http://localhost:9090"
KEEP_SRC_TF=false
KEEP_SRC_OTB=false
SUDO=true
```

## Bazel remote cache daemon
There is no way make a common build of OTB shared between every docker build, since we're using different BASE_IMG and because of the multi-stage Dockerfile.  
But if you just need to rebuild with different GUI or KEEP_SRC arguments, or may be a different branch of OTBTF, bazel cache may help you to rebuild everything except TF, even if docker cache was purged (after `docker system prune`).  
In order to recycle the cache, the bazel cmd (and env) has to be exactly the same, any change in [build-env-tf.sh](build-env-tf.sh) and `--build-arg` (if related to bazel env, cuda, mkl, xla...) may result in a complete new build, except if bazel is overwriting your variables with some --config flags.  

Start a cache - here with max 20GB but 12GB should be enough to save 2 TF builds (GPU, and CPU-MKL):  
```bash
mkdir -p $HOME/.cache/bazel-remote
docker run --detach -u 1000:1000 -v $HOME/.cache/bazel-remote:/data -p 9090:8080 buchgr/bazel-remote-cache --max_size=20
```
The cache should persist in your `.cache/bazel` folder next to the local bazel cache if it exists.  
They won't merge, I don't know if it is possible, but you can make use of a dummy remote cache as well if you want to recompile from host instead of docker).  

Then ust add ` --network='host'` to the docker build command, or connect bazel to another adress with the 'BZL_OPTIONS' build argument (the other way of docker is a virtual bridge).  


## Build examples
```bash
# Build for CPU using default Dockerfiles args (without AWS, HDFS and GCP support)
docker build --network='host' -t otbtf:cpu --build-arg BASE_IMG=ubuntu:18.04 .

# Clear bazel config var (deactivate optimizations and unset noaws/nogcp/nohdfs)
docker build --network='host' -t otbtf:cpu --build-arg BASE_IMG=ubuntu:18.04 --build-arg BZL_CONFIG= .

# Build with latest CUDA
docker build --network='host' -t otbtf:gpu-dev --build-arg BASE_IMG=nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04 --build-arg KEEP_SRC_OTB=true .

# Enable MKL
MKL_CONFIG="--config=nogcp --config=noaws --config=nohdfs --config=mkl --config=opt --copt='-mfpmath=both'"
docker build --network='host' -t otbtf:cpu-mkl --build-arg BZL_CONFIG=$MKL_CONFIG --build-arg BASE_IMG=nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04 .

# Manage versions
docker build --network='host' -t otbtf:oldstable-gpu --build-arg BASE_IMG=nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 \
    --build-arg TF=r2.1 --build-arg BAZEL=0.29.1 --build-arg OTB=release-7.2 \
    --build-arg BAZEL_OPTIONS="--noincompatible_do_not_split_linking_cmdline --verbose_failures" .
# You could edit the Dockerfile to clone an old branch of the repo instead of cp new files from the build context
```

### Debug build
If you fail to build, you can log into the last layer and check CMake logs.  
Run `docker images`, find the latest layer id and run a tmp container (e.g. `docker run -it d60496d9612e bash`).  
You may also need to split some multi-command layers in the Dockerfile.  
If you see OOM errors during SuperBuild you should decrease CPU_RATIO (ex 0.75).  


## Container examples
```bash

# Pull GPU image and create container with mounted home directory
# (requires apt package nvidia-docker2 and CUDA>=11.0)
docker create --gpus=all --volume $HOME:/home/otbuser/volume -it --name otbtf-gpu mdl4eo/otbtf2.1:gpu

# Run interactive
docker start -i otbtf-gpu

# Run in background
docker start otbtf-gpu
docker exec otbtf-gpu python -c 'import tensorflow as tf; print(tf.test.is_gpu_available())'

# Rebuild OTB with more modules (e.g. otbSelectiveHaralickTextures)
docker create --gpus=all -it --name otbtf-gpu-dev mdl4eo/otbtf2.1:gpu-dev
docker start -i otbtf-gpu-dev
```
In the container shell:
```bash
sudo -i
cd /src/otb/otb/Modules/Remote
git clone https://gitlab.irstea.fr/raffaele.gaetano/otbSelectiveHaralickTextures.git
cd /src/otb/build/OTB/build
cmake -DModule_OTBAppSelectiveHaralickTextures=ON /src/otb/otb && make install -j
```


## GUI
```bash
# With GUI (disabled by default): otbgui seems ok but monteverdi (opengl) isn't working
docker build --network='host' -t otbtf:cpu-gui --build-arg BASE_IMG=ubuntu:18.04 --build-arg GUI=true .
docker create -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -it --name otbtf-gui otbtf:cpu-gui
docker start -i otbtf-gui
$ mapla
```
