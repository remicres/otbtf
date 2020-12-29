# Docker multi-stage build with external bazel cache
Docker build has to be called from the root of the repository (i.e. `docker build .` or `bash tools/docker/multibuild.sh`).  
You may build a custom docker image using `--build-arg` and [build-env-tf.sh](build-env-tf.sh), to add more Ubuntu packages see [build-deps-cli.txt](build-deps-cli.txt).  
In order to set persistent environment variables you'll need to edit the [Dockerfile](../../Dockerfile).  


## Bazel remote cache daemon
This will prevent rebuilding TF for each docker build --network='host', even if docker cache was purged (after `docker system prune`).  
In order to use cache, the bazel cmd (and env) has to be exactly the same, any change in [build-env-tf.sh](build-env-tf.sh) and `--build-arg` (if related to bazel env) will result in a complete new build.  
You'll need to add ` --network='host'` to the docker build command in order to see localhost ports (the other way is a virtual docker network / bridge with a different IP).  

```bash
mkdir -p $HOME/.cache/bazel
docker run --detach -u 1000:1000 -v $HOME/.cache/bazel:/data -p 9090:8080 buchgr/bazel-remote-cache --max_size=20
```
Here with max 20GB but 12GB should be enough in order to save both GPU and CPU build artifacts.  
The cache should persist in your .cache/bazel folder next to the local (host) bazel cache if it exists but they won't be merged (I don't know if it is possible, but you could just use a dummy remote cache the same way, if you need to compile from host instead of docker).  


## Default arguments
```
BASE_IMG        (mandatory)
CPU_RATIO=0.95
GUI=false
BAZEL=3.1.0
TF=r2.4
PROTOBUF=3.9.2
OTB=release-7.2
USE_MKL=true
BZL_TARGETS="//tensorflow:libtensorflow_cc.so //tensorflow:libtensorflow_framework.so //tensorflow/tools/pip_package:build_pip_package"
BZL_CONFIG="--config=opt --config=nogcp --config=noaws --config=nohdfs"
BZL_OPTIONS="--compilation_mode opt --verbose_failures --remote_cache=http://localhost:9090"
KEEP_SRC_TF=false
KEEP_SRC_OTB=false
SUDO=true
```


## Build examples
```bash
# Build for CPU using default Dockerfiles args (without AWS, HDFS and GCP support)
docker build --network='host' -t otbtf:cpu --build-arg BASE_IMG=ubuntu:20.04 .

# Clear bazel config var 
docker build --network='host' -t otbtf:cpu-dev --build-arg BASE_IMG=ubuntu:20.04 --build-arg BZL_CONFIG="" KEEP_SRC_OTB=true .

# Build with latest CUDA
docker build --network='host' -t otbtf:cpu --build-arg BASE_IMG=nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04 .

# Manage versions
docker build --network='host' -t otbtf:oldstable-gpu --build-arg BASE_IMG=nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 \
    --build-arg TF=r2.1 --build-arg BAZEL=0.29.1 --build-arg PROTOBUF=3.8.0 --build-arg OTB=release-7.1 \
    --build-arg BAZEL_OPTIONS="--noincompatible_do_not_split_linking_cmdline -c opt --verbose_failures" .
# In order to build olstable you'll need to modify the Dockerfile to clone the repo at the desired branch 
# instead of copying files from the docker build --network='host' context

# Build with GUI (disabled by default)
docker build --network='host' -t otbtf:cpu-gui --build-arg BASE_IMG=ubuntu:20.04 --build-arg GUI=true .
```

### Debug build
If you failed to build, you can still access the last layer and check CMake errors and logs.  
Run `docker images`, find the latest layer id, and run a tmp container with it (example: `docker run -it d60496d9612e bash`).  
You may also need to split some multi-command layers in the Dockerfile in order to see logs before they get removed.  
If you run into "Out Of Memory errors" during SuperBuild you should decrease CPU_RATIO (ex 0.75).  

## Container examples
```bash
# Enable GUI
docker create -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -it --name otbtf-gui otbtf:cpu-gui
docker start -i otbtf-gui --volume /path/to/mount:/container/path otbtf-gui

# Pull GPU image and create container (requires apt package nvidia-docker2 and latest CUDA driver - now support RTX 30*)
docker create --gpus=all --volume $HOME:/home/otbuser/volume -it --name otbtf-gpu mdl4eo/otbtf2.1:gpu
# Run interactive
docker start -i otbtf-gpu
# Run in background
docker start otbtf-gpu
docker exec otbtf-gpu-cli otbcli_ImageClassifierFromDeepFeatures

# Rebuild OTB with more modules
docker create --gpus=all -it --name otbtf-gpu-dev mdl4eo/otbtf2.1:gpu-dev
docker start -i otbtf-gpu-dev
```
```
    $ sudo -i
    # cd /src/otb/otb/Modules/remote
    # git clone https://my/remote/module.git
    # cd /src/otb/build/OTB/build
    # cmake -DModule_MyRemoteModule=ON /src/otb/otb
    # make install -j
```