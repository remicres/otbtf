# Build with Docker
Docker build has to be called from the root of the repository (i.e. `docker build .` or `bash tools/docker/multibuild.sh`).  
You can build a custom image using `--build-arg` and several config files :
- Ubuntu : `BASE_IMG` should accept any version, for additional packages see [build-deps-cli.txt](build-deps-cli.txt) and [build-deps-gui.txt](build-deps-gui.txt)
- TensorFlow : `TF` arg for the git branch or tag + [build-env-tf.sh](build-env-tf.sh) and BZL_* arguments for the build configuration
- OrfeoToolBox : `OTB` arg for the git branch or tag + [build-flags-otb.txt](build-flags-otb.txt) to edit cmake flags

### Base images
```bash
UBUNTU=20.04            # or 16.04, 18.04
CUDA=11.0.3             # or 10.1, 10.2
CUDNN=8                 # or 7
IMG=ubuntu:$UBUNTU
GPU_IMG=nvidia/cuda:$CUDA-cudnn$CUDNN-devel-ubuntu$UBUNTU
```

### Default arguments
```bash
BASE_IMG                # mandatory
CPU_RATIO=0.95
GUI=false
NUMPY_SPEC="~=1.19"
TF=r2.4.1
OTB=7.2.0
BZL_TARGETS="//tensorflow:libtensorflow_cc.so //tensorflow/tools/pip_package:build_pip_package"
BZL_CONFIGS="--config=nogcp --config=noaws --config=nohdfs --config=opt"
BZL_OPTIONS="--verbose_failures --remote_cache=http://localhost:9090"
KEEP_SRC_TF=false
KEEP_SRC_OTB=false
SUDO=true

# NumPy version requirement :
# TF <  2.4 : "numpy<1.19.0,>=1.16.0"
# TF >= 2.4 : "numpy~=1.19"
```

### Bazel remote cache daemon
If you just need to rebuild with different GUI or KEEP_SRC arguments, or may be a different branch of OTB, bazel cache will help you to rebuild everything except TF, even if the docker cache was purged (after `docker [system|builder] prune`).  
In order to recycle the cache, bazel config and TF git tag should be exactly the same, any change in [build-env-tf.sh](build-env-tf.sh) and `--build-arg` (if related to bazel env, cuda, mkl, xla...) may result in a fresh new build.  

Start a cache daemon - here with max 20GB but 12GB should be enough to save 2 TF builds (GPU and CPU):  
```bash
mkdir -p $HOME/.cache/bazel-remote
docker run --detach -u 1000:1000 -v $HOME/.cache/bazel-remote:/data -p 9090:8080 buchgr/bazel-remote-cache --max_size=20
```
Then just add ` --network='host'` to the docker build command, or connect bazel to a remote server - see 'BZL_OPTIONS'.  
The other way of docker is a virtual bridge, but you'll need to edit the IP address.  

## Build examples
```bash
# Build for CPU using default Dockerfiles args (without AWS, HDFS or GCP support)
docker build --network='host' -t otbtf:cpu --build-arg BASE_IMG=ubuntu:20.04 .

# Clear bazel config var (deactivate default optimizations and unset noaws/nogcp/nohdfs)
docker build --network='host' -t otbtf:cpu --build-arg BASE_IMG=ubuntu:20.04 --build-arg BZL_CONFIGS= .

# Enable MKL
MKL_CONFIG="--config=nogcp --config=noaws --config=nohdfs --config=opt --config=mkl"
docker build --network='host' -t otbtf:cpu-mkl --build-arg BZL_CONFIGS="$MKL_CONFIG" --build-arg BASE_IMG=ubuntu:20.04 .

# Build for GPU (if you're building for your system only you should edit CUDA_COMPUTE_CAPABILITIES in build-env-tf.sh)
docker build --network='host' -t otbtf:gpu --build-arg BASE_IMG=nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04 .

# Build dev with TF and OTB sources (huge image) + set git branches/tags to clone
docker build --network='host' -t otbtf:gpu-dev-full --build-arg BASE_IMG=nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04 \
    --build-arg KEEP_SRC_OTB=true --build-arg KEEP_SRC_TF=true --build-arg TF=nightly --build-arg OTB=develop .

# Build old release
docker build --network='host' -t otbtf:oldstable-gpu --build-arg BASE_IMG=nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 \
    --build-arg TF=r2.1 --build-arg NUMPY_SPEC="<1.19" \
    --build-arg BAZEL_OPTIONS="--noincompatible_do_not_split_linking_cmdline --verbose_failures --remote_cache=http://localhost:9090" .
# You could edit the Dockerfile in order to clone an old branch of the repo instead of copying files from the build context
```

### Debug build
If you fail to build, you can log into the last layer and check CMake logs. Run `docker images`, find the latest layer ID and run a tmp container (`docker run -it d60496d9612e bash`).  
You may also need to split some multi-command layers in the Dockerfile.  
If you see OOM errors during SuperBuild you should decrease CPU_RATIO (e.g. 0.75).  

## Container examples
```bash
# Pull GPU image and create a new container with your home directory as volume (requires apt package nvidia-docker2 and CUDA>=11.0)
docker create --gpus=all --volume $HOME:/home/otbuser/volume -it --name otbtf-gpu mdl4eo/otbtf2.1:gpu

# Run interactive
docker start -i otbtf-gpu

# Run in background
docker start otbtf-gpu
docker exec otbtf-gpu python -c 'import tensorflow as tf; print(tf.test.is_gpu_available())'
```

### Rebuild OTB with more modules
```bash
docker create --gpus=all -it --name otbtf-gpu-dev mdl4eo/otbtf2.1:gpu-dev
docker start -i otbtf-gpu-dev
```
```bash
# From the container shell:
sudo -i
cd /src/otb/otb/Modules/Remote
git clone https://gitlab.irstea.fr/raffaele.gaetano/otbSelectiveHaralickTextures.git
cd /src/otb/build/OTB/build
cmake -DModule_OTBAppSelectiveHaralickTextures=ON /src/otb/otb && make install -j
```

### Container with GUI
```bash
# GUI is disabled by default in order to save space, and because docker xvfb isn't working properly with OpenGL.
# => otbgui seems OK but monteverdi isn't working
docker build --network='host' -t otbtf:cpu-gui --build-arg BASE_IMG=ubuntu:20.04 --build-arg GUI=true .
docker create -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -it --name otbtf-gui otbtf:cpu-gui
docker start -i otbtf-gui
$ mapla
```

### Common errors
Buid :  
`Error response from daemon: manifest for nvidia/cuda:11.0-cudnn8-devel-ubuntu20.04 not found: manifest unknown: manifest unknown`  
=> Image is missing from dockerhub

Run :  
`failed call to cuInit: UNKNOWN ERROR (303) / no NVIDIA GPU device is present: /dev/nvidia0 does not exist`  
=> Nvidia driver is missing or disabled, make sure to add ` --gpus=all` to your docker run or create command
