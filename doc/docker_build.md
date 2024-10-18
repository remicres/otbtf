# Build your own docker images

Docker build has to be called from the root of the repository (i.e. `docker
 build .`.
You can select target versions using `--build-arg`:

- **TensorFlow** : `TF` arg for the git branch or tag, `ZIP_COMP_FILES`
 allows you to save compiled tf binaries if you want to install it elsewhere.
- **OrfeoToolBox** : `OTB` arg for the git branch or tag, set `KEEP_SRC_OTB`
in order to preserve OTB sources

## Default build arguments

```bash
# Limit CPU usage e.g. 0.75
CPU_RATIO=1
# Can be used to install a specific numpy version
NUMPY="1.26.4"
# Git branch or tag to checkout
TF=v2.18.0
# Build with XLA
WITH_XLA=true
# Build with Intel MKL support
WITH_MKL=false
# Set to true to enable Nvidia GPU support
WITH_CUDA=false
# Custom compute capabilities, default are defined in repo tensorflow/.bazelrc
# Currently "sm_60,sm_70,sm_80,sm_89,compute_90"
CUDA_COMPUTE_CAPABILITIES=
# Targets for bazel build cmd
BZL_TARGETS="//tensorflow:libtensorflow_cc.so //tensorflow/tools/pip_package:wheel"
# Available for additional bazel options, e.g. --remote_cache
BZL_OPTIONS=
# Zip and save tf compiled files in /opt/otbtf, to install elsewhere
ZIP_COMP_FILES=false
# Git branch or tag to checkout
OTB=release-9.1
# Keep OTB sources
KEEP_SRC_OTB=false
# Enable sudo without password for "otbuser"
SUDO=false
```

## Bazel remote cache daemon

If you just need to rebuild with different arguments, or may be a different
 branch of OTB, bazel cache will help you to rebuild everything except TF,
 even if the docker cache was purged (after `docker [system|builder] prune`).
In order to recycle the cache, bazel config and TF git tag should be exactly
 the same, any change in Dockerfile or  `--build-arg` would create a new build.  

Start a cache daemon - 10GB should be enough to save 2 TF builds (GPU and CPU):

```bash
mkdir -p $HOME/.cache/bazel-remote
docker run --detach -u $UID:$GID -v $HOME/.cache/bazel-remote:/data \
  -p 9090:8080 buchgr/bazel-remote-cache --max_size=10
```

Then just add ` --network='host'` to the docker build command, or connect
 bazel to a remote server - see 'BZL_OPTIONS'.  
The other way of docker is a virtual bridge, but you'll need to edit the IP
 address. Changing the BZL_OPTIONS will invalidate docker build cache.

## Images build examples

```bash
# Build for CPU using default Dockerfiles args (without AWS, HDFS or GCP 
# support)
docker build --network='host' -t otbtf:cpu \
  --build-arg BZL_OPTIONS="--remote_cache=http://localhost:9090" .

# Build for GPU
docker build --network='host' -t otbtf:gpu \
  --build-arg BZL_OPTIONS="--remote_cache=http://localhost:9090" \
  --build-arg WITH_CUDA=true \
  .

# Build latest TF and OTB, set git branches/tags to clone
docker build --network='host' -t otbtf:gpu \
   --build-arg BZL_OPTIONS="--remote_cache=http://localhost:9090" \
  --build-arg WITH_CUDA=true \
  --build-arg KEEP_SRC_OTB=true \
  --build-arg TF=nightly \
  --build-arg OTB=develop \
  .
```

### Build for another machine and save TF compiled files

```bash
docker build --network='host' -t otbtf:gpu \
  --build-arg BZL_OPTIONS="--remote_cache=http://localhost:9090" \
  --build-arg WITH_CUDA=true \
  --build-arg ZIP_COMP_FILES=true \
  .

docker run -v $HOME:/home/otbuser/volume otbtf:custom \
  cp /opt/otbtf/tf-v2.18.0.zip /home/otbuser/volume

# Target machine shell
cd $HOME
unzip tf-v2.18.0.zip
sudo mkdir -p /opt/tensorflow/lib
sudo mv tf-v2.18.0/libtensorflow_cc* /opt/tensorflow/lib
# You may need to create a virtualenv, here TF and dependencies are installed 
# next to user's pip packages
pip3 install -U pip wheel mock six future deprecated "numpy<2"
pip3 install --no-deps keras_applications keras_preprocessing
pip3 install tf-v2.18.0/tensorflow-v2.18.0-cp310-cp310-linux_x86_64.whl

TF_WHEEL_DIR="$HOME/.local/lib/python3.10/site-packages/tensorflow"
# If you installed the wheel as regular user, with root pip it should be in 
# /usr/local/lib/python3.*, or in your virtualenv lib/ directory
mv tf-v2.18.0/tag_constants.h $TF_WHEEL_DIR/include/tensorflow/cc/saved_model/
# Then recompile OTB with OTBTF using libraries in /opt/tensorflow/lib and 
# instructions in build_from_sources.md.
cmake $OTB_GIT \
  -DOTB_USE_TENSORFLOW=ON \
  -DModule_OTBTensorflow=ON \
  -DTENSORFLOW_CC_LIB="/opt/tensorflow/lib/libtensorflow_cc.so.2" \
  -Dtensorflow_include_dir="$TF_WHEEL_DIR/include" \
  -DTENSORFLOW_FRAMEWORK_LIB="$TF_WHEEL_DIR/libtensorflow_framework.so.2" \
&& make install -j 
```

### Debug build
If you fail to build, you can log into the last layer and check CMake logs.
 Run `docker images`, find the latest layer ID and run a tmp container
(`docker run -it d60496d9612e bash`).
**This is only possible when building with legacy docker config DOCKER_BUILDKIT=0**.
You may also need to split some multi-command layers in the Dockerfile.
 If you see OOM errors during SuperBuild you should decrease CPU_RATIO.

## Container examples

```bash
# Pull GPU image and create a new container with your home directory as volume 
# (requires apt package nvidia-docker2 and CUDA>=11.0)
docker create --gpus=all --volume $HOME:/home/otbuser/volume -it \
  --name otbtf-gpu mdl4eo/otbtf:4.4.0-gpu

# Run interactive
docker start -i otbtf-gpu

# Run in background
docker start otbtf-gpu
docker exec otbtf-gpu \
  python -c 'import tensorflow as tf; print(tf.test.is_gpu_available())'
```

### Rebuild OTB with more modules

Enter a development ready docker image:

```bash
docker run -it --gpus=all -it --name otbtf-gpu-dev mdl4eo/otbtf:4.4.0-gpu-dev
# Then, from the container shell:
cd /src/otb/otb/Modules/Remote
git clone https://gitlab.irstea.fr/raffaele.gaetano/otbSelectiveHaralickTextures.git
cd /src/otb/build/OTB/build
cmake -DModule_OTBAppSelectiveHaralickTextures=ON /src/otb/otb && make install -j
exit
docker container ls
```

Then you can user `docker commit` to save this container as a new image.
