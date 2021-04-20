# OTBTF docker images overview

### Available images

Here is the list of OTBTF docker images hosted on [dockerhub](https://hub.docker.com/u/mdl4eo).

| Name                        | Os            | TF     | OTB   | Description            |
| --------------------------- | ------------- | ------ | ----- | ---------------------- |
| **mdl4eo/otbtf1.6:cpu**     | Ubuntu Xenial | r1.14  | 7.0.0 | CPU, no optimization   |
| **mdl4eo/otbtf1.7:cpu**     | Ubuntu Xenial | r1.14  | 7.0.0 | CPU, no optimization   |
| **mdl4eo/otbtf1.7:gpu**     | Ubuntu Xenial | r1.14  | 7.0.0 | GPU                    |
| **mdl4eo/otbtf2.0:cpu**     | Ubuntu Xenial | r2.1   | 7.1.0 | CPU, no optimization   |
| **mdl4eo/otbtf2.0:gpu**     | Ubuntu Xenial | r2.1   | 7.1.0 | GPU                    |
| **mdl4eo/otbtf2.4:cpu**     | Ubuntu Focal  | r2.4.1 | 7.2.0 | CPU, no optimization   |
| **mdl4eo/otbtf2.4:cpu-mkl** | Ubuntu Focal  | r2.4.1 | 7.2.0 | CPU, Intel MKL, AVX512 |
| **mdl4eo/otbtf2.4:gpu**     | Ubuntu Focal  | r2.4.1 | 7.2.0 | GPU                    |

- `cpu` tagged docker images are compiled without optimization.
- `gpu` tagged docker images are suited for **NVIDIA GPUs**. They use CUDA/CUDNN support and are built with compute capabilities 5.2, 6.1, 7.0, 7.5. 
- `cpu-mkl` tagged docker image is experimental, it is optimized for Intel CPUs with AVX512 flags.

You can also find plenty of interesting OTBTF flavored images at [LaTelescop gitlab registry](https://gitlab.com/latelescop/docker/otbtf/container_registry/).

### Development ready images

Until r2.4, all images are development-ready. For instance, you can recompile the whole OTB from `/work/otb/build/OTB/build`.
Since r2.4, only `gpu` tagged image is development-ready, and you can recompile OTB from `/src/otb/build/OTB/build`.

### Build your own images

If you want to use optimization flags, change GPUs compute capability, etc. you can build your own docker image using the provided dockerfile. 
See the [docker build documentation](../tools/docker/README.md).

# Mounting file systems

You can mount filesystem in the docker image.
For instance, suppose you have some data in `/mnt/my_device/` that you want to use inside the container:

The following command shows you how to access the folder from the docker image.

```bash
docker run -v /mnt/my_device/:/data/ -ti mdl4eo/otbtf2.4:cpu bash -c "ls /data"
```
Beware of ownership issues! see the last section of this doc.

# GPU enabled docker 

In Linux, this is quite straightforward. 
Just follow the steps described in the [nvidia-docker documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
You can then use the OTBTF `gpu` tagged docker images with the **NVIDIA runtime**:

```bash
docker run --runtime=nvidia -ti mdl4eo/otbtf2.4:gpu bash
```

You can find some details on the **GPU docker image** and some **docker tips and tricks** on [this blog](https://mdl4eo.irstea.fr/2019/10/15/otbtf-docker-image-with-gpu/). 
Be careful though, these infos might be a bit outdated...

# Docker Installation

### Installation and first steps on Windows 10

1. Install [WSL2](https://docs.microsoft.com/en-us/windows/wsl/install-win10#manual-installation-steps) (Windows Subsystem for Linux)
2. Install [docker desktop](https://www.docker.com/products/docker-desktop)
3. Start **docker desktop** and **enable WSL2** from *Settings* > *General* then tick the box *Use the WSL2 based engine*
3. Open a **cmd.exe** or **PowerShell** terminal, and type `docker create --name otbtf-cpu --interactive --tty mdl4eo/otbtf2.4:cpu`
4. Open **docker desktop**, and check that the docker is running in the **Container/Apps** menu
![Docker desktop, after the docker image is downloaded and ready to use](images/docker_desktop_1.jpeg)
5. From **docker desktop**, click on the icon highlighted as shown below, and use the bash terminal that should pop up!
![Click on the icon to run a session](images/docker_desktop_2.jpeg)

Troubleshooting:
- [Docker for windows WSL documentation](https://docs.docker.com/docker-for-windows/wsl)
- [WSL2 installation steps](https://docs.microsoft.com/en-us/windows/wsl/install-win10)

### Use the GPU with Windows 10 + WSL2

*Work in progress*

Some users have reported to use OTBTF with GPU in windows 10 using WSL2. 
How to install WSL2 with Cuda on windows 10:
https://docs.nvidia.com/cuda/wsl-user-guide/index.html
https://docs.docker.com/docker-for-windows/wsl/#gpu-support


### Debian and Ubuntu

See here how to install docker on Ubuntu [here](https://docs.docker.com/engine/install/ubuntu/).

# Docker Usage

This section is largely inspired from the [moringa docker help](https://gitlab.irstea.fr/raffaele.gaetano/moringa/-/raw/develop/docker/README.md). Big thanks to them.

## Useful diagnostic commands

Here are some usefull commands.

```bash
docker info         # System info
docker images       # List local images
docker container ls # List containers
docker ps           # Show running containers
```

On Linux, control state with systemd:
```bash
sudo systemctl {status,enable,disable,start,stop} docker
```

### Run some commands

Run a simple command in a one-shot container:

```bash
docker run mdl4eo/otbtf2.4:cpu otbcli_PatchesExtraction
```

You can also use the image in interactive mode with bash:
```bash
docker run -ti mdl4eo/otbtf2.4:cpu bash
```

### Persistent container

Persistent (named) container with volume, here with home dir, but it can be any directory.
Beware of ownership issues, see the last section of this doc.

```bash
docker create --interactive --tty --volume /home/$USER:/home/otbuser/ \
    --name otbtf mdl4eo/otbtf2.4:cpu /bin/bash
```

### Interactive session

```bash
docker start -i otbtf
```

### Background container

```bash
docker start otbtf
docker exec otbtf ls -alh
docker stop otbtf
```

### Running commands with root user

Background container is the easiest way:

```bash
docker start otbtf
# Example with apt update (you can't use &&, one docker exec is required for each command)
docker exec --user root otbtf apt-get update
docker exec --user root otbtf apt-get upgrade -y
```

### Container-specific commands, especially for background containers:

```bash
docker inspect otbtf         # See full container info dump
docker logs otbtf            # See command logs and outputs
docker stats otbtf           # Real time container statistics
docker {pause,unpause} otbtf # Freeze container
```

### Stop a background container

Don't forget to stop the container after you have done.

```bash
docker stop otbtf
```

### Remove a persistent container

```bash
docker rm otbtf
```

# Fix volume ownership issue (required if host's UID > 1000)

When mounting a volume, you may experience errors while trying to write files from within the container.
Since the default user (**otbuser**) is UID 1000, you won't be able to write files into your volume 
which is mounted with the same UID than your linux host user (may be UID 1001 or more). 
In order to address this, you need to edit the container's user UID and GID to match the right numerical value.
This will only persist in a named container, it is required every time you're creating a new one.


Create a named container (here with your HOME as volume), Docker will automatically pull image

```bash
docker create --interactive --tty --volume /home/$USER:/home/otbuser \
    --name otbtf mdl4eo/otbtf2.4:cpu /bin/bash
```

Start a background container process:

```bash
docker start otbtf
```

Exec required commands with user root (here with host's ID, replace $UID and $GID with desired values):

```bash
docker exec --user root otbtf usermod otbuser -u $UID
docker exec --user root otbtf groupmod otbuser -g $GID
```

Force reset ownership with updated UID and GID. 
Make sure to double check that `docker exec otbtf id` because recursive chown will apply to your volume in `/home/otbuser`

```bash
docker exec --user root otbtf chown -R otbuser:otbuser /home/otbuser
```

Stop the background container and start a new interactive shell:

```bash
docker stop otbtf
docker start -i otbtf
```

Check if ownership is right

```bash
id
ls -Alh /home/otbuser
touch /home/otbuser/test.txt
```
