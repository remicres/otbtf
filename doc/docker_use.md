# Install from docker

We recommend to use OTBTF from official docker images.

Latest CPU-only docker image:

```commandline
docker pull mdl4eo/otbtf:latest
```

Latest GPU-ready docker image:

```commandline
docker pull mdl4eo/otbtf:latest-gpu
```

Read more in the following sections.

## Latest images

Here is the list of the latest OTBTF docker images hosted on 
[dockerhub](https://hub.docker.com/u/mdl4eo).
Since OTBTF >= 3.2.1 you can find the latest docker images on 
[gitlab.irstea.fr](https://gitlab.irstea.fr/remi.cresson/otbtf/container_registry).

| Name                                                                               | Os            | TF    | OTB   | Description            | Dev files | Compute capability |
|------------------------------------------------------------------------------------| ------------- |-------|-------| ---------------------- | --------- | ------------------ |
| **mdl4eo/otbtf:4.2.2-cpu**                                                         | Ubuntu Jammy  | r2.12 | d74ab | CPU, no optimization   | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:4.2.2-cpu-dev**                                                     | Ubuntu Jammy  | r2.12 | d74ab | CPU, no optimization (dev) |  yes  | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:4.2.2-gpu**                                                         | Ubuntu Jammy  | r2.12 | d74ab | GPU, no optimization   | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:4.2.2-gpu-dev**                                                     | Ubuntu Jammy  | r2.12 | d74ab | GPU, no optimization (dev) | yes   | 5.2,6.1,7.0,7.5,8.6|
| **gitlab.irstea.fr/remi.cresson/otbtf/container_registry/otbtf:4.2.2-gpu-opt**     | Ubuntu Jammy  | r2.12 | d74ab | GPU with opt.          | no        | 5.2,6.1,7.0,7.5,8.6|
| **gitlab.irstea.fr/remi.cresson/otbtf/container_registry/otbtf:4.2.2-gpu-opt-dev** | Ubuntu Jammy  | r2.12 | d74ab | GPU with opt. (dev)    | yes       | 5.2,6.1,7.0,7.5,8.6|

The list of older releases is available [here](#older-images).

!!! Warning

    Until r2.4, all images are development-ready, and the sources are located 
    in `/work/`.
    Since r2.4, development-ready images have the source in `/src/` and are 
    tagged "...-dev".

## GPU enabled docker 

In Linux, this is quite straightforward. 
Just follow the steps described in the 
[nvidia-docker documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
You can then use the OTBTF `gpu` tagged docker images with the **NVIDIA runtime** : 

With Docker version earlier than 19.03 :

```bash
docker run --runtime=nvidia -ti mdl4eo/otbtf:latest-gpu bash
```

With Docker version including and after 19.03 :

```bash
docker run --gpus all -ti mdl4eo/otbtf:latest-gpu bash
```

You can find some details on the **GPU docker image** and some **docker tips 
and tricks** on 
[this blog](https://mdl4eo.irstea.fr/2019/10/15/otbtf-docker-image-with-gpu/). 
Be careful though, these infos might be a bit outdated...

## Docker Installation

This section is a very small insight on the installation of docker on Linux 
and Windows.

### Debian and Ubuntu

See here how to install docker on Ubuntu 
[here](https://docs.docker.com/engine/install/ubuntu/).

### Windows 10

1. Install [WSL2](https://docs.microsoft.com/en-us/windows/wsl/install-win10#manual-installation-steps) (Windows Subsystem for Linux)
2. Install [docker desktop](https://www.docker.com/products/docker-desktop)
3. Start **docker desktop** and **enable WSL2** from *Settings* > *General* then tick the box *Use the WSL2 based engine*
3. Open a **cmd.exe** or **PowerShell** terminal, and type `docker create --name otbtf-cpu --interactive --tty mdl4eo/otbtf:latest`
4. Open **docker desktop**, and check that the docker is running in the **Container/Apps** menu
![Docker desktop, after the docker image is downloaded and ready to use](images/docker_desktop_1.jpeg)
5. From **docker desktop**, click on the icon highlighted as shown below, and use the bash terminal that should pop up!
![Click on the icon to run a session](images/docker_desktop_2.jpeg)

Troubleshooting:

- [Docker for windows WSL documentation](https://docs.docker.com/docker-for-windows/wsl)
- [WSL2 installation steps](https://docs.microsoft.com/en-us/windows/wsl/install-win10)

!!! Info

    Some users have reported to use OTBTF with GPU in windows 10 using WSL2. 
    How to install WSL2 with Cuda on windows 10:

    - [WSL user guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
    - [XSL GPU support](https://docs.docker.com/docker-for-windows/wsl/#gpu-support)


## Build your own images

If you want to use optimization flags, change GPUs compute capability, etc. 
you can build your own docker image using the provided dockerfile. 
See the [docker build documentation](docker_build.html).

## Older images

Here you can find the list of older releases of OTBTF:

| Name                                                                               | Os            | TF     | OTB   | Description            | Dev files | Compute capability |
|------------------------------------------------------------------------------------| ------------- | ------ |-------| ---------------------- | --------- | ------------------ |
| **mdl4eo/otbtf:1.6-cpu**                                                           | Ubuntu Xenial | r1.14  | 7.0.0 | CPU, no optimization   | yes       | 5.2,6.1,7.0        |
| **mdl4eo/otbtf:1.7-cpu**                                                           | Ubuntu Xenial | r1.14  | 7.0.0 | CPU, no optimization   | yes       | 5.2,6.1,7.0        |
| **mdl4eo/otbtf:1.7-gpu**                                                           | Ubuntu Xenial | r1.14  | 7.0.0 | GPU                    | yes       | 5.2,6.1,7.0        |
| **mdl4eo/otbtf:2.0-cpu**                                                           | Ubuntu Xenial | r2.1   | 7.1.0 | CPU, no optimization   | yes       | 5.2,6.1,7.0,7.5    |
| **mdl4eo/otbtf:2.0-gpu**                                                           | Ubuntu Xenial | r2.1   | 7.1.0 | GPU                    | yes       | 5.2,6.1,7.0,7.5    |
| **mdl4eo/otbtf:2.4-cpu**                                                           | Ubuntu Focal  | r2.4.1 | 7.2.0 | CPU, no optimization   | yes       | 5.2,6.1,7.0,7.5    |
| **mdl4eo/otbtf:2.4-cpu-opt**                                                       | Ubuntu Focal  | r2.4.1 | 7.2.0 | CPU, few optimizations | no        | 5.2,6.1,7.0,7.5    |
| **mdl4eo/otbtf:2.4-cpu-mkl**                                                       | Ubuntu Focal  | r2.4.1 | 7.2.0 | CPU, Intel MKL, AVX512 | yes       | 5.2,6.1,7.0,7.5    |
| **mdl4eo/otbtf:2.4-gpu**                                                           | Ubuntu Focal  | r2.4.1 | 7.2.0 | GPU                    | yes       | 5.2,6.1,7.0,7.5    |
| **mdl4eo/otbtf:2.5-cpu**                                                           | Ubuntu Focal  | r2.5   | 7.4.0 | CPU, no optimization   | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:2.5:cpu-dev**                                                       | Ubuntu Focal  | r2.5   | 7.4.0 | CPU, no optimization (dev) |  yes  | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:2.5-cpu-opt**                                                       | Ubuntu Focal  | r2.5   | 7.4.0 | CPU, few optimization  | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:2.5-gpu-opt**                                                       | Ubuntu Focal  | r2.5   | 7.4.0 | GPU                    | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:2.5-gpu-opt-dev**                                                   | Ubuntu Focal  | r2.5   | 7.4.0 | GPU (dev)              | yes       | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.0-cpu**                                                           | Ubuntu Focal  | r2.5   | 7.4.0 | CPU, no optimization   | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.0-cpu-dev**                                                       | Ubuntu Focal  | r2.5   | 7.4.0 | CPU, no optimization (dev) |  yes  | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.0-gpu-opt**                                                       | Ubuntu Focal  | r2.5   | 7.4.0 | GPU                    | yes       | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.0-gpu-opt-dev**                                                   | Ubuntu Focal  | r2.5   | 7.4.0 | GPU (dev)              | yes       | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.1-cpu**                                                           | Ubuntu Focal  | r2.8   | 7.4.0 | CPU, no optimization   | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.1-cpu-dev**                                                       | Ubuntu Focal  | r2.8   | 7.4.0 | CPU, no optimization (dev) |  yes  | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.1-gpu**                                                           | Ubuntu Focal  | r2.8   | 7.4.0 | GPU, no optimization   | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.1-gpu-dev**                                                       | Ubuntu Focal  | r2.8   | 7.4.0 | GPU, no optimization (dev) | yes   | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.1-gpu-opt**                                                       | Ubuntu Focal  | r2.8   | 7.4.0 | GPU                    | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.1-gpu-opt-dev**                                                   | Ubuntu Focal  | r2.8   | 7.4.0 | GPU (dev)              | yes       | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.3.0-cpu**                                                         | Ubuntu Focal  | r2.8   | 8.1.0 | CPU, no optimization   | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.3.0-cpu-dev**                                                     | Ubuntu Focal  | r2.8   | 8.1.0 | CPU, no optimization (dev) |  yes  | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.3.0-gpu**                                                         | Ubuntu Focal  | r2.8   | 8.1.0 | GPU, no optimization   | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.3.0-gpu-dev**                                                     | Ubuntu Focal  | r2.8   | 8.1.0 | GPU, no optimization (dev) | yes   | 5.2,6.1,7.0,7.5,8.6|
| **gitlab.irstea.fr/remi.cresson/otbtf/container_registry/otbtf:3.3.0-gpu-opt**     | Ubuntu Focal  | r2.8   | 8.1.0 | GPU with opt.          | no        | 5.2,6.1,7.0,7.5,8.6|
| **gitlab.irstea.fr/remi.cresson/otbtf/container_registry/otbtf:3.3.0-gpu-opt-dev** | Ubuntu Focal  | r2.8   | 8.1.0 | GPU with opt. (dev)    | yes       | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.3.2-cpu**                                                         | Ubuntu Focal  | r2.8   | 8.1.0 | CPU, no optimization   | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.3.2-cpu-dev**                                                     | Ubuntu Focal  | r2.8   | 8.1.0 | CPU, no optimization (dev) |  yes  | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.3.2-gpu**                                                         | Ubuntu Focal  | r2.8   | 8.1.0 | GPU, no optimization   | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.3.2-gpu-dev**                                                     | Ubuntu Focal  | r2.8   | 8.1.0 | GPU, no optimization (dev) | yes   | 5.2,6.1,7.0,7.5,8.6|
| **gitlab.irstea.fr/remi.cresson/otbtf/container_registry/otbtf:3.3.2-gpu-opt**     | Ubuntu Focal  | r2.8   | 8.1.0 | GPU with opt.          | no        | 5.2,6.1,7.0,7.5,8.6|
| **gitlab.irstea.fr/remi.cresson/otbtf/container_registry/otbtf:3.3.2-gpu-opt-dev** | Ubuntu Focal  | r2.8   | 8.1.0 | GPU with opt. (dev)    | yes       | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.3.3-cpu**                                                         | Ubuntu Focal  | r2.8   | 8.1.0 | CPU, no optimization   | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.3.3-cpu-dev**                                                     | Ubuntu Focal  | r2.8   | 8.1.0 | CPU, no optimization (dev) |  yes  | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.3.3-gpu**                                                         | Ubuntu Focal  | r2.8   | 8.1.0 | GPU, no optimization   | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.3.3-gpu-dev**                                                     | Ubuntu Focal  | r2.8   | 8.1.0 | GPU, no optimization (dev) | yes   | 5.2,6.1,7.0,7.5,8.6|
| **gitlab.irstea.fr/remi.cresson/otbtf/container_registry/otbtf:3.3.3-gpu-opt**     | Ubuntu Focal  | r2.8   | 8.1.0 | GPU with opt.          | no        | 5.2,6.1,7.0,7.5,8.6|
| **gitlab.irstea.fr/remi.cresson/otbtf/container_registry/otbtf:3.3.3-gpu-opt-dev** | Ubuntu Focal  | r2.8   | 8.1.0 | GPU with opt. (dev)    | yes       | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.4.0-cpu**                                                         | Ubuntu Focal  | r2.8   | 8.1.0 | CPU, no optimization   | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.4.0-cpu-dev**                                                     | Ubuntu Focal  | r2.8   | 8.1.0 | CPU, no optimization (dev) |  yes  | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.4.0-gpu**                                                         | Ubuntu Focal  | r2.8   | 8.1.0 | GPU, no optimization   | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:3.4.0-gpu-dev**                                                     | Ubuntu Focal  | r2.8   | 8.1.0 | GPU, no optimization (dev) | yes   | 5.2,6.1,7.0,7.5,8.6|
| **gitlab.irstea.fr/remi.cresson/otbtf/container_registry/otbtf:3.4.0-gpu-opt**     | Ubuntu Focal  | r2.8   | 8.1.0 | GPU with opt.          | no        | 5.2,6.1,7.0,7.5,8.6|
| **gitlab.irstea.fr/remi.cresson/otbtf/container_registry/otbtf:3.4.0-gpu-opt-dev** | Ubuntu Focal  | r2.8   | 8.1.0 | GPU with opt. (dev)    | yes       | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:4.0.0-cpu**                                                         | Ubuntu Jammy  | r2.12  | 8.1.0 | CPU, no optimization   | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:4.0.0-cpu-dev**                                                     | Ubuntu Jammy  | r2.12  | 8.1.0 | CPU, no optimization (dev) |  yes  | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:4.0.0-gpu**                                                         | Ubuntu Jammy  | r2.12  | 8.1.0 | GPU, no optimization   | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:4.0.0-gpu-dev**                                                     | Ubuntu Jammy  | r2.12  | 8.1.0 | GPU, no optimization (dev) | yes   | 5.2,6.1,7.0,7.5,8.6|
| **gitlab.irstea.fr/remi.cresson/otbtf/container_registry/otbtf:4.0.0-gpu-opt**     | Ubuntu Jammy  | r2.12  | 8.1.0 | GPU with opt.          | no        | 5.2,6.1,7.0,7.5,8.6|
| **gitlab.irstea.fr/remi.cresson/otbtf/container_registry/otbtf:4.0.0-gpu-opt-dev** | Ubuntu Jammy  | r2.12  | 8.1.0 | GPU with opt. (dev)    | yes       | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:4.1.0-cpu**                                                         | Ubuntu Jammy  | r2.12 | 8.1.0 | CPU, no optimization   | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:4.1.0-cpu-dev**                                                     | Ubuntu Jammy  | r2.12 | 8.1.0 | CPU, no optimization (dev) |  yes  | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:4.1.0-gpu**                                                         | Ubuntu Jammy  | r2.12 | 8.1.0 | GPU, no optimization   | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:4.1.0-gpu-dev**                                                     | Ubuntu Jammy  | r2.12 | 8.1.0 | GPU, no optimization (dev) | yes   | 5.2,6.1,7.0,7.5,8.6|
| **gitlab.irstea.fr/remi.cresson/otbtf/container_registry/otbtf:4.1.0-gpu-opt**     | Ubuntu Jammy  | r2.12 | 8.1.0 | GPU with opt.          | no        | 5.2,6.1,7.0,7.5,8.6|
| **gitlab.irstea.fr/remi.cresson/otbtf/container_registry/otbtf:4.1.0-gpu-opt-dev** | Ubuntu Jammy  | r2.12 | 8.1.0 | GPU with opt. (dev)    | yes       | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:4.2.0-cpu**                                                         | Ubuntu Jammy  | r2.12 | 8.1.0 | CPU, no optimization   | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:4.2.0-cpu-dev**                                                     | Ubuntu Jammy  | r2.12 | 8.1.0 | CPU, no optimization (dev) |  yes  | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:4.2.0-gpu**                                                         | Ubuntu Jammy  | r2.12 | 8.1.0 | GPU, no optimization   | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:4.2.0-gpu-dev**                                                     | Ubuntu Jammy  | r2.12 | 8.1.0 | GPU, no optimization (dev) | yes   | 5.2,6.1,7.0,7.5,8.6|
| **gitlab.irstea.fr/remi.cresson/otbtf/container_registry/otbtf:4.2.0-gpu-opt**     | Ubuntu Jammy  | r2.12 | 8.1.0 | GPU with opt.          | no        | 5.2,6.1,7.0,7.5,8.6|
| **gitlab.irstea.fr/remi.cresson/otbtf/container_registry/otbtf:4.2.0-gpu-opt-dev** | Ubuntu Jammy  | r2.12 | 8.1.0 | GPU with opt. (dev)    | yes       | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:4.2.1-cpu**                                                         | Ubuntu Jammy  | r2.12 | 8.1.2 | CPU, no optimization   | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:4.2.1-cpu-dev**                                                     | Ubuntu Jammy  | r2.12 | 8.1.2 | CPU, no optimization (dev) |  yes  | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:4.2.1-gpu**                                                         | Ubuntu Jammy  | r2.12 | 8.1.2 | GPU, no optimization   | no        | 5.2,6.1,7.0,7.5,8.6|
| **mdl4eo/otbtf:4.2.1-gpu-dev**                                                     | Ubuntu Jammy  | r2.12 | 8.1.2 | GPU, no optimization (dev) | yes   | 5.2,6.1,7.0,7.5,8.6|
| **gitlab.irstea.fr/remi.cresson/otbtf/container_registry/otbtf:4.2.1-gpu-opt**     | Ubuntu Jammy  | r2.12 | 8.1.2 | GPU with opt.          | no        | 5.2,6.1,7.0,7.5,8.6|
| **gitlab.irstea.fr/remi.cresson/otbtf/container_registry/otbtf:4.2.1-gpu-opt-dev** | Ubuntu Jammy  | r2.12 | 8.1.2 | GPU with opt. (dev)    | yes       | 5.2,6.1,7.0,7.5,8.6|

