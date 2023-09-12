# Docker troubleshooting

You can find plenty of help on the web about docker. 
This section only provides the basics for newcomers that are eager to use 
OTBTF!
This section is largely inspired from the 
[moringa docker help](https://gitlab.irstea.fr/raffaele.gaetano/moringa/blob/develop/docker/README.md). 
Big thanks to the authors.

## Common errors

### Manifest unknown

```
Error response from daemon: 
manifest for nvidia/cuda:11.0-cudnn8-devel-ubuntu20.04 not found: 
manifest unknown: manifest unknown
```

This means that the docker image is missing from dockerhub.

### failed call to cuInit

```
failed call to cuInit: 
UNKNOWN ERROR (303) / no NVIDIA GPU device is present: 
/dev/nvidia0 does not exist
```

Nvidia driver is missing or disabled, make sure to add 
` --gpus=all` to your docker run or create command

## Useful diagnostic commands

Here are some useful commands.

```bash
docker info         # System info
docker images       # List local images
docker container ls # List containers
docker ps           # Show running containers
```

On Linux, control state with `service`:

```bash
sudo service docker {status,enable,disable,start,stop,restart}
```

### Run some commands

Run a simple command in a one-shot container:

```bash
docker run mdl4eo/otbtf:4.2.0-cpu otbcli_PatchesExtraction
```

You can also use the image in interactive mode with bash:

```bash
docker run -ti mdl4eo/otbtf:4.2.0-cpu bash
```

### Mounting file systems

You can mount filesystem in the docker image.
For instance, suppose you have some data in `/mnt/disk1/` that you want 
to use inside the container:

The following command shows you how to access the folder from the docker image.

```bash
docker run -v /mnt/disk1/:/data/ -ti mdl4eo/otbtf:4.2.0-cpu bash -c "ls /data"
```
Beware of ownership issues! see the last section of this doc.

### Persistent container

Persistent (named) container with volume, here with home dir, but it can be 
any directory.

```bash
docker create --interactive --tty --volume /home/$USER:/home/otbuser/ \
    --name otbtf mdl4eo/otbtf:4.2.0-cpu /bin/bash
```

!!! warning

    Beware of ownership issues, see 
    [this section](#fix-volume-ownership-sissues).

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

Background container is one easy way:

```bash
docker start otbtf
# Example with apt update 
# (you can't use &&, one docker exec is
# required for each command)
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

## Fix volume ownership issues

Generally, this is required if host's UID > 1000.

When mounting a volume, you may experience errors while trying to write files 
from within the container.
Since the default user (**otbuser**) is UID 1000, you won't be able to write 
files into your volume 
which is mounted with the same UID than your linux host user (may be UID 1001
or more). 
In order to address this, you need to edit the container's user UID and GID to 
match the right numerical value.
This will only persist in a named container, it is required every time you're 
creating a new one.


Create a named container (here with your HOME as volume), Docker will 
automatically pull image

```bash
docker create --interactive --tty --volume /home/$USER:/home/otbuser \
    --name otbtf mdl4eo/otbtf:4.2.0-cpu /bin/bash
```

Start a background container process:

```bash
docker start otbtf
```

Exec required commands with user root (here with host's ID, replace $UID and 
$GID with desired values):

```bash
docker exec --user root otbtf usermod otbuser -u $UID
docker exec --user root otbtf groupmod otbuser -g $GID
```

Force reset ownership with updated UID and GID. 
Make sure to double check that `docker exec otbtf id` because recursive chown 
will apply to your volume in `/home/otbuser`

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
