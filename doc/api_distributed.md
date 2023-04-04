# Distributed training

Thanks to the new API of Tensorflow since version 2.0, it's very easy to 
perform distributed tranining with the exact same code. Just one single line 
has to be changed!

## Overview

In the following, we will explain how to use multiple nodes of a GPU cluster
like the [Jean-Zay supercomputer](http://www.idris.fr/eng/jean-zay/jean-zay-presentation-eng.html),
using `tf.distribute.MultiWorkerMirroredStrategy`.
We will adopt the data parallelism scheme, meaning that all the computing 
devices will have replicas of the model, but different chunks of data.
The principle is that once the forward propagation is performed, the gradients 
from the different devices are aggregated together, and the weights are updated
on all GPUs.


<div align="center" width="50%">
<img src="http://www.idris.fr/media/images/jean-zay-annonce-01.jpg?id=web%3Ajean-zay%3Ajean-zay-presentation" width=50%>
<figcaption>Jean-Zay has several hundreds of computing nodes with 4 or 8 GPUs. Copyright Photothèque CNRS/Cyril Frésillon</figcaption>
</div>

## Python code

We can start from the codebase of the fully convolutional model example 
described in the OTBTF [Python API tutorial](#api_tutorial.html).

### Dataset

For distributed training, we recommend to use the TFRecords rather than the
Patch based images.
This has two advantages:

- Performance in terms of I/O
- `otbtf` can be imported without anything else than `tensorflow` as 
dependency. Indeed, the `otbtf.TFRecords` class just needs the `tensorflow` 
module to work.

!!! Info

    When imported, OTBTF tries to import the GDAL-related classes (e.g.
    `PatchesImagesReader`) and skip the import if an `ImportError` occurs (i.e.
    when GDAL is not present in the environment). This allows to safely use the
    other classes that rely purely on the `tensorflow` module (e.g. 
    `otbtf.ModelBase`, `otbtf.TFRecords`, etc.).

### Prerequisites

To use OTBTF on environment where only Tensorflow is available, you can just 
clone the OTBTF repository somewhere and install it in your favorite virtual
environment with `pip`. Or you can also just update the `PYTHONPATH` to include
the *otbtf* folder. You just have to be able to perform the import of the
module from python code:

```python
import otbtf
```

### Strategy

We change the strategy from `tf.distribute.MirroredStrategy` to
`tf.distribute.MultiWorkerMirroredStrategy`.

First, we have to instantiate a cluster resolver for SLURM, which is the job 
scheduler of the cluster. The cluster resolver uses the environment variables 
provided by SLURM to grab the useful parameters. On the Jean-Zay computer,
the port base is **13565**:

```python
cluster_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(
    port_base=13565
)
```

Then we specify a communication protocol. The Jean-Zay computer supports 
the NVIDIA NCCL communication protocol, which links tightly GPUs from different
nodes:

```python
implementation = tf.distribute.experimental.CommunicationImplementation.NCCL
communication_options = tf.distribute.experimental.CommunicationOptions(
    implementation=implementation
)
```

Finally, we can replace the strategy with the distributed one:

```python
#strategy = tf.distribute.MirroredStrategy()  # <-- that was before
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    cluster_resolver=cluster_resolver,
    communication_options=communication_options
)
```

The rest of the code is identical.

!!! Warning

    Be careful when calling `mymodel.save()` to export the SavedModel. When 
    multiple nodes are used in parallel, this can lead to a corrupt save.
    One good practice is to defer the call only to the master worker (e.g. node
    0). You can identify the master worker using `otbtf.model._is_chief()`.

## SLURM job

Now we have to provide a SLURM job to run our python code over several nodes.
Below is the content of the *job.slurm* file:

```commandline
#!/bin/bash
#SBATCH -A <your_account>@gpu
#SBATCH --job-name=<job_name>
#SBATCH --nodes=4               # number of nodes
#SBATCH --ntasks-per-node=4     # number of MPI task per node
#SBATCH --gres=gpu:4            # number of GPU per node
#SBATCH --cpus-per-task=10      # number of cores per task
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=00:59:00
#SBATCH -C v100-16g             # Multiworker strategy wants homogeneous GPUs

cd ${SLURM_SUBMIT_DIR}

# deactivate the HTTP proxy (mandatory for multi-node)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
module purge
module load tensorflow-gpu/py3/2.8.0

export PYTHONPATH=$PYTHONPATH:/path/to/otbtf/
srun
python3 /path/to/your_code.py
```

To submit the job, run the following command:
```commandline
sbatch job.slurm
```

## References

- [Jean-Zay users documentation](https://jean-zay-doc.readthedocs.io/en/latest/)
- [Official Jean-Zay documentation](http://www.idris.fr/jean-zay/gpu/jean-zay-gpu-tf-multi.html)
