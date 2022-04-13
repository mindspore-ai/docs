# Distributed Parallel Training Example (GPU)

`GPU` `Distributed Parallel` `Whole Process`

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/train_gpu.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

This tutorial describes how to train the ResNet-50 network using MindSpore data parallelism and automatic parallelism on the GPU hardware platform.

## Preparation

### Downloading the Dataset

The `CIFAR-10` dataset is used as an example. The method of downloading and loading the dataset is the same as that for the Ascend 910 AI processor.

The method of downloading and loading the dataset: <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html>

### Configuring Distributed Environment

- `OpenMPI-4.0.3`: multi-process communication library used by MindSpore.

  Download the OpenMPI-4.0.3 source code package `openmpi-4.0.3.tar.gz` from <https://www.open-mpi.org/software/ompi/v4.0/>.

  For details about how to install OpenMPI, see the official tutorial: <https://www.open-mpi.org/faq/?category=building#easy-build>.

- Password-free login between hosts (required for multi-host training). If multiple hosts are involved in the training, you need to configure password-free login between them. The procedure is as follows:
  1. Ensure that the same user is used to log in to each host. (The root user is not recommended.)
  2. Run the `ssh-keygen -t rsa -P ""` command to generate a key.
  3. Run the `ssh-copy-id DEVICE-IP` command to set the IP address of the host that requires password-free login.
  4. Run the`ssh DEVICE-IP` command. If you can log in without entering the password, the configuration is successful.
  5. Run the preceding command on all hosts to ensure that every two hosts can communicate with each other.

### Calling the Collective Communication Library

On the GPU hardware platform, MindSpore parallel distributed training uses NCCL for communication.

> On the GPU platform, MindSpore does not support the following operations:
>
> `get_local_rank`, `get_local_size`, `get_world_rank_from_group_rank`, `get_group_rank_from_world_rank` and `create_group`

The sample code for calling the HCCL is as follows:

```python
from mindspore import context
from mindspore.communication import init

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    init("nccl")
    ...
```

In the preceding information,

- `mode=context.GRAPH_MODE`: sets the running mode to graph mode for distributed training. (The PyNative mode does not support parallel running.)
- `init("nccl")`: enables NCCL communication and completes the distributed training initialization.

## Defining the Network

On the GPU hardware platform, the network definition is the same as that for the Ascend 910 AI processor.

For details about the definitions of the network, optimizer, and loss function, see <https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html>.

## Running the Script

On the GPU hardware platform, MindSpore uses OpenMPI `mpirun` for distributed training.

### Single-host Training

The following takes the distributed training script for eight devices as an example to describe how to run the script:

> Obtain the running script of the example from:
>
> <https://gitee.com/mindspore/docs/blob/master/docs/sample_code/distributed_training/run_gpu.sh>
>
> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

```bash
#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_gpu.sh DATA_PATH"
echo "For example: bash run_gpu.sh /path/dataset"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}

rm -rf device
mkdir device
cp ./resnet50_distributed_training.py ./resnet.py ./device
cd ./device
echo "start training"
mpirun -n 8 pytest -s -v ./resnet50_distributed_training.py > train.log 2>&1 &
```

The script will run in the bachground. The log file is saved in the device directory, we will run 10 epochs and each epochs contain 234 steps, and the loss result is saved in train.log. The output loss values of the grep command are as follows:

```text
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
```

### Multi-host Training

If multiple hosts are involved in the training, you need to set the multi-host configuration in the `mpirun` command. You can use the `-H` option in the `mpirun` command. For example, `mpirun -n 16 -H DEVICE1_IP:8,DEVICE2_IP:8 python hello.py` indicates that eight processes are started on the hosts whose IP addresses are DEVICE1_IP and DEVICE2_IP, respectively. Alternatively, you can create a hostfile similar to the following and transfer its path to the `--hostfile` option of `mpirun`. Each line in the hostfile is in the format of `[hostname] slots=[slotnum]`, where hostname can be an IP address or a host name.

```text
DEVICE1 slots=8
DEVICE2 slots=8
```

The following is the execution script of the 16-device two-host cluster. The variables `DATA_PATH` and `HOSTFILE` need to be transferred, indicating the dataset path and hostfile path. For details about more mpirun options, see the OpenMPI official website.

```bash
#!/bin/bash

DATA_PATH=$1
HOSTFILE=$2

rm -rf device
mkdir device
cp ./resnet50_distributed_training.py ./resnet.py ./device
cd ./device
echo "start training"
mpirun -n 16 --hostfile $HOSTFILE -x DATA_PATH=$DATA_PATH -x PATH -mca pml ob1 pytest -s -v ./resnet50_distributed_training.py > train.log 2>&1 &
```

Run running on GPU, the model parameters can be saved and loaded by referring to [Distributed Training Model Parameters Saving and Loading](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#distributed-training-model-parameters-saving-and-loading).
