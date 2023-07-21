# Distributed Parallel Training (GPU)

`Linux` `GPU` `Model Training` `Intermediate` `Expert` 

[![View Source On Gitee](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/tutorials/training/source_en/advanced_use/distributed_training_gpu.md)

## Overview

This tutorial describes how to train the ResNet-50 network using MindSpore data parallelism and automatic parallelism on GPU hardware platform.

## Preparation

### Downloading the Dataset

The `CIFAR-10` dataset is used as an example. The method of downloading and loading the dataset is the same as that for the Ascend 910 AI processor.

> The method of downloading and loading the dataset:
>
> <https://www.mindspore.cn/tutorial/training/en/r1.0/advanced_use/distributed_training_ascend.html>

### Configuring Distributed Environment

- `OpenMPI-3.1.5`: multi-process communication library used by MindSpore.

  > Download the OpenMPI-3.1.5 source code package `openmpi-3.1.5.tar.gz` from <https://www.open-mpi.org/software/ompi/v3.1/>.
  >
  > For details about how to install OpenMPI, see the official tutorial: <https://www.open-mpi.org/faq/?category=building#easy-build>.

- `NCCL-2.7.6`: Nvidia collective communication library.

  > Download NCCL-2.7.6 from <https://developer.nvidia.com/nccl/nccl-legacy-downloads>.
  >
  > For details about how to install NCCL, see the official tutorial: <https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html#debian>.

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
from mindspore.communication.management import init

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

> For details about the definitions of the network, optimizer, and loss function, see <https://www.mindspore.cn/tutorial/training/en/r1.0/advanced_use/distributed_training_ascend.html>.

## Running the Script

On the GPU hardware platform, MindSpore uses OpenMPI `mpirun` for distributed training. The following takes the distributed training script for eight devices as an example to describe how to run the script:

> Obtain the running script of the example from:
>
> <https://gitee.com/mindspore/docs/blob/r1.0/tutorials/tutorial_code/distributed_training/run_gpu.sh>
>
> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

```bash
#!/bin/bash

DATA_PATH=$1
export DATA_PATH=${DATA_PATH}

rm -rf device
mkdir device
cp ./resnet50_distributed_training.py ./resnet.py ./device
cd ./device
echo "start training"
mpirun -n 8 pytest -s -v ./resnet50_distributed_training.py > train.log 2>&1 &
```

The script requires the variable `DATA_PATH`, which indicates the path of the dataset. In addition, you need to modify the `resnet50_distributed_training.py` file. Since the `DEVICE_ID` environment variable does not need to be set on the GPU, you do not need to call `int(os.getenv('DEVICE_ID'))` in the script to obtain the physical sequence number of the device, and `context` does not require `device_id`. You need to set `device_target` to `GPU` and call `init("nccl")` to enable the NCCL. The log file is saved in the device directory, and the loss result is saved in train.log. The output loss values of the grep command are as follows:

```
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
epoch: 1 step: 1, loss is 2.3025854
```

## Running the Multi-Host Script

If multiple hosts are involved in the training, you need to set the multi-host configuration in the `mpirun` command. You can use the `-H` option in the `mpirun` command. For example, `mpirun -n 16 -H DEVICE1_IP:8,DEVICE2_IP:8 python hello.py` indicates that eight processes are started on the host whose IP addresses are DEVICE1_IP and DEVICE2_IP, respectively. Alternatively, you can create a hostfile similar to the following and transfer its path to the `--hostfile` option of `mpirun`. Each line in the hostfile is in the format of `[hostname] slots=[slotnum]`, where hostname can be an IP address or a host name.
```bash
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

Run running on GPU, the model parameters can be saved and loaded for reference[Distributed Training Model Parameters Saving and Loading](https://www.mindspore.cn/tutorial/training/en/r1.0/advanced_use/distributed_training_ascend.html#distributed-training-model-parameters-saving-and-loading)
