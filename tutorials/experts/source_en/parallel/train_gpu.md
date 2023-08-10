# Distributed Parallel Training Example (GPU)

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/train_gpu.md)

## Overview

This tutorial describes how to train a ResNet-50 network by using a CIFAR-10 dataset on a GPU processor hardware platform through MindSpore and data parallelism and automatic parallelism mode.

> You can download the complete sample code here: [distributed_training](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_training)

The directory structure is as follows:

```text
└─sample_code
    ├─distributed_training
       ├── rank_table_16pcs.json
       ├── rank_table_8pcs.json
       ├── rank_table_2pcs.json
       ├── cell_wrapper.py
       ├── model_accu.py
       ├── resnet.py
       ├── resnet50_distributed_training.py
       ├── resnet50_distributed_training_gpu.py
       ├── resnet50_distributed_training_grad_accu.py
       ├── run.sh
       ├── run_gpu.sh
       ├── run_grad_accu.sh
       ├── run_cluster.sh
```

Where `resnet.py` and `resnet50_distributed_training_gpu.py` are the scripts that define the structure of the network. `run_gpu.sh` is the execution script and the remaining files are Ascend 910.

## Preparation

In order to ensure the normal progress of the distributed training, we need to configure and initially test the distributed environment first. After completion, prepare the CIFAR-10 dataset.

### Configuring Distributed Environment

- `OpenMPI-4.1.4`: multi-process communication library used by MindSpore.

  Download the OpenMPI-4.1.4 source code package [openmpi-4.1.4.tar.gz](https://www.open-mpi.org/software/ompi/v4.1/).

  For details about how to install OpenMPI, see the official tutorial: [easy-build](https://www.open-mpi.org/faq/?category=building#easy-build).

- Password-free login between hosts (required for multi-host training). If multiple hosts are involved in the training, you need to configure password-free login between them. The procedure is as follows:
  1. Ensure that the same user is used to log in to each host. (The root user is not recommended.)
  2. Run the `ssh-keygen -t rsa -P ""` command to generate a key.
  3. Run the `ssh-copy-id DEVICE-IP` command to set the IP address of the host that requires password-free login.
  4. Run the`ssh DEVICE-IP` command. If you can log in without entering the password, the configuration is successful.
  5. Run the preceding command on all hosts to ensure that every two hosts can communicate with each other.

### Calling the Collective Communication Library

On the GPU hardware platform, communication in MindSpore distributed parallel training uses NVIDIA's collective communication library `NVIDIA Collective Communication Library` (NCCL for short).

> On the GPU platform, MindSpore does not support the following operations:
>
> `get_local_rank`, `get_local_size`, `get_world_rank_from_group_rank`, `get_group_rank_from_world_rank` and `create_group`

The sample code for calling the HCCL is as follows and sets the file name as nccl_allgather.py:

```python
# nccl_allgather.py
import numpy as np
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore as ms
from mindspore.communication import init, get_rank


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.allgather = ops.AllGather()

    def construct(self, x):
        return self.allgather(x)


if __name__ == "__main__":
    ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")
    init("nccl")
    value = get_rank()
    input_x = ms.Tensor(np.array([[value]]).astype(np.float32))
    net = Net()
    output = net(input_x)
    print(output)
```

In the preceding information,

- `mode=ms.GRAPH_MODE`: sets the running mode to graph mode for distributed training. (The PyNative mode only supports data parallel running.)
- `device_target="GPU"`: specifies device as GPU.
- `init("nccl")`: enables NCCL communication and completes the distributed training initialization.
- `get_rank()`: obtains the rank number of  the current process.
- `ops.AllGather`: invokes the NCCL's AllGather communication operation on the GPU, the meaning of which and more can be found in Distributed Set Communication Primitives.

On GPU hardware platform, MindSpore uses mpirun of OpenMPI to initiate processes, usually one computing device for each process.

```bash
mpirun -n DEVICE_NUM python nccl_allgather.py
```

Where the DEVICE_NUM is the number of GPUs of the machine. Taking DEVICE_NUM=4 as an example, the expected output is:

```text
[[0.],
 [1.],
 [2.],
 [3.]]
```

The output log can be output from the terminal after the program is executed. If the above output is obtained, it means that OpenMPI and NCCL are working normally, and the process is starting normally.

### Downloading the Dataset

This example uses the `CIFAR-10` dataset, which consists of 10 classes of 32*32 color pictures, each containing 6,000 pictures, for a total of 60,000 pictures. The training set has a total of 50,000 pictures, and the test set has a total of 10,000 pictures.

> Download [CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz). If the download is unsuccessful, please try copying the link address and download it.

The Linux machine can use the following command to download to the current path of the terminal and extract the dataset, and the folder of the extracted data is `cifar-10-batches-bin`.

```bash
wget http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -zxvf cifar-10-binary.tar.gz
```

## Loading the Dataset in the Data Parallel Mode

During distributed training, data is imported in the data parallel mode. Taking the CIFAR-10 dataset as an example, we introduce the method of importing the CIFAR-10 dataset in parallel. `data_path` refers to the path of the dataset, that is, the path of the `cifar-10-batches-bin` folder.

```python
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindspore.communication import get_rank, get_group_size


def create_dataset(data_path, repeat_num=1, batch_size=32, rank_id=0, rank_size=1):
    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0

    # get rank_id and rank_size
    rank_id = get_rank()
    rank_size = get_group_size()
    data_set = ds.Cifar10Dataset(data_path, num_shards=rank_size, shard_id=rank_id)

    # define map operations
    random_crop_op = vision.RandomCrop((32, 32), (4, 4, 4, 4))
    random_horizontal_op = vision.RandomHorizontalFlip()
    resize_op = vision.Resize((resize_height, resize_width))
    rescale_op = vision.Rescale(rescale, shift)
    normalize_op = vision.Normalize((0.4465, 0.4822, 0.4914), (0.2010, 0.1994, 0.2023))
    changeswap_op = vision.HWC2CHW()
    type_cast_op = transforms.TypeCast(ms.int32)

    c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op, changeswap_op]

    # apply map operations on images
    data_set = data_set.map(operations=type_cast_op, input_columns="label")
    data_set = data_set.map(operations=c_trans, input_columns="image")

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=10)

    # apply batch operations
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    # apply repeat operations
    data_set = data_set.repeat(repeat_num)

    return data_set
```

Unlike stand-alone machines, the `num_shards` and `shard_id` parameters need to be passed in on the dataset interface, which correspond to the number of cards and logical ordinal numbers, respectively, and it is recommended to obtain them through the NCCL interface:

- `get_rank`: obtains the ID of the current device in the cluster.
- `get_group_size`: obtains the number of the clusters.

> When loading datasets in a data-parallel scenario, it is recommended that you specify the same dataset file for each card. If the datasets loaded by each card are different, the calculation accuracy may be affected.

## Defining the Network

On the GPU hardware platform, the network definition is the same as that for the Ascend 910 AI processor.

In the **Data Parallelism** and **Auto Parallelism** modes, the network is defined in the same way as the stand-alone writing, see [ResNet Network Sample Script](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/resnet/resnet.py).

> - In semi-automatic parallel mode, operators without a policy configured default to data parallelism.
> - The automatic parallel mode supports automatically obtaining efficient operator parallel policies through the policy search algorithm, and also allows users to manually configure specific parallel policies for operators.
> - If a `parameter` is used by more than one operator, the segment policy of each operator for that `parameter` needs to be consistent, otherwise an error will be reported.

## Defining the Loss Function and Optimizer

Consistent with the [Distributed Parallel Training Basics Sample](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html) on Ascend.

### Defining the Loss Function

Automatic parallelism takes the operator as the granularity segment model, and the optimal parallelism policy is obtained through algorithm search. Unlike stand-alone training, in order to have a better parallel training effect, the loss function is recommended to use the MindSpore operator to achieve it, rather than directly using the encapsulated loss function class.

In the Loss section, we take the form of `SoftmaxCrossEntropyWithLogits`, that is, according to the mathematical formula, expand it into multiple MindSpore operators for implementation, the sample code is as follows:

```python
import mindspore.ops as ops
import mindspore as ms
import mindspore.nn as nn


class SoftmaxCrossEntropyExpand(nn.Cell):
    def __init__(self, sparse=False):
        super(SoftmaxCrossEntropyExpand, self).__init__()
        self.exp = ops.Exp()
        self.sum = ops.ReduceSum(keep_dims=True)
        self.onehot = ops.OneHot()
        self.on_value = ms.Tensor(1.0, ms.float32)
        self.off_value = ms.Tensor(0.0, ms.float32)
        self.div = ops.RealDiv()
        self.log = ops.Log()
        self.sum_cross_entropy = ops.ReduceSum(keep_dims=False)
        self.mul = ops.Mul()
        self.mul2 = ops.Mul()
        self.mean = ops.ReduceMean(keep_dims=False)
        self.sparse = sparse
        self.max = ops.ReduceMax(keep_dims=True)
        self.sub = ops.Sub()

    def construct(self, logit, label):
        logit_max = self.max(logit, -1)
        exp = self.exp(self.sub(logit, logit_max))
        exp_sum = self.sum(exp, -1)
        softmax_result = self.div(exp, exp_sum)
        if self.sparse:
            label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)
        softmax_result_log = self.log(softmax_result)
        loss = self.sum_cross_entropy((self.mul(softmax_result_log, label)), -1)
        loss = self.mul2(ops.scalar_to_tensor(-1.0), loss)
        loss = self.mean(loss, -1)

        return loss
```

### Defining the Optimizer

The `Momentum` optimizer is used as a parameter update tool. The definition here is consistent with that of the stand-alone machine, and it is no longer expanded, which can be referred to the implementation in the sample code.

## Training the Network

Before training, we need to configure some automatic parallelism parameters. `set_auto_parallel_context` is the interface for configuring the parallel training mode and must be called before the network is initialized. Common parameters include:

- `parallel_mode`: distributed parallel mode. The default is stand-alone mode `ParallelMode.STAND_ALONE`. In this example, you can select data parallelism `ParallelMode.DATA_PARALLEL` and automatic parallel `ParallelMode.AUTO_PARALLEL`.
- `parameter_broadcast`: before the start of training, the parameter weights of data parallelism on card 0 are automatically broadcast to other cards, and the default value is `False`.
- `gradients_mean`: in reverse calculation, the framework will scatter the data parallel parameters in the gradient values of multiple machines for collection, obtain the global gradient values, and then pass them into the optimizer for update. The default value is `False`, which is set to True for the `allreduce_mean` operation and False for the `allreduce_sum` action.
- `device_num`和`global_rank`: it is recommended to use the default value, and the NCCL interface will be called within the framework to get it.

If there are multiple network use cases in the script, call `reset_auto_parallel_context` to restore all parameters to the default values before executing the next use case.

In the example below, we specify the parallel mode as automatic parallelism, and the user needs to switch to data parallel mode by simply changing `parallel_mode` to `DATA_PARALLEL`.

> The pynative mode currently supports data parallelism, and the usage is consistent with the data parallelism in graph mode. You only need to change 'mode' to ' PYNATIVE_MODE '.

```python
import mindspore as ms
from mindspore.train import Model, LossMonitor
from mindspore.nn import Momentum
from mindspore.communication import init
from resnet import resnet50

ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")
init("nccl")


def test_train_cifar(epoch_size=10):
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL, gradients_mean=True)
    loss_cb = LossMonitor()
    dataset = create_dataset(data_path)
    batch_size = 32
    num_classes = 10
    net = resnet50(batch_size, num_classes)
    loss = SoftmaxCrossEntropyExpand(sparse=True)
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    model = Model(net, loss_fn=loss, optimizer=opt)
    model.train(epoch_size, dataset, callbacks=[loss_cb], dataset_sink_mode=True)
```

Where,

- `dataset_sink_mode=True`: indicates a sinking pattern that takes a dataset, that is, the computation of the training is performed in a hardware platform.
- `LossMonitor`: returns the loss value via a callback function for monitoring loss functions.

## Running the Script

On GPU hardware platform, MindSpore uses `mpirun`of OpenMPI for distributed training. After completing the definition of the model, loss function, and optimizer, we have completed the configuration of the model parallelism strategy, and then execute the running script directly.

### Single-host Training

The following takes the distributed training script for eight devices as an example to describe how to run the script:

> Obtain the running script of the example from [run_gpu.sh](https://gitee.com/mindspore/docs/blob/master/docs/sample_code/distributed_training/run_gpu.sh).
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
cp ./resnet50_distributed_training_gpu.py ./resnet.py ./device
cd ./device
echo "start training"
mpirun -n 8 pytest -s -v ./resnet50_distributed_training_gpu.py > train.log 2>&1 &
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

Before running multi-host training, you need to ensure that you have the same openMPI, NCCL, Python, and MindSpore versions on each node.

#### mpirun -H

If multiple hosts are involved in the training, you need to set the multi-host configuration in the `mpirun` command. You can use the `-H` option in the `mpirun` command. For example,

```text
mpirun -n 16 -H DEVICE1_IP:8,DEVICE2_IP:8 python hello.py
```

indicates that eight processes are started on the hosts whose IP addresses are DEVICE1_IP and DEVICE2_IP, respectively.

#### mpirun --hostfile

Multi-host execution of GPU can also be performed by constructing hostfile files. For ease of debugging, it is recommended to use this method to execute multi-host scripts. This is then executed in the form of `mpirun --hostfile $HOST_FILE`. Below we give a detailed Multi-host configuration in the hostfile boot method.

Each line in the hostfile is in the format of `[hostname] slots=[slotnum]`, where hostname can be an IP address or a host name. It should be noted that the usernames on different machines need to be the same, but hostnames cannot be the same. As follows, it means that there are 8 cards on DEVICE1 and 8 cards on machines with IP 192.168.0.1:

```text
DEVICE1 slots=8
192.168.0.1 slots=8
```

The following is the execution script of the 16-device two-host cluster. The variables `DATA_PATH` and `HOSTFILE` need to be transferred, indicating the dataset path and hostfile path. We need to set the btl parameter of mca in mpi to specify the network card that communicates with mpi, otherwise it may fail to initialize when calling the mpi interface. The btl parameter specifies the TCP protocol between nodes and the loop within the nodes for communication. The IP address of the network card through which the specified nodes communication of  btl_tcp_if_include needs to be in the given subnet. For details about more mpirun options, see the OpenMPI official website.

```bash
#!/bin/bash

DATA_PATH=$1
HOSTFILE=$2

rm -rf device
mkdir device
cp ./resnet50_distributed_training_gpu.py ./resnet.py ./device
cd ./device
echo "start training"
mpirun -n 16 --mca btl tcp,self --mca btl_tcp_if_include 192.168.0.0/24 --hostfile $HOSTFILE -x DATA_PATH=$DATA_PATH -x PATH -mca pml ob1 mpirun_gpu_clusher.sh &
```

Considering that some environment variables on different machines may be different, we take the form of mpirun to start a `mpirun_gpu_cluster.sh` and specify the required environment variables in the script file on different machines. Here we have configured the `NCCL_SOCKET_IFNAME` to specify the NIC when the NCCL communicates.

```bash
#!/bin/bash
# mpirun_gpu_clusher.sh
# Here you can set different environment variables on each machine, such as the name of the network card below

export NCCL_SOCKET_IFNAME="en5" # The name of the network card that needs to communicate between nodes may be inconsistent on different machines, and use ifconfig to view.
pytest -s -v ./resnet50_distributed_training_gpu.py > train.log 2>&1 &
```

## Saving and Loading the Distributed Training Model Parameter

When performing distributed training on a GPU, the method of saving and loading the model parameters is the same as that on Ascend, which can be referred to [Distributed Training Model Parameters Saving and Loading](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#saving-and-loading-distributed-training-model-parameters).

## Training without Relying on OpenMPI

Please refer to [Dynamic Cluster](https://www.mindspore.cn/tutorials/experts/en/master/parallel/dynamic_cluster.html) for more details about **Training without Relying on OpenMPI**