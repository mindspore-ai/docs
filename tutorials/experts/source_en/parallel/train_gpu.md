# Distributed Parallel Training Example (GPU)

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/train_gpu.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

This tutorial describes how to train a ResNet-50 network by using a CIFAR-10 dataset on a GPU processor hardware platform through MindSpore and data parallelism and automatic parallelism mode.

> You can download the complete sample code here:
>
> <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_training>

The directory structure is as follows:

```text
└─sample_code
    ├─distributed_training
    │      rank_table_16pcs.json
    │      rank_table_8pcs.json
    │      rank_table_2pcs.json
    │      cell_wrapper.py
    │      model_accu.py
    │      resnet.py
    │      resnet50_distributed_training.py
    │      resnet50_distributed_training_gpu.py
    │      resnet50_distributed_training_grad_accu.py
    │      run.sh
    │      run_gpu.sh
    │      run_grad_accu.sh
    │      run_cluster.sh
```

Where `resnet.py` and `resnet50_distributed_training_gpu.py` are the scripts that define the structure of the network. `run_gpu.sh` is the execution script and the remaining files are Ascend 910.

## Preparation

In order to ensure the normal progress of the distributed training, we need to configure and initially test the distributed environment first. After completion, prepare the CIFAR-10 dataset.

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
from mindspore import set_context, GRAPH_MODE, Tensor
from mindspore.communication import init, get_rank


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.allgather = ops.AllGather()

    def construct(self, x):
        return self.allgather(x)


if __name__ == "__main__":
    set_context(mode=GRAPH_MODE, device_target="GPU")
    init("nccl")
    value = get_rank()
    input_x = Tensor(np.array([[value]]).astype(np.float32))
    net = Net()
    output = net(input_x)
    print(output)
```

In the preceding information,

- `mode=GRAPH_MODE`: sets the running mode to graph mode for distributed training. (The PyNative mode does not support parallel running.)
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

The output log can be found in `log/1/rank.0` after the program is executed. If the above output is obtained, it means that OpenMPI and NCCL are working normally, and the process is starting normally.

### Downloading the Dataset

This example uses the `CIFAR-10` dataset, which consists of 10 classes of 32*32 color pictures, each containing 6,000 pictures, for a total of 60,000 pictures. The training set has a total of 50,000 pictures, and the test set has a total of 10,000 pictures.

> `CIFAR-10` dataset download link: <http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz>. If the download is unsuccessful, please try copying the link address and download it.

The Linux machine can use the following command to download to the current path of the terminal and extract the dataset, and the folder of the extracted data is `cifar-10-batches-bin`.

```bash
wget http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -zxvf cifar-10-binary.tar.gz
```

## Loading the Dataset in the Data Parallel Mode

During distributed training, data is imported in the data parallel mode. Taking the CIFAR-10 dataset as an example, we introduce the method of importing the CIFAR-10 dataset in parallel. `data_path` refers to the path of the dataset, that is, the path of the `cifar-10-batches-bin` folder.

```python
from mindspore import dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as vision
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
    type_cast_op = C.TypeCast(mstype.int32)

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
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore.nn as nn


class SoftmaxCrossEntropyExpand(nn.Cell):
    def __init__(self, sparse=False):
        super(SoftmaxCrossEntropyExpand, self).__init__()
        self.exp = ops.Exp()
        self.sum = ops.ReduceSum(keep_dims=True)
        self.onehot = ops.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
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
        loss = self.mul2(ops.scalar_to_array(-1.0), loss)
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

```python
from mindspore import Model, ParallelMode, set_context, GRAPH_MODE, set_auto_parallel_context
from mindspore.nn import Momentum
from mindspore import LossMonitor
from mindspore.communication import init
from resnet import resnet50

set_context(mode=GRAPH_MODE, device_target="GPU")
init("nccl")


def test_train_cifar(epoch_size=10):
    set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True)
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
DEVICE2 slots=8
```

The following is the execution script of the 16-device two-host cluster. The variables `DATA_PATH` and `HOSTFILE` need to be transferred, indicating the dataset path and hostfile path. We need to set the btl parameter of mca in mpi to specify the network card that communicates with mpi, otherwise it may fail to initialize when calling the mpi interface. The btl parameter specifies the TCP protocol between nodes and the loop within the nodes for communication. The IP address of the network card through which the specified nodes communication of  btl_tcp_if_include needs to be in the given subnet. For details about more mpirun options, see the OpenMPI official website.

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

Considering that some environment variables on different machines may be different, we take the form of mpirun to start a `mpirun_gpu_cluster.sh` and specify the required environment variables in the script file on different machines. Here we have configured the `NCCL_SOCKET_IFNAME` to specify the NIC when the NCCL communicates.

```bash
#!/bin/bash
# mpirun_gpu_clusher.sh
# Here you can set different environment variables on each machine, such as the name of the network card below

export NCCL_SOCKET_IFNAME="en5" # The name of the network card that needs to communicate between nodes may be inconsistent on different machines, and use ifconfig to view.
pytest -s -v ./resnet50_distributed_training_gpu.py > train.log 2>&1 &
```

## Saving and Loading the Distributed Training Model Parameter

When performing distributed training on a GPU, the method of saving and loading the model parameters is the same as that on Ascend, which can be referred to [Distributed Training Model Parameters Saving and Loading](https://www.mindspore.cn/tutorials/experts/en/master/parallel/train_ascend.html#distributed-training-model-parameters-saving-and-loading).

## Training without Relying on OpenMPI

Due to training safety and reliability requirements, MindSpore GPUs also support **distributed training without relying on OpenMPI**.

OpenMPI plays the role of synchronizing data and inter-process networking on the Host side in distributed training scenarios. MindSpore replaces openMPI capabilities by **reusing the Parameter Server mode training architecture**.

Refer to the [Parameter Server Mode](https://www.mindspore.cn/docs/en/master/design/parameter_server_training.html) training tutorial to start multiple MindSpore training processes as `Workers`, and start an additional `Scheduler` with minor modifications to the script. You can perform **distributed training without relying on OpenMPI**.

Before executing the Worker script, you need to export environment variables, such as [Environment Variable Settings](https://www.mindspore.cn/docs/en/master/design/parameter_server_training.html#environment-variable-setting):

```text
export MS_SERVER_NUM=0                # Server number
export MS_WORKER_NUM=8                # Worker number
export MS_SCHED_HOST=127.0.0.1        # Scheduler IP address
export MS_SCHED_PORT=6667             # Scheduler port
export MS_ROLE=MS_WORKER              # The role of this process: MS_SCHED represents the scheduler, MS_WORKER represents the worker, MS_PSERVER represents the Server
```

> In this mode, it is not recommended to start a process that MS_PSERVER role because this role has no effect in data parallel training.

### Running the Script

On GPU hardware platform, the following shows how to run a distributed training script by using 8 cards as an example:

> You can find the running directory of the sample here:
>
> <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_training>.

Compared with openMPI mode startup, this mode requires calling the `set_ps_context` interface in [Parameter Server mode](https://www.mindspore.cn/docs/en/master/design/parameter_server_training.html). This mission of MindSpore uses the PS mode training architecture:

```python
from mindspore import set_context, GRAPH_MODE, set_ps_context, set_auto_parallel_context, ParallelMode
from mindspore.communication import init

if __name__ == "__main__":
    set_context(mode=GRAPH_MODE, device_target="GPU")
    set_ps_context(config_file_path="/path/to/config_file.json", enable_ssl=True,
                           client_password="123456", server_password="123456")
    init("nccl")
    ...
```

Where:

- `mode=GRAPH_MODE`: uses the distributed training, which requires specifying the run mode as graph mode (PyNative mode does not support parallelism).
- `init("nccl")`: enables NCCL communication and completes distributed training initialization.
- By default, the secure encrypted channel is closed, and the secure encrypted channel needs to be configured correctly through the `set_ps_context` or the secure encrypted channel must be closed before init ("nccl" can be called, otherwise the initialization of the networking will fail.

To use a secure encrypted tunnel, set the configuration of `set_ps_context(config_file_path="/path/to/config_file.json", enable_ssl=True, client_password="123456", server_password="123456")`. For detailed parameter configurations, refer to [mindspore.set_ps_context](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.set_ps_context.html#mindspore.set_ps_context), and [Safety Certification](#security-authentication) section.

The script content `run_gpu_cluster.sh` is as follows, before starting the Worker and Scheduler, you need to add the relevant environment variable settings:

```bash
#!/bin/bash

echo "=========================================="
echo "Please run the script as: "
echo "bash run_gpu_cluster.sh DATA_PATH"
echo "For example: bash run_gpu_cluster.sh /path/dataset"
echo "It is better to use the absolute path."
echo "==========================================="
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}

rm -rf device
mkdir device
cp ./resnet50_distributed_training_gpu.py ./resnet.py ./device
cd ./device
echo "start training"

# Launch 8 workers.
for((i=0;i<8;i++));
do
    export MS_WORKER_NUM=8
    export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
    export MS_SCHED_PORT=XXXX             # Scheduler port
    export MS_ROLE=MS_WORKER
    pytest -s -v ./resnet50_distributed_training_gpu.py > worker_$i.log 2>&1 &
done

# Launch 1 scheduler.
export MS_WORKER_NUM=8
export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
export MS_SCHED_PORT=XXXX             # Scheduler port
export MS_ROLE=MS_SCHED
pytest -s -v ./resnet50_distributed_training_gpu.py > scheduler.log 2>&1 &
```

Execute the following commands:

```bash
./run_gpu_cluster.sh DATA_PATH
```

If you want to perform cross-machine training, you need to split the script, such as performing 2-host 8-card training, and each machine performs the start of 4Worker:

The script `run_gpu_cluster_1.sh` starts 1 `Scheduler` and `Worker1` to `Worker4` on machine 1:

```bash
#!/bin/bash

echo "=========================================="
echo "Please run the script as: "
echo "bash run_gpu_cluster.sh DATA_PATH"
echo "For example: bash run_gpu_cluster.sh /path/dataset"
echo "It is better to use the absolute path."
echo "==========================================="
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}

rm -rf device
mkdir device
cp ./resnet50_distributed_training_gpu.py ./resnet.py ./device
cd ./device
echo "start training"

# Launch 1-4 workers.
for((i=0;i<4;i++));
do
    export MS_WORKER_NUM=8
    export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
    export MS_SCHED_PORT=XXXX             # Scheduler port
    export MS_ROLE=MS_WORKER
    pytest -s -v ./resnet50_distributed_training_gpu.py > worker_$i.log 2>&1 &
done

# Launch 1 scheduler.
export MS_WORKER_NUM=8
export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
export MS_SCHED_PORT=XXXX             # Scheduler port
export MS_ROLE=MS_SCHED
pytest -s -v ./resnet50_distributed_training_gpu.py > scheduler.log 2>&1 &
```

Script `run_gpu_cluster_2.sh` starts `Worker5` to `Worker8` on machine 2 (no longer needs to execute Scheduler):

```bash
#!/bin/bash

echo "=========================================="
echo "Please run the script as: "
echo "bash run_gpu_cluster.sh DATA_PATH"
echo "For example: bash run_gpu_cluster.sh /path/dataset"
echo "It is better to use the absolute path."
echo "==========================================="
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}

rm -rf device
mkdir device
cp ./resnet50_distributed_training_gpu.py ./resnet.py ./device
cd ./device
echo "start training"

# Launch 5-8 workers.
for((i=4;i<8;i++));
do
    export MS_WORKER_NUM=8
    export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
    export MS_SCHED_PORT=XXXX             # Scheduler port
    export MS_ROLE=MS_WORKER
    pytest -s -v ./resnet50_distributed_training_gpu.py > worker_$i.log 2>&1 &
done
```

Execute on the two hosts separately:

```bash
./run_gpu_cluster_1.sh DATA_PATH
```

```bash
./run_gpu_cluster_2.sh DATA_PATH
```

that is, perform 2-host and 8-card distributed training tasks.

If you want to start data parallel mode training, you need to change the `set_auto_parallel_context` in the script `resnet50_distributed_training_gpu.py` to `DATA_PARALLEL`:

```python
set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
```

The script will run in the background, and the log file will be saved to the current directory. A total of 10 epochs are run, each of which has 234 steps, and the results of the Loss part are saved in the worker_*.log. After the loss value grep is out, the example is as follows:

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

### Security Authentication

To support SSL security authentication between nodes/processes, to enable security authentication, configure `enable_ssl=True` through python API `mindspore.set_ps_context` (false by default when not passed in, indicating that SSL security authentication is not enabled), the config.json configuration file specified config_file_path  needs to add the following fields:

```json
{
  "server_cert_path": "server.p12",
  "crl_path": "",
  "client_cert_path": "client.p12",
  "ca_cert_path": "ca.crt",
  "cipher_list": "ECDHE-R SA-AES128-GCM-SHA256:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-AES256-GCM-SHA384:DHE-RSA-AES128-GCM-SHA256:DHE-DSS-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384:DHE-DSS-AES256-GCM-SHA384:DHE-PSK-AES128-GCM-SHA256:DHE-PSK-AES256-GCM-SHA384:DHE-PSK-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:ECDHE-PSK-CHACHA20-POLY1305:DHE-RSA-AES128-CCM:DHE-RSA-AES256-CCM:DHE-RSA-CHACHA20-POLY1305:DHE-PSK-AES128-CCM:DHE-PSK-AES256-CCM:ECDHE-ECDSA-AES128-CCM:ECDHE-ECDSA-AES256-CCM:ECDHE-ECDSA-CHACHA20-POLY1305",
  "cert_expire_warning_time_in_day": 90
}
```

- server_cert_path: the server contains the path to the p12 file (SSL private certificate file) containing the ciphertext of the certificate and key on the server side.

- crl_path: the file path of the revocation list (that distinguishes between invalid and untrusted certificates and valid trusted certificates).
- client_cert_path: the client contains the path to the p12 file (SSL private certificate file) containing the ciphertext of the certificate and key on the server side.
- ca_cert_path: the root certification path.
- cipher_list: Cipher suites (list of supported SSL encryption types).
- cert_expire_warning_time_in_day: the alarm time when the certificate expires.

The key in the p12 file is stored in ciphertext, and the password needs to be passed in at startup. For specific parameters, please refer to `client_password` and `server_password` fields in the Python API [mindspore.set_ps_context](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.set_ps_context.html#mindspore.set_ps_context).

### Disaster Tolerance Recovery

Model training has high requirements for the reliability and serviceability of the distributed training architecture, and MindSpore supports data parallel disaster recovery. There are abnormal process exits in the Multicard data parallel training scenario cluster (multiple Workers and 1 Scheduler), and after being pulled up, the training task can continue to perform normally;

Scene constraints:

In the graph mode, the `MindData` is used for data sinking mode training, and the data parallel mode is turned on. The above non-`OpenMPI` method is used to pull up the worker process.

In the above scenario, if there are nodes hanging up during the training process, it is guaranteed that under the same environment variables (`MS_ENABLE_RECOVERY` and `MS_RECOVERY_PATH`). The training can continue after re-pulling the script corresponding to the corresponding process, and does not affect the precision convergence.

1)  Start Disaster Tolerance:

Enable disaster tolerance with environment variables:

```bash
export MS_ENABLE_RECOVERY=1             # enable disaster tolerance
export MS_RECOVERY_PATH=“/xxx/xxx”      # Configure the persistence path folder, and the Worker and Scheduler processes perform the necessary persistence during execution, such as recovering the node information for networking and training the intermediate state of the service
```

2) Configure the checkpoint save interval, for example:

```python
ckptconfig = CheckpointConfig(save_checkpoint_steps=100, keep_checkpoint_max=5)
ckpoint_cb = ModelCheckpoint(prefix='train', directory="./ckpt_of_rank_/"+str(get_rank()), config=ckptconfig)
```

Each Worker turns on save checkpoint and uses a different path (as in the example above, the directory setting uses the rank id to ensure that the paths are not the same) to prevent checkpoint save conflicts of the same name. checkpoint is used for abnormal process recovery and normal process rollback. Training rollback means that each worker in the cluster is restored to the state corresponding to the latest checkpoint, and the data side also falls back to the corresponding step, and then continues training. The interval between saving checkpoints is configurable, which determines the granularity of disaster recovery. The smaller the interval, the smaller the number of steps that are reverted to the last save checkpoint, but the frequent saving of checkpoints may also affect the training efficiency, and the larger the interval, the opposite effect. keep_checkpoint_max set to at least 2 (to prevent checkpoint save failure).

> The running directory of the sample:
>
> <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_training>.

The scripts involved are `run_gpu_cluster_recovery.sh`, `resnet50_distributed_training_gpu_recovery.py`, `resnet.py`. The script content `run_gpu_cluster_recovery.sh` is as follows:

```bash
#!/bin/bash

echo "=========================================="
echo "Please run the script as: "
echo "bash run_gpu_cluster_recovery.sh DATA_PATH"
echo "For example: bash run_gpu_cluster_recovery.sh /path/dataset"
echo "It is better to use the absolute path."
echo "==========================================="
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}

export MS_ENABLE_RECOVERY=1      # Enable recovery
export MS_RECOVERY_PATH=/XXX/XXX # Set recovery path

rm -rf device
mkdir device
cp ./resnet50_distributed_training_gpu_recovery.py ./resnet.py ./device
cd ./device
echo "start training"

# Launch 1 scheduler.
export MS_WORKER_NUM=8
export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
export MS_SCHED_PORT=XXXX             # Scheduler port
export MS_ROLE=MS_SCHED
export MS_NODE_ID=sched               # The node id for Scheduler
pytest -s -v ./resnet50_distributed_training_gpu_recovery.py > scheduler.log 2>&1 &

# Launch 8 workers.
for((i=0;i<8;i++));
do
    export MS_WORKER_NUM=8
    export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
    export MS_SCHED_PORT=XXXX             # Scheduler port
    export MS_ROLE=MS_WORKER
    export MS_NODE_ID=worker_$i           # The node id for Workers
    pytest -s -v ./resnet50_distributed_training_gpu_recovery.py > worker_$i.log 2>&1 &
done
```

Before starting Worker and Scheduler, you need to add the relevant environment variable settings, such as IP and Port of Scheduler, and whether the role of the current process is Worker or Scheduler.

Execute the following command to start a single-host 8-card data parallel training

```bash
bash run_gpu_cluster_recovery.sh YOUR_DATA_PATH"
```

Distributed training starts, if an exception is encountered during the training process, such as an abnormal process exit, and then restart the corresponding process, the training process can be resumed:

For example, if the Scheduler process exits abnormally during training, you can execute the following command to restart Scheduler:

```bash
export DATA_PATH=YOUR_DATA_PATH
export MS_ENABLE_RECOVERY=1           # Enable recovery
export MS_RECOVERY_PATH=/XXX/XXX      # Set recovery path

cd ./device

# Launch 1 scheduler.
export MS_WORKER_NUM=8
export MS_SCHED_HOST=XXX.XXX.XXX.XXX  # Scheduler IP address
export MS_SCHED_PORT=XXXX             # Scheduler port
export MS_ROLE=MS_SCHED
export MS_NODE_ID=sched               # The node id for Scheduler
pytest -s -v ./resnet50_distributed_training_gpu_recovery.py > scheduler.log 2>&1 &
```

Worker and Scheduler's networking is automatically restored.

The exception exit of the Worker process is handled in a similar way (Note: the Worker process has an abnormal exit, and it needs to wait for 30s to pull up before resuming training. Before that, Scheduler refuses to register the worker with the same node id again in order to prevent network jitter and malicious registration).

## Using ms-operators for Distributed Training in K8s Clusters

MindSpore Operator is a plugin for MindSpore to conduct distributed training on Kubernetes. The CRD (Custom Resource Definition) defines three roles of Scheduler, PS, and Worker, and users only need to configure the yaml file to easily implement distributed training.

The current ms-operator supports ordinary single Worker training, single Worker training in PS mode, and Scheduler and Worker startups for automatic parallelism (such as data parallelism and model parallelism). For detailed procedures, see [ms-operator](https://gitee.com/mindspore/ms-operator).



