# Parallel Distributed Training (Ascend)

<a href="https://gitee.com/mindspore/docs/blob/r0.6/tutorials/source_en/advanced_use/distributed_training_ascend.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

This tutorial describes how to train the ResNet-50 network in data parallel and automatic parallel modes on MindSpore based on the Ascend 910 AI processor.
> Download address of the complete sample code: <https://gitee.com/mindspore/docs/blob/r0.6/tutorials/tutorial_code/distributed_training/resnet50_distributed_training.py>

## Preparations

### Downloading the Dataset

This sample uses the `CIFAR-10` dataset, which consists of color images of 32 x 32 pixels in 10 classes, with 6000 images per class. There are 50,000 images in the training set and 10,000 images in the test set.

> `CIFAR-10` dataset download address: <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>

Download the dataset and decompress it to a local path. The folder generated after the decompression is `cifar-10-batches-bin`.

### Configuring Distributed Environment Variables

When distributed training is performed in the bare-metal environment (compared with the cloud environment where the Ascend 910 AI processor is deployed on the local host), you need to configure the networking information file for the current multi-device environment. If the HUAWEI CLOUD environment is used, skip this section because the cloud service has been configured.

The following uses the Ascend 910 AI processor as an example. The JSON configuration file for an environment with eight devices is as follows. In this example, the configuration file is named `rank_table_8pcs.json`. For details about how to configure the 2-device environment, see the `rank_table_2pcs.json` file in the sample code.

```json
{
    "version": "1.0",
    "server_count": "1",
    "server_list": [
        {
            "server_id": "10.*.*.*",
            "device": [
                {"device_id": "0","device_ip": "192.1.27.6","rank_id": "0"},
                {"device_id": "1","device_ip": "192.2.27.6","rank_id": "1"},
                {"device_id": "2","device_ip": "192.3.27.6","rank_id": "2"},
                {"device_id": "3","device_ip": "192.4.27.6","rank_id": "3"},
                {"device_id": "4","device_ip": "192.1.27.7","rank_id": "4"},
                {"device_id": "5","device_ip": "192.2.27.7","rank_id": "5"},
                {"device_id": "6","device_ip": "192.3.27.7","rank_id": "6"},
                {"device_id": "7","device_ip": "192.4.27.7","rank_id": "7"}],
             "host_nic_ip": "reserve"
        }
    ],
    "status": "completed"
}
```
The following parameters need to be modified based on the actual training environment:

- `server_count`: number of hosts.
- `server_id`: IP address of the local host.
- `device_id`: physical sequence number of a device, that is, the actual sequence number of the device on the corresponding host.
- `device_ip`: IP address of the integrated NIC. You can run the `cat /etc/hccn.conf` command on the current host. The key value of `address_x` is the IP address of the NIC.
- `rank_id`: logical sequence number of a device, which starts from 0.


### Calling the Collective Communication Library

The Huawei Collective Communication Library (HCCL) is used for the communication of MindSpore parallel distributed training and can be found in the Ascend 310 AI processor software package. In addition, `mindspore.communication.management` encapsulates the collective communication API provided by the HCCL to help users configure distributed information.
> HCCL implements multi-device multi-node communication based on the Ascend AI processor. The common restrictions on using the distributed service are as follows. For details, see the HCCL documentation.
> - In a single-node system, a cluster of 1, 2, 4, or 8 devices is supported. In a multi-node system, a cluster of 8 x N devices is supported.
> - Each host has four devices numbered 0 to 3 and four devices numbered 4 to 7 deployed on two different networks. During training of 2 or 4 devices, the devices must be connected and clusters cannot be created across networks.
> - The server hardware architecture and operating system require the symmetrical multi-processing (SMP) mode.

The sample code for calling the HCCL as follows:

```python
import os
from mindspore import context
from mindspore.communication.management import init

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=int(os.environ["DEVICE_ID"]))
    init()
    ...   
```

In the preceding code:  
- `mode=context.GRAPH_MODE`: sets the running mode to graph mode for distributed training. (The PyNative mode does not support parallel running.)
- `device_id`: physical sequence number of a device, that is, the actual sequence number of the device on the corresponding host.
- `init`: enables HCCL communication and completes the distributed training initialization.

## Loading the Dataset in Data Parallel Mode

During distributed training, data is imported in data parallel mode. The following takes the CIFAR-10 dataset as an example to describe how to import the CIFAR-10 dataset in data parallel mode. `data_path` indicates the dataset path, which is also the path of the `cifar-10-batches-bin` folder.


```python
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.transforms.vision.c_transforms as vision
from mindspore.communication.management import get_rank, get_group_size

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
    data_set = data_set.map(input_columns="label", operations=type_cast_op)
    data_set = data_set.map(input_columns="image", operations=c_trans)

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=10)

    # apply batch operations
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    # apply repeat operations
    data_set = data_set.repeat(repeat_num)

    return data_set
```
Different from the single-node system, the multi-node system needs to transfer the `num_shards` and `shard_id` parameters to the dataset API. The two parameters correspond to the number of devices and logical sequence numbers of devices, respectively. You are advised to obtain the parameters through the HCCL API.  
- `get_rank`: obtains the ID of the current device in the cluster.
- `get_group_size`: obtains the number of devices.

## Defining the Network

In data parallel and automatic parallel modes, the network definition method is the same as that in a single-node system. The reference code is as follows: <https://gitee.com/mindspore/docs/blob/r0.6/tutorials/tutorial_code/resnet/resnet.py>

## Defining the Loss Function and Optimizer

### Defining the Loss Function

Automatic parallelism splits models using the operator granularity and obtains the optimal parallel strategy through algorithm search. Therefore, to achieve a better parallel training effect, you are advised to use small operators to implement the loss function.

In the Loss function, the `SoftmaxCrossEntropyWithLogits` is expanded into multiple small operators for implementation according to a mathematical formula. The sample code is as follows:

```python
from mindspore.ops import operations as P
from mindspore import Tensor
import mindspore.ops.functional as F
import mindspore.common.dtype as mstype
import mindspore.nn as nn

class SoftmaxCrossEntropyExpand(nn.Cell):
    def __init__(self, sparse=False):
        super(SoftmaxCrossEntropyExpand, self).__init__()
        self.exp = P.Exp()
        self.sum = P.ReduceSum(keep_dims=True)
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.div = P.Div()
        self.log = P.Log()
        self.sum_cross_entropy = P.ReduceSum(keep_dims=False)
        self.mul = P.Mul()
        self.mul2 = P.Mul()
        self.mean = P.ReduceMean(keep_dims=False)
        self.sparse = sparse
        self.max = P.ReduceMax(keep_dims=True)
        self.sub = P.Sub()
        
    def construct(self, logit, label):
        logit_max = self.max(logit, -1)
        exp = self.exp(self.sub(logit, logit_max))
        exp_sum = self.sum(exp, -1)
        softmax_result = self.div(exp, exp_sum)
        if self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)
        softmax_result_log = self.log(softmax_result)
        loss = self.sum_cross_entropy((self.mul(softmax_result_log, label)), -1)
        loss = self.mul2(F.scalar_to_array(-1.0), loss)
        loss = self.mean(loss, -1)

        return loss
```

### Defining the Optimizer

The `Momentum` optimizer is used as the parameter update tool. The definition is the same as that in the single-node system. For details, see the implementation in the sample code.

## Training the Network

`context.set_auto_parallel_context` is an API for users to set parallel training parameters and must be called before the initialization of `Model`. If no parameters are specified, MindSpore will automatically set parameters to the empirical values based on the parallel mode. For example, in data parallel mode, `parameter_broadcast` is enabled by default. The related parameters are as follows:

- `parallel_mode`: parallel distributed mode. The default value is `ParallelMode.STAND_ALONE`. The options are `ParallelMode.DATA_PARALLEL` and `ParallelMode.AUTO_PARALLEL`.
- `parameter_broadcast`: whether to broadcast initialized parameters. The default value is `True` in `DATA_PARALLEL` and `HYBRID_PARALLEL` mode.
- `mirror_mean`: During backward computation, the framework collects gradients of parameters in data parallel mode across multiple hosts, obtains the global gradient value, and transfers the global gradient value to the optimizer for update. The default value is `False`, which indicates that the `allreduce_sum` operation is applied. The value `True` indicates that the `allreduce_mean` operation is applied.
- `enable_parallel_optimizer`: a developing feature. Whether to use optimizer model parallel, which improves performance by distributing the parameters to be updated to each worker, and applying Broadcast among workers to share updated parameters. This feature can be used only in data parallel mode and when the number of parameters is larger than the number of devices.

> You are advised to set `device_num` and `global_rank` to their default values. The framework calls the HCCL API to obtain the values.

If multiple network cases exist in the script, call `context.reset_auto_parallel_context()` to restore all parameters to default values before executing the next case.

In the following sample code, the automatic parallel mode is specified. To switch to the data parallel mode, you only need to change `parallel_mode` to `DATA_PARALLEL`.

```python
from mindspore import context
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import LossMonitor
from mindspore.train.model import Model, ParallelMode
from resnet import resnet50

device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
context.set_context(device_id=device_id) # set device_id

def test_train_cifar(epoch_size=10):
    context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, mirror_mean=True)
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
In the preceding code:  
- `dataset_sink_mode=True`: uses the dataset sink mode. That is, the training computing is sunk to the hardware platform for execution.
- `LossMonitor`: returns the loss value through the callback function to monitor the loss function.

## Running the Script
After the script required for training is edited, run the corresponding command to call the script.

Currently, MindSpore distributed execution uses the single-device single-process running mode. That is, one process runs on each device, and the number of total processes is the same as the number of devices that are being used. For device 0, the corresponding process is executed in the foreground. For other devices, the corresponding processes are executed in the background. You need to create a directory for each process to store log information and operator compilation information. The following takes the distributed training script for eight devices as an example to describe how to run the script:

```bash
#!/bin/bash

DATA_PATH=$1
export DATA_PATH=${DATA_PATH}
RANK_SIZE=$2

EXEC_PATH=$(pwd)

test_dist_8pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json
    export RANK_SIZE=8
}

test_dist_2pcs()
{
    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_2pcs.json
    export RANK_SIZE=2
}

test_dist_${RANK_SIZE}pcs

for((i=1;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cp ./resnet50_distributed_training.py ./resnet.py ./device$i
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    pytest -s -v ./resnet50_distributed_training.py > train.log$i 2>&1 &
    cd ../
done
rm -rf device0
mkdir device0
cp ./resnet50_distributed_training.py ./resnet.py ./device0
cd ./device0
export DEVICE_ID=0
export RANK_ID=0
echo "start training for device 0"
env > env0.log
pytest -s -v ./resnet50_distributed_training.py > train.log0 2>&1
if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
cd ../
```

The variables `DATA_PATH` and `RANK_SIZE` need to be transferred to the script, which indicate the path of the dataset and the number of devices, respectively.

The necessary environment variables are as follows:  
- `RANK_TABLE_FILE`: path for storing the networking information file.
- `DEVICE_ID`: actual sequence number of the current device on the corresponding host.
- `RANK_ID`: logical sequence number of the current device.
For details about other environment variables, see configuration items in the installation guide.

The running time is about 5 minutes, which is mainly occupied by operator compilation. The actual training time is within 20 seconds. You can use `ps -ef | grep pytest` to monitor task processes.

Log files are saved in the `device` directory. The `env.log` file records environment variable information. The `train.log` file records the loss function information. The following is an example:

```
epoch: 1 step: 156, loss is 2.0084016
epoch: 2 step: 156, loss is 1.6407638
epoch: 3 step: 156, loss is 1.6164391
epoch: 4 step: 156, loss is 1.6838071
epoch: 5 step: 156, loss is 1.6320667
epoch: 6 step: 156, loss is 1.3098773
epoch: 7 step: 156, loss is 1.3515002
epoch: 8 step: 156, loss is 1.2943741
epoch: 9 step: 156, loss is 1.2316195
epoch: 10 step: 156, loss is 1.1533381
```
