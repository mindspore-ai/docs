# Distributed Training

## Overview

MindSpore supports `DATA_PARALLEL` and `AUTO_PARALLEL`. Automatic parallel is a distributed parallel mode that integrates data parallel, model parallel, and hybrid parallel. It can automatically establish cost models and select a parallel mode for users.

Among them:

- Data parallel: A parallel mode for dividing data in batches.
- Layerwise parallel: A parallel mode for dividing parameters by channel.
- Hybrid parallel: A parallel mode that covers both data parallel and model parallel.
- Cost model: A cost model built based on the memory computing cost and communication cost, for which an efficient algorithm is designed to find the parallel strategy with the shorter training time.

In this tutorial, we will learn how to train the ResNet-50 network in `DATA_PARALLEL` or `AUTO_PARALLEL` mode on MindSpore.

> The current sample is for the Ascend 910 AI processor. CPU and GPU processors are not supported for now.
> You can find the complete executable sample code at：<https://gitee.com/mindspore/docs/blob/r0.2/tutorials/tutorial_code/distributed_training/resnet50_distributed_training.py>.

## Preparations

### Configuring Distributed Environment Variables

When distributed training is performed in the lab environment, you need to configure the networking information file for the current multi-card environment. If HUAWEI CLOUD is used, skip this section.

The Ascend 910 AI processor and AIServer are used as an example. The JSON configuration file of a two-card environment is as follows. In this example, the configuration file is named rank_table.json.

```json
{
    "board_id": "0x0000",
    "chip_info": "910",
    "deploy_mode": "lab",
    "group_count": "1",
    "group_list": [
        {
            "device_num": "2",
            "server_num": "1",
            "group_name": "",
            "instance_count": "2",
            "instance_list": [
                     {"devices":[{"device_id":"0","device_ip":"192.1.27.6"}],"rank_id":"0","server_id":"10.155.111.140"},
                     {"devices":[{"device_id":"1","device_ip":"192.2.27.6"}],"rank_id":"1","server_id":"10.155.111.140"}
               ]
        }
    ],
    "para_plane_nic_location": "device",
    "para_plane_nic_name": [
        "eth0", "eth1"
    ],
    "para_plane_nic_num": "2",
    "status": "completed"
}

```

The following parameters need to be modified based on the actual training environment:

1. `board_id` indicates the environment in which the program runs.
2. `server_num` indicates the number of hosts, and `server_id` indicates the IP address of the local host.
3. `device_num`, `para_plane_nic_num`, and `instance_count` indicate the number of cards.
4. `rank_id` indicates the logical sequence number of a card, which starts from 0 fixedly. `device_id` indicates the physical sequence number of a card, that is, the actual sequence number of the host where the card is located.
5. `device_ip` indicates the IP address of the NIC. You can run the `cat /etc/hccn.conf` command on the current host to obtain the IP address of the NIC.
6. `para_plane_nic_name` indicates the name of the corresponding NIC.

After the networking information file is ready, add the file path to the environment variable `MINDSPORE_HCCL_CONFIG_PATH`. In addition, the `device_id` information needs to be transferred to the script. In this example, the information is transferred by configuring the environment variable DEVICE_ID.

```bash
export MINDSPORE_HCCL_CONFIG_PATH="./rank_table.json"
export DEVICE_ID=0
```

### Invoking the Collective Communication Library

You need to enable the distributed API `enable_hccl` in the `context.set_context()` API, set the `device_id` parameter, and invoke `init()` to complete the initialization operation.

In the sample, the graph mode is used during runtime. On the Ascend AI processor, Huawei Collective Communication Library (HCCL) is used.

```python
import os
from mindspore import context
from mindspore.communication.management import init

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", enable_hccl=True, device_id=int(os.environ["DEVICE_ID"]))
    init()
    ...
```

`mindspore.communication.management` encapsulates the collective communication API provided by the HCCL to help users obtain distributed information. The common types include `get_rank` and `get_group_size`, which correspond to the ID of the current card in the cluster and the number of cards, respectively.
> HCCL implements multi-device multi-card communication based on the Da Vinci architecture chip. The restrictions on using the distributed service are as follows:
>
> 1. In a single-node system, a cluster of 1, 2, 4, or 8 cards is supported. In a multi-node system, a cluster of 8 x N cards is supported.
> 2. Each server has four NICs (numbered 0 to 3) and four NICs (numbered 4 to 7) deployed on two different networks. During training of two or four cards, the NICs must be connected and clusters cannot be created across networks.
> 3. The operating system needs to use the symmetric multiprocessing (SMP) mode.

## Loading Datasets

During distributed training, data is imported in data parallel mode. The following uses Cifar10Dataset as an example to describe how to import the CIFAR-10 data set in parallel mode, `data_path` is the path of the dataset.
Different from a single-node system, the multi-node system needs to transfer `num_shards` and `shard_id` parameters to the dataset API, which correspond to the number of cards and logical sequence number of the NIC, respectively. You are advised to obtain the parameters through the HCCL API.

```python
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.transforms.vision.c_transforms as vision
from mindspore.communication.management import get_rank, get_group_size

def create_dataset(repeat_num=1, batch_size=32, rank_id=0, rank_size=1):
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

## Defining the Network

In `DATA_PARALLEL` and `AUTO_PARALLEL` modes, the network definition mode is the same as that of a single-node system. For sample code, see at

 <https://gitee.com/mindspore/docs/blob/r0.2/tutorials/tutorial_code/resnet/resnet.py>.

## Defining the Loss Function and Optimizer

### Defining the Loss Function

In the Loss function, the SoftmaxCrossEntropyWithLogits is expanded into multiple small operators for implementation according to a mathematical formula.
Compared with fusion loss, the loss in `AUTO_PARALLEL` mode searches and finds optimal parallel strategy by operator according to an algorithm.

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

The `Momentum` optimizer is used as the parameter update tool. The definition is the same as that of a single-node system.

```python
from mindspore.nn.optim.momentum import Momentum
lr = 0.01
momentum = 0.9
opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, momentum)
```

## Training the Network

`context.set_auto_parallel_context()` is an API provided for users to set parallel parameters, which can be invoked only before the initialization of `Model`. If users did not set parameters, MindSpore will automatically set parameters to the empirical values according to the parallel mode. For example, `parameter_broadcast` is `True` in data parallel mode. The parameters are as follows:

- `parallel_mode`: distributed parallel mode. The default value is `ParallelMode.STAND_ALONE`. The options are `ParallelMode.DATA_PARALLEL` and `ParallelMode.AUTO_PARALLEL`.
- `paramater_broadcast`： specifies whether to broadcast initialized parameters. The Default value is `False` in non-data parallel mode.
- `mirror_mean`: During backward computation, the framework collects gradients of parameters in data parallel mode across multiple machines, obtains the global gradient value, and transfers the global gradient value to the optimizer for update. The default value is `False`, which indicates the `allreduce_sum` operation that would be applied. And the value `True` indicates the `allreduce_mean` operation that would be applied.

In the following example, the parallel mode is set to `AUTO_PARALLEL`. `dataset_sink_mode=False` indicates that the non-sink mode is used. `LossMonitor` can return the loss value through the callback function.

```python
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import LossMonitor
from mindspore.train.model import Model, ParallelMode
from resnet import resnet50

def test_train_cifar(num_classes=10, epoch_size=10):
    context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, mirror_mean=True)
    loss_cb = LossMonitor()
    dataset = create_dataset(epoch_size)
    net = resnet50(32, num_classes)
    loss = SoftmaxCrossEntropyExpand(sparse=True)
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    model = Model(net, loss_fn=loss, optimizer=opt)
    model.train(epoch_size, dataset, callbacks=[loss_cb], dataset_sink_mode=False)
```

## Running Test Cases

Currently, MindSpore distributed execution uses the single-card single-process running mode. The number of processes must be the same as the number of used cards. Each single-process will create a folder to save log and building information. The following is an example of a running script for two-card distributed training:

```bash
  #!/bin/bash

  export MINDSPORE_HCCL_CONFIG_PATH=./rank_table.json
  export RANK_SIZE=2
  for((i=0;i<$RANK_SIZE;i++))
  do
      mkdir device$i
      cp ./resnet50_distributed_training.py ./device$i
      cd ./device$i
      export RANK_ID=$i
      export DEVICE_ID=$i
      echo "start training for device $i"
      env > env$i.log
      pytest -s -v ./resnet50_distributed_training.py > log$i 2>&1 &
      cd ../
  done
```
