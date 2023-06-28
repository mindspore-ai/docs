# 分布式并行训练

## 概述

MindSpore支持数据并行及自动并行。自动并行是MindSpore融合了数据并行、模型并行及混合并行的一种分布式并行模式，可以自动建立代价模型，为用户选择一种并行模式。

其中：

- 数据并行（Data Parallel）：对数据batch维度切分的一种并行模式。
- 模型并行（Layerwise Parallel）：对参数channel维度切分的一种并行模式。
- 混合并行（Hybrid Parallel）：涵盖数据并行和模型并行的一种并行模式。
- 代价模型（Cost Model）：同时考虑内存的计算代价和通信代价对训练时间建模，并设计了高效的算法来找到训练时间较短的并行策略。

本篇教程我们主要了解如何在MindSpore上通过数据并行及自动并行模式训练ResNet-50网络。
样例代码请参考 <https://gitee.com/mindspore/docs/blob/r0.1/tutorials/tutorial_code/distributed_training/resnet50_distributed_training.py>。

> 当前样例面向Ascend AI处理器。

## 准备环节

### 配置分布式环境变量

在实验室环境进行分布式训练时，需要配置当前多卡环境的组网信息文件。如果使用华为云环境，可以跳过本小节。

以Ascend 910 AI处理器、1980 AIServer为例，一个两卡环境的json配置文件示例如下，本样例将该配置文件命名为rank_table.json。

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

其中需要根据实际训练环境修改的参数项有：

1. `server_num`表示机器数量， `server_id`表示本机IP地址。
2. `device_num`、`para_plane_nic_num`及`instance_count`表示卡的数量。
3. `rank_id`表示卡逻辑序号，固定从0开始编号，`device_id`表示卡物理序号，即卡所在机器中的实际序号。
4. `device_ip`表示网卡IP地址，可以在当前机器执行指令`cat /etc/hccn.conf`获取网卡IP地址。
5. `para_plane_nic_name`对应网卡名称。

组网信息文件准备好后，将文件路径加入环境变量`MINDSPORE_HCCL_CONFIG_PATH`中。此外需要将`device_id`信息传入脚本中，本样例通过配置环境变量DEVICE_ID的方式传入。

```bash
export MINDSPORE_HCCL_CONFIG_PATH="./rank_table.json"
export DEVICE_ID=0
```

### 调用集合通信库

我们需要在`context.set_context()`接口中使能分布式接口`enable_hccl`，设置`device_id`参数，并通过调用`init()`完成初始化操作。

在样例中，我们指定运行时使用图模式，在Ascend AI处理器上，使用华为集合通信库`Huawei Collective Communication Library`（以下简称HCCL）。

```python
import os
from mindspore import context
from mindspore.communication.management import init

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", enable_hccl=True, device_id=int(os.environ["DEVICE_ID"]))
    init()
    ...
```

`mindspore.communication.management`中封装了HCCL提供的集合通信接口，方便用户获取分布式信息。常用的包括`get_rank`和`get_group_size`，分别对应当前设备在集群中的ID和集群数量。

> HCCL实现了基于Davinci架构芯片的多机多卡通信。当前使用分布式服务存在如下约束：
>
> 1. 单机场景下支持1、2、4、8卡设备集群，多机场景下支持8*n卡设备集群。
> 2. 每台机器的0-3卡和4-7卡各为一个组网，2卡和4卡训练时网卡必须相连且不支持跨组网创建集群。
> 3. 操作系统需使用SMP (symmetric multiprocessing)处理模式。

## 加载数据集

分布式训练时，数据是以数据并行的方式导入的。下面我们以Cifar10Dataset为例，介绍以数据并行方式导入CIFAR-10数据集的方法，`data_path`是指数据集的路径。
与单机不同的是，在数据集接口需要传入`num_shards`和`shard_id`参数，分别对应网卡数量和逻辑序号，建议通过HCCL接口获取。

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

    # apply repeat operations
    data_set = data_set.repeat(repeat_num)

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=10)

    # apply batch operations
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    return data_set
```

## 定义网络

数据并行及自动并行模式下，网络定义方式与单机一致。代码请参考： <https://gitee.com/mindspore/docs/blob/r0.1/tutorials/tutorial_code/resnet/resnet.py>

## 定义损失函数及优化器

### 定义损失函数

在Loss部分，我们采用SoftmaxCrossEntropyWithLogits的展开形式，即按照数学公式，将其展开为多个小算子进行实现。
相较于融合loss，自动并行以展开loss中的算子为粒度，通过算法搜索得到最优并行策略。

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

### 定义优化器

采用`Momentum`优化器作为参数更新工具，这里定义与单机一致。

```python
from mindspore.nn.optim.momentum import Momentum
lr = 0.01
momentum = 0.9
opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, momentum)
```

## 训练网络

`context.set_auto_parallel_context()`是提供给用户设置并行参数的接口。主要参数包括：

- `parallel_mode`：分布式并行模式。可选数据并行`ParallelMode.DATA_PARALLEL`及自动并行`ParallelMode.AUTO_PARALLEL`。
- `mirror_mean`: 反向计算时，框架内部会将数据并行参数分散在多台机器的梯度进行收集，得到全局梯度值后再传入优化器中更新。

设置为True对应`allreduce_mean`操作，False对应`allreduce_sum`操作。

在下面的样例中我们指定并行模式为自动并行，其中`dataset_sink_mode=False`表示采用数据非下沉模式，`LossMonitor`能够通过回调函数返回loss值。

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

## 运行测试用例

目前MindSpore分布式执行采用单卡单进程运行方式，进程数量应当与卡的使用数量保持一致。每个进程创建一个目录，用来保存日志信息以及算子编译信息。下面以一个2卡分布式训练的运行脚本为例：

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
