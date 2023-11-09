# 手动并行

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/tutorials/experts/source_zh_cn/parallel/manual_parallel.md)

## 概述

除了MindSpore提供的自动并行和半自动并行，用户还可以基于通信原语来编码并行过程，手动把模型切分到多个节点上并行。在这种手动并行模式中，用户需要感知图切分、算子切分、集群拓扑，才能实现最优性能。

## 基本原理

MindSpore的集合通信算子包括`AllReduce`、`AllGather`、`ReduceScatter`、`Broadcast`、`NeighborExchange`、`NeighborExchangeV2`、`AlltoAll`，这些算子是分布式训练中集合通信的基本组成单元。所谓集合通信是指模型切分后，通过集合通信算子来实现不同模型切片之间的数据交互。用户可以手动调用这些算子进行数据传输，实现分布式训练。

集合通信算子的详细介绍参见[分布式集合通信原语](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/samples/ops/communicate_ops.html)。

## 操作实践

下面以Ascend或者GPU单机8卡为例，进行手动数据并行操作说明：

### 样例代码说明

> 下载完整的样例代码：[manual_parallel](https://gitee.com/mindspore/docs/tree/r2.3/docs/sample_code/manual_parallel)。

目录结构如下：

```text
└─ sample_code
    ├─ manual_parallel
       ├── train.py
       └── run.sh
    ...
```

其中，`train.py`是定义网络结构和训练过程的脚本。`run.sh`是执行脚本。

### 配置分布式环境

通过init初始化HCCL或NCCL通信，并设置随机种子，由于是手动并行，此处不指定任何并行模式。`get_rank()`接口可以获取当前设备在通信组中的rank_id，`get_group_size()`接口获取当前通信组的设备数量，通信组默认为全局通信组，包含所有设备。

```python
import mindspore as ms
from mindspore.communication import init, get_rank, get_group_size

ms.set_context(mode=ms.GRAPH_MODE)
init()
cur_rank = get_rank()
batch_size = 32
device_num = get_group_size()
shard_size = batch_size // device_num
```

### 网络定义

在单卡网络的基础上，增加了对输入数据的切分：

```python
from mindspore import nn
from mindspore.communication import get_rank, get_group_size

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Dense(28*28, 512)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Dense(512, 512)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Dense(512, 10)

    def construct(self, x):
        x = x[cur_rank*shard_size:cur_rank*shard_size + shard_size]
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        logits = self.layer3(x)
        return logits

net = Network()
```

### 数据集加载

数据集加载方式与单卡网络一致：

```python
import os
import mindspore.dataset as ds

def create_dataset():
    dataset_path = os.getenv("DATA_PATH")
    dataset = ds.MnistDataset(dataset_path)
    image_transforms = [
        ds.vision.Rescale(1.0 / 255.0, 0),
        ds.vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        ds.vision.HWC2CHW()
    ]
    label_transform = ds.transforms.TypeCast(ms.int32)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

data_set = create_dataset()
```

### 损失函数定义

在损失函数中，需要增加对label的切分，以及通信原语算子`ops.AllReduce`来聚合各卡的损失：

```python
from mindspore import nn, ops
from mindspore.communication import get_rank, get_group_size

class ReduceLoss(nn.Cell):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.all_reduce = ops.AllReduce()

    def construct(self, data, label):
        label = label[cur_rank*shard_size:cur_rank*shard_size + shard_size]
        loss_value = self.loss(data, label)
        loss_value = self.all_reduce(loss_value) / device_num
        return loss_value

loss_fn = ReduceLoss()
```

### 训练过程定义

优化器、训练过程与单卡网络一致：

```python
import mindspore as ms
from mindspore import nn, train

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_cb = train.LossMonitor(20)
model = ms.Model(net, loss_fn=loss_fn, optimizer=optimizer)
model.train(10, data_set, callbacks=[loss_cb])
```

### 运行单机8卡脚本

接下来通过命令调用对应的脚本，以`mpirun`启动方式，8卡的分布式训练脚本为例，进行分布式训练：

```bash
bash run.sh
```

训练完后，日志文件保存到`log_output`目录下，通过在`train.py`中设置context: `save_graphs=2`，可以打印出编译过程中的IR图，其中部分文件目录结构如下：

```text
└─ log_output
    └─ 1
        ├─ rank.0
        |   └─ stdout
        ├─ rank.1
        |   └─ stdout
        ...
```

关于Loss部分结果保存在`log_output/1/rank.*/stdout`中，示例如下：

```text
epoch: 1 step: 20, loss is 2.241283893585205
epoch: 1 step: 40, loss is 2.1842331886291504
epoch: 1 step: 60, loss is 2.0627782344818115
epoch: 1 step: 80, loss is 1.9561686515808105
epoch: 1 step: 100, loss is 1.8991656303405762
epoch: 1 step: 120, loss is 1.6239635944366455
epoch: 1 step: 140, loss is 1.465965747833252
epoch: 1 step: 160, loss is 1.3662006855010986
epoch: 1 step: 180, loss is 1.1562917232513428
epoch: 1 step: 200, loss is 1.116426944732666
...
```
