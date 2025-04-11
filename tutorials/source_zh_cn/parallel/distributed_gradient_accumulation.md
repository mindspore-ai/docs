# 梯度累加

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_zh_cn/parallel/distributed_gradient_accumulation.md)

## 简介

梯度累加是一种能够在内存受限的情况下，采用更大Batch Size来训练网络的优化技术。通常情况下，训练大型神经网络需要大量的内存，因为在每个Batch上计算梯度并更新模型参数需要保存梯度值。Batch Size越大需要的内存越大，可能会导致内存不足的问题。梯度累加通过将多个MicroBatch的梯度值相加，从而允许在不增加内存需求的情况下以更大的Batch Size训练模型。本文主要介绍分布式场景下的梯度累加。

### 基本原理

梯度累加的核心思想是将多个MicroBatch的梯度相加，然后使用累加的梯度来更新模型参数。下面是梯度累加的步骤：

1. 选择MicroBatch大小：MicroBatch大小的数据是每一次正反向传播的基本批次，同时根据Batch Size除以Micro Batch Size得到累加步数，可以确定在多少个MicroBatch之后进行一次参数更新。

2. 前向传播和反向传播：对于每个MicroBatch，执行标准的前向传播和反向传播操作。计算小批次的梯度。

3. 梯度累加：将每个MicroBatch的梯度值相加，直到达到累加步数。

4. 梯度更新：在达到累加步数后，使用累加的梯度来通过优化器更新模型参数。

5. 梯度清零：在梯度更新后，将梯度值清零，以便下一个累加周期的计算。

### 相关接口

`mindspore.parallel.GradAccumulation(network, micro_size)`：用更细粒度的MicroBatch包装网络。`micro_size`是MicroBatch的大小。

> - 在梯度累加场景下，推荐使用lazy_inline装饰器来缩短编译时间，并且仅支持将lazy_inline装饰器配置在最外层的Cell上。

## 操作实践

下面以Ascend单机8卡为例，进行梯度累加操作说明：

### 样例代码说明

> 下载完整的样例代码：[distributed_gradient_accumulation](https://gitee.com/mindspore/docs/tree/r2.6.0/docs/sample_code/distributed_gradient_accumulation)。

目录结构如下：

```text
└─ sample_code
    ├─ distributed_gradient_accumulation
       ├── train.py
       └── run.sh
    ...
```

其中，`train.py`是定义网络结构和训练过程的脚本。`run.sh`是执行脚本。

### 配置分布式环境

通过init初始化HCCL通信。

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
init()
```

### 数据集加载与网络定义

此处数据集加载和网络定义与单卡模型一致，通过 `no_init_parameters` 接口延后初始化网络参数和优化器参数。代码如下：

```python
import os
import mindspore.dataset as ds
from mindspore import nn
from mindspore.parallel.auto_parallel import AutoParallel
from mindspore.nn.utils import no_init_parameters

def create_dataset(batch_size):
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

data_set = create_dataset(32)

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(512, 512, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(512, 10, weight_init="normal", bias_init="zeros")
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

with no_init_parameters():
    net = Network()
    optimizer = nn.SGD(net.trainable_params(), 1e-2)
```

### 训练网络

在这一步，我们需要定义损失函数以及训练过程，通过顶层 `AutoParallel` 接口设置并行模式为半自动并行模式和优化器并行，调用两个接口来配置梯度累加：

- 首先需要定义LossCell，本例中调用了`nn.WithLossCell`接口封装网络和损失函数。
- 然后需要在LossCell外包一层`GradAccumulation`，并指定MicroBatch的size为4。详细请参考本章概述中的相关接口。

```python
import mindspore as ms
from mindspore import nn, train
from mindspore.parallel import GradAccumulation

loss_fn = nn.CrossEntropyLoss()
loss_cb = train.LossMonitor(100)
net = GradAccumulation(nn.WithLossCell(net, loss_fn), 4)
# set paralllel mode and enable parallel optimizer
net = AutoParallel(net)
net.hsdp()
model = ms.Model(net, optimizer=optimizer)
model.train(10, data_set, callbacks=[loss_cb])
```

> 梯度累加训练更适合用`model.train`的方式，这是因为梯度累加下的TrainOneStep逻辑复杂，而`model.train`内部封装了针对梯度累加的TrainOneStepCell，易用性更好。

### 运行单机8卡脚本

接下来通过命令调用对应的脚本，以`msrun`启动方式，8卡的分布式训练脚本为例，进行分布式训练：

```bash
bash run.sh
```

训练完后，关于Loss部分结果保存在`log_output/worker_*.log`中，示例如下：

```text
epoch: 1 step: 100, loss is 7.793933868408203
epoch: 1 step: 200, loss is 2.6476094722747803
epoch: 1 step: 300, loss is 1.784448266029358
epoch: 1 step: 400, loss is 1.402374029159546
epoch: 1 step: 500, loss is 1.355136752128601
epoch: 1 step: 600, loss is 1.1950846910476685
...
```
