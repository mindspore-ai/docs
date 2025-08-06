# 优化器并行

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/tutorials/source_zh_cn/parallel/optimizer_parallel.md)

## 简介

在进行数据并行训练时，模型的参数更新部分在各卡间存在冗余计算，优化器并行通过将优化器的计算量分散到数据并行维度的卡上，在大规模网络上（比如Bert、GPT）可以有效减少内存消耗并提升网络性能。

下面以Ascend单机8卡为例，进行优化器并行操作说明：

## 样例代码说明

> 下载完整的样例代码：[distributed_optimizer_parallel](https://gitee.com/mindspore/docs/tree/br_base/docs/sample_code/distributed_optimizer_parallel)。

目录结构如下：

```text
└─ sample_code
    ├─ distributed_optimizer_parallel
       ├── distributed_optimizer_parallel.py
       └── run.sh
    ...
```

其中，`distributed_optimizer_parallel.py`是定义网络结构和训练过程的脚本。`run.sh`是执行脚本。

## 配置分布式环境

通过context接口指定运行模式、运行设备、运行卡号等，与单卡脚本不同，并行脚本还需init初始化HCCL或NCCL通信。

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
init()
ms.set_seed(1)
```

## 数据集加载

在优化器并行场景下，数据集加载方式与单卡加载方式一致，代码如下：

```python
import os
import mindspore.dataset as ds

def create_dataset(batch_size):
    """create dataset"""
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
```

## 定义网络和优化器

优化器并行网络结构与单卡网络结构基本一致，区别在于增加了通信算子融合的配置，以及需要对网络和优化器进行延后初始化：

```python
from mindspore import nn
from mindspore.nn.utils import no_init_parameters

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Dense(28*28, 512)
        self.layer2 = nn.Dense(512, 512)
        self.layer3 = nn.Dense(512, 10)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        logits = self.layer3(x)
        return logits

with no_init_parameters:
    net = Network()
    optimizer = nn.SGD(net.trainable_params(), 1e-2)
net.layer1.set_comm_fusion(0)
net.layer2.set_comm_fusion(1)
net.layer3.set_comm_fusion(2)
```

> 这里为了减少通信成本，为不同层配置了通信融合，详细可以参考[通信算子融合](https://www.mindspore.cn/tutorials/zh-CN/br_base/parallel/comm_fusion.html)。

## 训练网络定义

在这一步，我们需要定义损失函数以及训练步骤，这部分与单卡写法一致：

```python
import mindspore as ms
from mindspore import nn

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()

def forward_fn(data, target):
    logits = net(data)
    loss = loss_fn(logits, target)
    return loss, logits

grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)

@ms.jit
def train_step(inputs, targets):
    (loss_value, _), grads = grad_fn(inputs, targets)
    optimizer(grads)
    return loss_value

```

## 并行配置

我们需要进一步设置并行有关的配置，指定并行模式`semi_auto`为半自动并行模式，此外，还需开启优化器并行，配置`hsdp`。

```python
from mindspore.parallel.auto_parallel import AutoParallel

parallel_net = AutoParallel(train_step, parallel_mode="semi_auto")
parallel_net.hsdp()

```

## 训练循环

这一步进行训练循环，外层循环是训练的epoch数，内层循环遍历数据集，调用parallel_net进行训练并获得损失值。

```python
for epoch in range(10):
    i = 0
    for image, label in data_set:
        loss_output = parallel_net(image, label)
        if i % 10 == 0:
            print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss_output))
        i += 1
```

## 运行单机8卡脚本

接下来通过命令调用对应的脚本，以`msrun`启动方式，8卡的分布式训练脚本为例，进行分布式训练：

```bash
bash run.sh
```

训练完后，日志文件保存到`log_output`目录下，其中部分文件目录结构如下：

```text
└─ log_output
    ├─ scheduler.log
    ├─ worker_0.log
    ├─ worker_1.log
...
```

结果保存在`log_output/worker_*.py`中，示例如下：

```text
epoch: 0, step: 0, loss is 2.3024087
epoch: 0, step: 10, loss is 2.2921634
epoch: 0, step: 20, loss is 2.278274
epoch: 0, step: 30, loss is 2.2537143
epoch: 0, step: 40, loss is 2.1638
epoch: 0, step: 50, loss is 1.984318
epoch: 0, step: 60, loss is 1.6061916
epoch: 0, step: 70, loss is 1.20966
epoch: 0, step: 80, loss is 0.98156196
epoch: 0, step: 90, loss is 0.77229893
epoch: 0, step: 100, loss is 0.6854114
...
```

其他启动方式如`mpirun`、`rank table`的启动可参考[启动方式](https://www.mindspore.cn/tutorials/zh-CN/br_base/parallel/startup_method.html)。

