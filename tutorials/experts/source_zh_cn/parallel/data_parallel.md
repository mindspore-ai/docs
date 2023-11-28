# 数据并行

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.2/tutorials/experts/source_zh_cn/parallel/data_parallel.md)

## 概述

数据并行是最常用的并行训练方式，用于加速模型训练和处理大规模数据集。在数据并行模式下，训练数据被划分成多份，然后将每份数据分配到不同的计算节点上，例如多卡或者多台设备。每个节点独立地处理自己的数据子集，并使用相同的模型进行前向传播和反向传播，最终对所有节点的梯度进行同步后，进行模型参数更新。

> 数据并行支持的硬件平台包括Ascend、GPU和CPU，此外还同时支持PyNative模式和Graph模式。

相关接口：

1. `mindspore.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL)`：设置数据并行模式。
2. `mindspore.nn.DistributedGradReducer()`：进行多卡梯度聚合。

## 整体流程

![整体流程](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/docs/mindspore/source_zh_cn/design/images/data_parallel.png)

1. 环境依赖

    每次开始进行并行训练前，通过调用`mindspore.communication.init`接口初始化通信资源，并自动创建全局通信组`WORLD_COMM_GROUP`。通信组能让通信算子在卡间和机器间进行信息收发，全局通信组是最大的一个通信组，包括了当前训练的所有设备。通过调用`mindspore.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL)`设置当前模式为数据并行模式。

2. 数据分发（Data distribution）

    数据并行的核心在于将数据集在样本维度拆分并下发到不同的卡上。在`mindspore.dataset`模块提供的所有数据集加载接口中都有`num_shards`和`shard_id`两个参数，它们用于将数据集拆分为多份并循环采样的方式，采集`batch`大小的数据到各自的卡上，当出现数据量不足的情况时将会从头开始采样。

3. 网络构图

    数据并行网络的书写方式与单卡网络没有差别，这是因为在正反向传播（Forward propagation & Backward propagation）过程中各卡的模型间是独立执行的，只是保持了相同的网络结构。唯一需要特别注意的是为了保证各卡间训练同步，相应的网络参数初始化值应当是一致的，在`DATA_PARALLEL`模式下可以通过设置seed或通过使能`parameter_broadcast`达到多卡间权重初始化一致的目的。

4. 梯度聚合（Gradient aggregation）

    数据并行理论上应该实现和单卡一致的训练效果，为了保证计算逻辑的一致性，通过调用`mindspore.nn.DistributedGradReducer()`接口，在梯度计算完成后自动插入`AllReduce`算子实现各卡间的梯度聚合操作。MindSpore设置了`mean`开关，用户可以选择是否要对求和后的梯度值进行求平均操作，也可以将其视为超参项。

5. 参数更新（Parameter update）

    因为引入了梯度聚合操作，所以各卡的模型会以相同的梯度值一起进入参数更新步骤。

## 操作实践

下面以Ascend或者GPU单机8卡为例，进行数据并行操作说明：

### 样例代码说明

> 您可以在这里下载完整的样例代码：
>
> <https://gitee.com/mindspore/docs/tree/r2.2/docs/sample_code/distributed_data_parallel>。

目录结构如下：

```text
└─ sample_code
    ├─ distributed_data_parallel
       ├── distributed_data_parallel.py
       └── run.sh
    ...
```

其中，`distributed_data_parallel.py`是定义网络结构和训练过程的脚本。`run.sh`是执行脚本。

### 配置分布式环境

通过context接口可以指定运行模式、运行设备、运行卡号等，与单卡脚本不同，并行脚本还需指定并行模式`parallel_mode`为数据并行模式，并通过init初始化HCCL或NCCL通信。在数据并行模式还需要设置`gradients_mean`指定梯度聚合方式。此处不设置`device_target`会自动指定为MindSpore包对应的后端硬件设备。

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
init()
ms.set_seed(1)
```

其中，`gradients_mean=True`是为了在反向计算时，框架内部会将数据并行参数分散在多台机器的梯度值进行聚合，得到全局梯度值后再传入优化器中更新。默认值为`False`，设置为`True`对应聚合方式为`AllReduce.Mean`操作，`False`对应`AllReduce.Sum`操作。

### 数据并行模式加载数据集

数据并行模式跟其他模式最大区别在于数据加载方式的不同，数据是以并行的方式导入的。下面我们以MNIST数据集为例，介绍以数据并行方式导入MNIST数据集的方法，`dataset_path`是指数据集的路径。

```python
import mindspore.dataset as ds
from mindspore.communication import get_rank, get_group_size

rank_id = get_rank()
rank_size = get_group_size()
dataset = ds.MnistDataset(dataset_path, num_shards=rank_size, shard_id=rank_id)
```

其中，与单卡不同的是，在数据集接口需要传入`num_shards`和`shard_id`参数，分别对应卡的数量和逻辑序号，建议通过`mindspore.communication`接口获取：

- `get_rank`：获取当前设备在集群中的ID。
- `get_group_size`：获取集群数量。

> 数据并行场景加载数据集时，建议对每卡指定相同的数据集文件，若是各卡加载的数据集不同，可能会影响计算精度。

完整的数据处理代码：

```python
import os
import mindspore.dataset as ds
from mindspore.communication import get_rank, get_group_size

def create_dataset(batch_size):
    dataset_path = os.getenv("DATA_PATH")
    rank_id = get_rank()
    rank_size = get_group_size()
    dataset = ds.MnistDataset(dataset_path, num_shards=rank_size, shard_id=rank_id)
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

### 定义网络

数据并行模式下，网络定义方式与单卡网络写法一致，网络的主要结构如下：

```python
from mindspore import nn

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

net = Network()
```

### 训练网络

在这一步，我们需要定义损失函数、优化器以及训练过程。与单卡模型不同的地方在于，数据并行模式还需要增加`mindspore.nn.DistributedGradReducer()`接口，来对所有卡的梯度进行聚合，该接口第一个参数为需要更新的网络参数：

```python
from mindspore import nn, ops

loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(net.trainable_params(), 1e-2)

def forward_fn(data, label):
    logits = net(data)
    loss = loss_fn(logits, label)
    return loss, logits

grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)
grad_reducer = nn.DistributedGradReducer(optimizer.parameters)

for epoch in range(10):
    i = 0
    for data, label in data_set:
        (loss, _), grads = grad_fn(data, label)
        grads = grad_reducer(grads)
        optimizer(grads)
        if i % 10 == 0:
            print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss))
        i += 1
```

> 此处也可以用[Model.train](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/train/mindspore.train.Model.html#mindspore.train.Model.train)的方式进行训练。

### 运行单机8卡脚本

接下来通过命令调用对应的脚本，以`mpirun`启动方式，8卡的分布式训练脚本为例，进行分布式训练：

```bash
bash run.sh
```

训练完后，日志文件保存到`log_output`目录下，其中部分文件目录结构如下：

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
epoch: 0 step: 0, loss is 2.3084016
epoch: 0 step: 10, loss is 2.3107638
epoch: 0 step: 20, loss is 2.2864391
epoch: 0 step: 30, loss is 2.2938071
...
```

其他启动方式如动态组网、`rank table`的启动可参考[启动方式](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.2/parallel/startup_method.html)。
