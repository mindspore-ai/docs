# 数据并行

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/parallel/data_parallel.md)

## 简介

数据并行是最常用的并行训练方式，用于加速模型训练和处理大规模数据集。在数据并行模式下，训练数据被划分成多份，然后将每份数据分配到不同的计算节点上，例如多卡或者多台设备。每个节点独立地处理自己的数据子集，并使用相同的模型进行前向传播和反向传播，最终对所有节点的梯度进行同步后，进行模型参数更新。

下面以Ascend单机8卡为例，进行数据并行操作说明：

## 样例代码说明

> 下载完整的样例代码：[distributed_data_parallel](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/distributed_data_parallel)。

目录结构如下：

```text
└─ sample_code
    ├─ distributed_data_parallel
       ├── distributed_data_parallel.py
       └── run.sh
    ...
```

其中，`distributed_data_parallel.py`是定义网络结构和训练过程的脚本。`run.sh`是执行脚本。

## 配置分布式环境

通过context接口可以指定运行模式、运行设备、运行卡号等。与单卡脚本不同，并行脚本还需指定并行模式`parallel_mode`为数据并行模式，并通过init根据不同的设备需求初始化HCCL、NCCL或者MCCL 通信。在数据并行模式还可以设置`gradients_mean`指定梯度聚合方式。此处未设置`device_target`，会自动指定为MindSpore包对应的后端硬件设备（默认为Ascend）。

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
init()
ms.set_seed(1)
```

其中，`gradients_mean=True`是为了在反向计算时，框架内部会将数据并行参数分散在多台机器的梯度值进行聚合，得到全局梯度值后再传入优化器中更新。首先通过AllReduce(op=ReduceOp.SUM)对梯度做规约求和，接着根据gradients_mean的值来判断是否求均值（设置为True则求均值，否则不求，默认为False）。

## 数据集加载

数据并行模式跟其他模式最大区别在于数据加载方式的不同，数据是以并行的方式导入的。下面我们以MNIST数据集为例，介绍以数据并行方式导入MNIST数据集的方法，`dataset_path`是指数据集的路径。

```python
import mindspore.dataset as ds
from mindspore.communication import get_rank, get_group_size

rank_id = get_rank()
rank_size = get_group_size()
dataset = ds.MnistDataset(dataset_path, num_shards=rank_size, shard_id=rank_id)
```

其中，与单卡不同的是，在数据集接口需要传入`num_shards`和`shard_id`参数，分别对应卡的数量和逻辑序号，建议通过[mindspore.communication](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.communication.html)模块的以下接口获取：

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

## 定义网络

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

## 训练网络

在这一步，我们需要定义损失函数、优化器以及训练过程。与单卡模型不同的地方在于，数据并行模式还需要增加[mindspore.nn.DistributedGradReducer()](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.DistributedGradReducer.html)接口，来对所有卡的梯度进行聚合，该接口第一个参数为需要更新的网络参数：

```python
from mindspore import nn
import mindspore as ms

loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(net.trainable_params(), 1e-2)

def forward_fn(data, label):
    logits = net(data)
    loss = loss_fn(logits, label)
    return loss, logits

grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)
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

> 此处也可以用[Model.train](https://www.mindspore.cn/docs/zh-CN/master/api_python/train/mindspore.train.Model.html#mindspore.train.Model.train)的方式进行训练。

## 运行单机8卡脚本

接下来通过命令调用对应的脚本，以8卡的分布式训练脚本为例，使用`msrun`启动方式进行分布式训练：

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
epoch: 0 step: 0, loss is 2.3026438
epoch: 0 step: 50, loss is 2.2963896
epoch: 0 step: 100, loss is 2.2882829
epoch: 0 step: 150, loss is 2.2822685
...
```

其他启动方式如`mpirun`、`rank table`的启动可参考[启动方式](https://www.mindspore.cn/tutorials/zh-CN/master/parallel/startup_method.html)。