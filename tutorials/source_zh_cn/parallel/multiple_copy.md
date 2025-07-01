# 多副本并行

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/parallel/multiple_copy.md)

## 简介

大模型训练中，模型并行引入的通信是一个显著的性能瓶颈，从网络算法的角度来看，由于该部分通信对前序数据有依赖，无法与计算掩盖。为了解决这个问题，提出了多副本并行方案。

使用场景：当在半自动模式以及网络中存在模型并行时，第1份的切片数据的前向计算同时，第2份的数据将会进行模型并行的通信，以此来达到通信计算并发的性能加速。

### 原理

多副本并行指在数据并行的基础上，进一步把每张卡上的输入数据沿Batch维度切分为多份，各部分之间的计算与通信相互独立。其底层在通信的时候，另一副本进行计算操作，无需等待，这样就能保证多副本的计算和通信的时间相互互补，提升模型性能，同时将数据拆成多副本的形式还能减少算子输入的参数量，从而减少单个算子的计算时间，对提升模型性能有很大帮助。

![多副本并行](./images/multi_copy.png "多副本并行")

### 相关接口

- [mindspore.parallel.nn.MicroBatchInterleaved(cell_network, interleave_num=2)](https://www.mindspore.cn/docs/zh-CN/master/api_python/parallel/mindspore.parallel.nn.MicroBatchInterleaved.html)：这个函数的作用是将输入在第零维度拆成 `interleave_num`份，然后执行包裹的cell的计算。

## 操作实践

下面以Ascend单机8卡为例，进行多副本并行操作说明：

### 样例代码说明

> 下载完整的样例代码：[multiple_copy](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/multiple_copy)。

目录结构如下：

```text
└─ sample_code
    ├─ multiple_copy
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

此处数据集加载和网络定义与单卡模型一致。
通过 [no_init_parameters](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.utils.no_init_parameters.html) 接口延后初始化网络参数和优化器参数。

```python
import os
import mindspore.dataset as ds
from mindspore import nn
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

在这一步，我们需要定义损失函数、训练过程，调用两个接口来配置多副本并行：

- 首先需要定义LossCell，本例中调用了[nn.WithLossCell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.WithLossCell.html)接口封装网络和损失函数。
- 然后需要在LossCell外包一层[mindspore.parallel.nn.MicroBatchInterleaved](https://www.mindspore.cn/docs/zh-CN/master/api_python/parallel/mindspore.parallel.nn.MicroBatchInterleaved.html)，并指定interleave_num的size为2。详细请参考本章概述中的相关接口。

最后，`AutoParallel` 包裹 `net` 设置并行模式为半自动并行模式。

```python
import mindspore as ms
from mindspore import nn, train

loss_fn = nn.CrossEntropyLoss()
loss_cb = train.LossMonitor(100)
net = ms.parallel.nn.MicroBatchInterleaved(nn.WithLossCell(net, loss_fn), 2)
net = AutoParallel(net, parallel_mode="semi_auto")
model = ms.Model(net, optimizer=optimizer)
model.train(10, data_set, callbacks=[loss_cb])
```

> 多副本并行训练更适合用`model.train`的方式，这是因为多副本并行下的TrainOneStep逻辑复杂，而`model.train`内部封装了针对多副本并行的TrainOneStepCell，易用性更好。

### 运行单机8卡脚本

接下来通过命令调用对应的脚本，以`msrun`启动方式，8卡的分布式训练脚本为例，进行分布式训练：

```bash
bash run.sh
```

训练完后，关于Loss部分结果保存在`log_output/worker_*.log`中，示例如下：

```text
epoch: 1 step: 100, loss is 4.514171123504639
epoch: 1 step: 200, loss is 3.835113048553467
epoch: 1 step: 300, loss is 1.9824411869049072
epoch: 1 step: 400, loss is 1.2429465055465698
epoch: 1 step: 500, loss is 1.0608973503112793
epoch: 1 step: 600, loss is 0.9407652616500854
epoch: 1 step: 700, loss is 0.8292769193649292
...
```
