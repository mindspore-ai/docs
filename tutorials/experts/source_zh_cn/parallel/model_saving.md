# 模型保存

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/parallel/model_saving.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

本篇教程我们主要讲解，如何利用MindSpore进行分布式网络训练并保存模型文件。在分布式训练场景下，模型保存可以分为合并保存和非合并保存：合并保存需要额外的通信和内存开销，每张卡保存相同的模型文件；非合并保存则只保存当前卡切分后的权重，有效减少了聚合需要的通信和内存开销。

相关接口：

1. `mindspore.set_auto_parallel_context(strategy_ckpt_config=strategy_ckpt_dict)`：用于设置并行策略文件的配置。`strategy_ckpt_dict`是用于设置并行策略文件的配置，是字典类型。strategy_ckpt_dict = {"load_file": "./stra0.ckpt", "save_file": "./stra1.ckpt", "only_trainable_params": False}，其中：
    - `load_file(str)`：加载并行切分策略的路径。默认值：""。
    - `save_file(str)`：保存并行切分策略的路径，分布式训练场景中该参数必须设置。默认值：""。
    - `only_trainable_params(bool)`：仅保存/加载可训练参数的策略信息。默认值：`True`。

2. `mindspore.train.ModelCheckpoint(prefix='CKP', directory=None, config=None)`：在训练过程中调用该接口保存网络参数。该接口中可以通过配置`config`来配置具体的策略，参见接口`mindspore.train.CheckpointConfig`，需要注意的是，并行模式下需要对每张卡上运行的脚本指定不同的checkpoint保存路径，防止读写文件时发生冲突。

3. `mindspore.train.CheckpointConfig(save_checkpoint_steps=10, integrated_save=True)`：配置保存Checkpoint的策略。`save_checkpoint_steps`表示每隔多少个step保存一次Checkpoint。`integrated_save`表示在自动并行场景下，是否合并保存拆分后的模型文件。合并保存功能仅支持在自动并行场景中使用，在手动并行场景中不支持。

## 操作实践

下面以单机8卡为例，进行分布式训练下保存模型文件的操作说明：

### 样例代码说明

> 下载完整的样例代码：[model_saving_loading](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/model_saving_loading)。

目录结构如下：

```text
└─ sample_code
    ├─ model_saving_loading
       ├── train_saving.py
       ├── run_saving.sh
       ...
    ...
```

其中，`train_saving.py`是定义网络结构和训练过程的脚本。`run_saving.sh`是执行脚本。

### 配置分布式环境

通过context接口指定运行模式、运行设备、运行卡号等，与单卡脚本不同，并行脚本还需指定并行模式`parallel_mode`为半自动并行模式，通过`strategy_ckpt_config`配置保存分布式策略文件，并通过init初始化HCCL或NCCL通信。`device_target`会自动指定为MindSpore包对应的后端硬件设备。

```python
import mindspore as ms
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
ms.set_auto_parallel_context(strategy_ckpt_config={"save_file": "./src_strategy.ckpt"})
init()
ms.set_seed(1)
```

### 网络定义

网络定义中加入了`ops.MatMul()`算子的切分策略：

```python
from mindspore import nn, ops
from mindspore.common.initializer import initializer

class Dense(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = ms.Parameter(initializer("normal", [in_channels, out_channels], ms.float32))
        self.bias = ms.Parameter(initializer("normal", [out_channels], ms.float32))
        self.matmul = ops.MatMul()
        self.add = ops.Add()

    def construct(self, x):
        x = self.matmul(x, self.weight)
        x = self.add(x, self.bias)
        return x

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = ops.Flatten()
        self.layer1 = Dense(28*28, 512)
        self.relu1 = ops.ReLU()
        self.layer2 = Dense(512, 512)
        self.relu2 = ops.ReLU()
        self.layer3 = Dense(512, 10)

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        logits = self.layer3(x)
        return logits

net = Network()
net.layer1.matmul.shard(((2, 1), (1, 2)))
net.layer3.matmul.shard(((2, 2), (2, 1)))
```

### 数据集加载

数据集加载方式与单卡模型一致，代码如下：

```python
import os
import mindspore.dataset as ds

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
```

### 训练网络

对于网络中切分的参数框架默认会自动聚合保存到模型文件，但考虑到在超大模型场景下，单个完整的模型文件过大会带来传输慢、难加载等问题，所以用户可以通过`CheckpointConfig`中`integrated_save`参数选择非合并保存，即每张卡保存各自卡上的参数切片。

```python
import mindspore as ms
from mindspore.communication import get_rank
from mindspore import nn, train

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()
loss_cb = train.LossMonitor(20)
ckpt_config = train.CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=1, integrated_save=False)
ckpoint_cb = train.ModelCheckpoint(prefix="checkpoint",
                                   directory="./src_checkpoints/rank_{}".format(get_rank()),
                                   config=ckpt_config)
model = ms.Model(net, loss_fn=loss_fn, optimizer=optimizer)
model.train(10, data_set, callbacks=[loss_cb, ckpoint_cb])
```

### 运行单机八卡脚本

接下来通过命令调用对应的脚本，以`mpirun`启动方式，8卡的分布式训练脚本为例，进行分布式训练：

```bash
bash run_saving.sh
```

训练完后，日志文件保存到`log_output`目录下，Checkpoint文件保存在`src_checkpoints`文件夹下，文件目录结构如下：

```text
├─ src_strategy.ckpt
├─ log_output
|   └─ 1
|       ├─ rank.0
|       |   └─ stdout
|       ├─ rank.1
|       |   └─ stdout
|       ...
├─ src_checkpoints
|   ├─ rank_0
|   |   ├─ checkpoint-10_1875.ckpt
|   |   └─ checkpoint-graph.meta
|   ├─ rank_1
|   |   ├─ checkpoint-10_1875.ckpt
|   |   ...
|   ...
...
```

关于Loss部分结果保存在`log_output/1/rank.*/stdout`中，示例如下：

```text
epoch: 1 step: 20, loss is 2.2978780269622803
epoch: 1 step: 40, loss is 2.2965049743652344
epoch: 1 step: 60, loss is 2.2927846908569336
epoch: 1 step: 80, loss is 2.294496774673462
epoch: 1 step: 100, loss is 2.2829630374908447
epoch: 1 step: 120, loss is 2.2793829441070557
epoch: 1 step: 140, loss is 2.2842094898223877
epoch: 1 step: 160, loss is 2.269033670425415
epoch: 1 step: 180, loss is 2.267289400100708
epoch: 1 step: 200, loss is 2.257275342941284
...
```

通过配置`mindspore.train.CheckpointConfig`中的`integrated_save`为`True`，可以开启合并保存，需要替换的代码如下：

```python
...
ckpt_config = train.CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=3, integrated_save=True)
ckpoint_cb = train.ModelCheckpoint(prefix="checkpoint",
                                   directory="./src_checkpoints_integrated/rank_{}".format(get_rank()),
                                   config=ckpt_config)
...
```
