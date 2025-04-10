# 基于双递归搜索的多维混合并行案例

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/parallel/multiple_mixed.md)

## 概述

基于双递归搜索的多维混合并行是指用户可以配置重计算、优化器并行、流水线并行等优化方法，在用户配置的基础上，通过双递归策略搜索算法进行算子级策略自动搜索，进而生成最优的并行策略。

## 操作实践

下面以Ascend单机8卡为例，进行基于双递归搜索的多维混合并行案例说明：

### 样例代码说明

> 下载完整的样例代码：[multiple_mix](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/multiple_mix)。

目录结构如下：

```text
└─ sample_code
    ├─ multiple_mix
       ├── sapp_mix_train.py
       └── run_sapp_mix_train.sh
    ...
```

其中，`sapp_mix_train.py`是定义网络结构和训练过程的脚本。`run_sapp_mix_train.sh`是执行脚本。

### 配置分布式环境

通过init初始化HCCL或NCCL通信。`device_target`会自动指定为MindSpore包对应的后端硬件设备。

```python
import os
import mindspore as ms
from mindspore.communication import init

os.environ['MS_DEV_SAVE_GRAPHS'] = '2'
ms.set_context(mode=ms.GRAPH_MODE)
ms.runtime.set_memory(max_size="25GB")
init()
ms.set_seed(1)
```

### 网络定义

网络定义在双递归策略搜索算法提供的数据并行和模型并行基础上，加入重计算、流水线并行：

```python
from mindspore import nn

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Dense(28*28, 512)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Dense(512, 512)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Dense(512, 1)

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        logits = self.layer3(x)
        return logits

with no_init_parameters():
    net = Network()
    optimizer = nn.SGD(net.trainable_params(), 1e-2)
# 配置relu算子的重计算
net.relu1.recompute()
net.relu2.recompute()
```

### 数据集加载

数据集加载方式与单卡模型一致，代码如下：

```python
import os
import mindspore.dataset as ds
from mindspore.parallel.auto_parallel import AutoParallel
from mindspore import nn, train
from mindspore.communication import init

def create_dataset(batch_size):
    """create dataset"""
    dataset_path = "./MNIST_Data/train"
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

这部分与流水线并行的训练代码一致。在单机训练代码基础上需要调用两个额外的接口：`nn.WithLossCell`用于封装网络和损失函数、`ms.parallel.nn.Pipeline`用于封装LossCell和配置MicroBatch大小。通过`Autoparallel`接口指定运行模式、运行设备、运行卡号等，与单卡脚本不同，并行脚本还需指定并行模式`parallel_mode`为双递归策略搜索模式`recursive_programming`，用于自动切分数据并行和模型并行，`stages`为流水线并行中stage的数量，`hsdp`用于开启优化器并行。代码如下：

```python
import mindspore as ms
from mindspore import nn, train

loss_fn = nn.MAELoss()
loss_cb = train.LossMonitor()
# 配置每一层在流水线并行中的pipeline_stage编号
net_with_grads = ms.parallel.nn.Pipeline(nn.WithLossCell(net, loss_fn), 4,
                                            stage_config={"_backbone.layer1": 0,
                                                        "_backbone.relu1": 0,
                                                        "_backbone.layer2": 1,
                                                        "_backbone.relu2": 1,
                                                        "_backbone.layer3": 1,})
net_with_grads_new = AutoParallel(net_with_grads, parallel_mode="recursive_programming")
net_with_grads_new.hsdp()
net_with_grads_new.full_batch = True
net_with_grads_new.pipeline(stages=2, scheduler="1f1b")
model = ms.Model(net_with_grads, optimizer=optimizer)
model.train(10, data_set, callbacks=[loss_cb], dataset_sink_mode=True)
```

### 运行单机8卡脚本

接下来通过命令调用对应的脚本，以`msrun`启动方式，8卡的分布式训练脚本为例，进行分布式训练：

```bash
bash run_sapp_mix_train.sh
```

结果保存在`log_output/1/rank.*/stdout`中，示例如下：

```text
epoch: 1 step: 1875, loss is 11.6961808800697327
epoch: 2 step: 1875, loss is 10.2737872302532196
epoch: 3 step: 1875, loss is 8.87508840560913086
epoch: 4 step: 1875, loss is 8.1057268142700195
```