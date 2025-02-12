# UCE故障快速恢复

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/model_train/train_availability/UCE_fault_recover.md)

## 概述

模型并行训练过程中，可能会遇到UCE(Uncorrectable Error)故障导致训练中断。重新启动训练，各种资源的开销是巨大的。为此MindSpore提供了故障恢复的方案。使得在发生故障时，模型在故障发生处快速恢复并继续训练，无需重启训练。

### 场景限制

1. 目前仅支持图模式。
2. 不支持网络中使用对训练结果产生影响的全局状态变量。

## 用例

下面以一个4卡数据并行网络训练为例，介绍如何配置UCE故障快速恢复。 配置完成后，在训练中如遇到UCE故障，MindSpore和MindIO会停止所有卡的训练， 对故障卡进行清洗和修复， 从故障卡的备份卡拷贝参数到故障卡并继续训练。如果故障发生在第n个step， 那继续训练将从第n+1个step开始。

### 环境准备

开启UCE快速恢复功能需要先安装`MindIO`, 详情参见[MindIO](https://www.hiascend.com/document/detail/zh/mindx-dl/60rc2/mindio/mindiottp/mindiottp001.html)。

### 准备数据

下载MNIST数据集，并解压数据集到项目目录。

```bash
wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
unzip MNIST_Data.zip
```

### 模型定义

开启UCE快速恢复功能需要设置TFT优化器, 在优化器更新前向MindIO TFT上报状态。用`OptTFTWrapper`来配置, 详情参见[OptTFTWrapper](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.OptTFTWrapper.html)。

```python

import os
import math
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, ops, Parameter, train
from mindspore.communication import init
from mindspore.common.initializer import initializer, HeUniform


ms.set_context(mode=ms.GRAPH_MODE,
                jit_level='O1')
ms.set_device(device_target="Ascend")

ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
init()

class MatMulCell(nn.Cell):
    """
    MatMulCell definition.
    """
    def __init__(self, param=None, shape=None):
        super().__init__()
        if shape is None:
            shape = [28 * 28, 512]
        weight_init = HeUniform(math.sqrt(5))
        self.param = Parameter(initializer(weight_init, shape), name="param")
        if param is not None:
            self.param = param
        self.print = ops.Print()
        self.matmul = ops.MatMul()

    def construct(self, x):
        out = self.matmul(x, self.param)
        self.print("out is:", out)
        return out


class Network(nn.Cell):
    """
    Network definition.
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = MatMulCell()
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Dense(512, 512)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Dense(512, 10)

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        logits = self.layer3(x)
        return logits

net = Network()


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

dataset = create_dataset(32)

optimizer = nn.SGD(net.trainable_params(), 1e-2)
#配置TFT优化器
optimizer_wrapper = nn.OptTFTWrapper(optimizer)
loss_fn = nn.CrossEntropyLoss()

model = ms.Model(net, loss_fn=loss_fn, optimizer=optimizer_wrapper)
```

### Callback

开启UCE快速恢复功能需要设置 `TrainFaultTolerance` Callback对象，并传入参数来配置，详情参见[TrainFaultTolerance](https://www.mindspore.cn/docs/zh-CN/master/api_python/train/mindspore.train.TrainFaultTolerance.html)。

```python
time_monitor = train.TimeMonitor(data_size=1)
loss_cb = train.LossMonitor(1)

# 设置callback对象
tft_cb = train.TrainFaultTolerance()

model.train(5, dataset, callbacks=[time_monitor, loss_cb, tft_cb])

```

### 配置环境变量并启动训练

开启UCE故障快速恢复功能，需要设置环境变量 `MS_ENABLE_TFT='{UCE:1, TTP:1}'`。 其中 `UCE:1` 表示开启UCE快速恢复功能，`TTP:1` 表示开启临终遗言功能。 开启UCE会默认开启临终遗言功能， 如果想仅开启临终功能，可以设置环境变量  `MS_ENABLE_TFT='{UCE:0, TTP:1}'` 。此外还需要设置环境变量 `MINDIO_FOR_MINDSPORE=1`， 使能 `MindIO` 适配 MindSpore。

使用 `msrun` 命令启动训练。

```bash
export MS_ENABLE_TFT='{UCE:1 TTP:1}'
export MINDIO_FOR_MINDSPORE=1
export DATA_PATH=${EXEC_PATH}/MNIST_DATA/train/
export MS_TFT_IP = "127.0.0.1"
export MS_TFT_PORT = 30051

# UCE_case.py 按照上述代码创建
msrun --worker_num=4 --local_worker_num=4 --master_port=10970 --join=False --log_dir=./uce_logs UCE_case.py
```
