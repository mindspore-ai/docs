# PyNative应用

`动态图` `静态图` `PyNative` `动静统一`

<!-- TOC -->

- [PyNative应用](#pynative应用)
    - [概述](#概述)
    - [设置模式](#设置模式)
    - [执行单算子](#执行单算子)
    - [执行函数](#执行函数)
    - [执行网络](#执行网络)
    - [构建网络](#构建网络)
    - [Loss函数及优化器](#loss函数及优化器)
    - [模型参数保存](#模型参数保存)
    - [训练网络](#训练网络)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/pynative.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

本文主要介绍PyNative模式下的应用示例。

## 设置模式

```python
context.set_context(mode=context.PYNATIVE_MODE)
```

## 执行单算子

```python
import numpy as np
import mindspore.ops as ops
from mindspore import context, Tensor

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

x = Tensor(np.ones([1, 3, 5, 5]).astype(np.float32))
y = Tensor(np.ones([1, 3, 5, 5]).astype(np.float32))
z = ops.add(x, y)
print(z.asnumpy())
```

## 执行函数

```python
import numpy as np
from mindspore import context, Tensor
import mindspore.ops as ops

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

def add_func(x, y):
    z = ops.add(x, y)
    z = ops.add(z, x)
    return z

x = Tensor(np.ones([3, 3], dtype=np.float32))
y = Tensor(np.ones([3, 3], dtype=np.float32))
output = add_func(x, y)
print(output.asnumpy())
```

## 执行网络

在construct中定义网络结构，在具体运行时，下例中，执行net(x, y)时，会从construct函数中开始执行。

```python
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context, Tensor

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = ops.Mul()

    def construct(self, x, y):
        return self.mul(x, y)

x = Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
y = Tensor(np.array([4.0, 5.0, 6.0]).astype(np.float32))

net = Net()
print(net(x, y))
```

## 构建网络

可以在网络初始化时，明确定义网络所需要的各个部分，在construct中定义网络结构。

```python
import mindspore.nn as nn
from mindspore.common.initializer import Normal

class LeNet5(nn.Cell):
    def __init__(self, num_class=10, num_channel=1, include_top=True):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.include_top = include_top
        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
            self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
            self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))


    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if not self.include_top:
            return x
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

```

## Loss函数及优化器

在PyNative模式下，通过针对每个参数对应的梯度进行参数更新。

```python
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
net_opt = nn.Momentum(network.trainable_params(), config.lr, config.momentum)
```

## 模型参数保存

保存模型可以通过定义CheckpointConfig来指定模型保存的参数。

save_checkpoint_steps：每多少个step保存一下参数；keep_checkpoint_max：最多保存多少份模型参数。详细使用方式请参考[保存模型](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/save_model.html)。

```python
config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", directory=config.ckpt_path, config=config_ck)
```

## 训练网络

```python
context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target)
ds_train = create_dataset(os.path.join(config.data_path, "train"), config.batch_size)
network = LeNet5(config.num_classes)
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
net_opt = nn.Momentum(network.trainable_params(), config.lr, config.momentum)
time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                keep_checkpoint_max=config.keep_checkpoint_max)
ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", directory=config.ckpt_path, config=config_ck)

model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()}, amp_level="O2")
```

完整的运行代码可以到ModelZoo下载[lenet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/lenet)，并设置context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target)。
