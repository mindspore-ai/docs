# 故障恢复

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/tutorials/source_zh_cn/train_availability/fault_recover.md)

## 概述

在模型训练过程中可能会遇到故障，重新启动训练会产生巨大的资源开销。为此，MindSpore提供了故障恢复方案，通过周期性保存模型参数，使模型能够在故障发生处快速恢复并继续训练。

MindSpore支持以step或epoch为周期保存模型参数。模型参数保存在Checkpoint（简称ckpt）文件中。当模型训练期间发生故障时，可以载入最新保存的模型参数，恢复到该状态继续训练。

> 本文档介绍故障恢复的用例，仅在每个epoch结束保存Checkpoint文件。

## 数据和模型准备

为了提供完整的体验，这里使用MNIST数据集和LeNet5网络模拟故障恢复的过程，如已准备好，可直接跳过本章节。

### 数据准备

下载MNIST数据集，并解压数据集到项目目录。

```python
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=True)
```

### 模型定义

```python
import os

import mindspore
from mindspore.common.initializer import Normal
from mindspore.dataset import MnistDataset, vision
from mindspore import nn
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, Callback
import mindspore.dataset.transforms as transforms

mindspore.set_context(mode=mindspore.GRAPH_MODE)


# 创建训练数据集
def create_dataset(data_path, batch_size=32):
    train_dataset = MnistDataset(data_path, shuffle=False)
    image_transfroms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Resize(size=(32, 32)),
        vision.HWC2CHW()
    ]
    train_dataset = train_dataset.map(image_transfroms, input_columns='image')
    train_dataset = train_dataset.map(transforms.TypeCast(mindspore.int32), input_columns='label')
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    return train_dataset


# 加载训练数据集
data_path = "MNIST_Data/train"
train_dataset = create_dataset(data_path)

# 模拟训练过程中发生故障
class myCallback(Callback):
    def __init__(self, break_epoch_num=6):
        super(myCallback, self).__init__()
        self.epoch_num = 0
        self.break_epoch_num = break_epoch_num

    def on_train_epoch_end(self, run_context):
        self.epoch_num += 1
        if self.epoch_num == self.break_epoch_num:
            raise Exception("Some errors happen.")


class LeNet5(nn.Cell):
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode="valid")
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode="valid")
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = LeNet5()  # 模型初始化
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")  # 损失函数
optim = nn.Momentum(net.trainable_params(), 0.01, 0.9)  # 优化器
model = Model(net, loss_fn=loss, optimizer=optim)  # Model封装
```

## 周期性保存Checkpoint文件

### 配置CheckpointConfig

`mindspore.train.CheckpointConfig` 支持根据迭代次数进行配置，主要参数如下：

- `save_checkpoint_steps`：表示每隔多少个step保存一个Checkpoint文件，默认值为1。
- `keep_checkpoint_max`：表示最多保存多少个Checkpoint文件，默认值为5。

在迭代过程正常结束时，会默认保存最后一个step的Checkpoint文件。

模型训练过程中，使用 `Model.train` 里面的 `callbacks` 参数传入保存模型的对象 `ModelCheckpoint` （与 `mindspore.train.CheckpointConfig` 配合使用），可以周期性地保存模型参数，生成Checkpoint文件。

### 用户自定义保存数据

`CheckpointConfig` 的参数 `append_info` 可以在Checkpoint文件中保存用户自定义信息。`append_info` 支持传入 ``epoch_num`` 、 ``step_num`` 和字典类型数据。``epoch_num`` 和 ``step_num`` 可以在Checkpoint文件中保存训练过程中的epoch数和step数。
字典类型数据的 `key` 必须是string类型，`value` 必须是int、float、bool、string、Parameter或Tensor类型。

```python
# 用户自定义保存的数据
append_info = ["epoch_num", "step_num", {"lr": 0.01, "momentum": 0.9}]
# 数据下沉模式下，默认保存最后一个step的Checkpoint文件
config_ck = CheckpointConfig(append_info=append_info)
# 保存的Checkpoint文件前缀是"lenet"，文件保存在"./lenet"路径下
ckpoint_cb = ModelCheckpoint(prefix='lenet', directory='./lenet', config=config_ck)

# 模拟程序故障，默认是在第6个epoch结束后故障
my_callback = myCallback()

# 数据下沉模式下，使用Model.train进行10个epoch的训练
model.train(10, train_dataset, callbacks=[ckpoint_cb, my_callback], dataset_sink_mode=True)
```

## 自定义脚本找到最新的Checkpoint文件

程序在第6个epoch结束后发生故障。故障发生后，`./lenet` 目录下保存了最新生成的5个epoch的Checkpoint文件。

```text
└── lenet
     ├── lenet-graph.meta  # 编译后的计算图
     ├── lenet-2_1875.ckpt  # Checkpoint文件后缀名为'.ckpt'
     ├── lenet-3_1875.ckpt  # 文件的命名方式表示保存参数所在的epoch和step数，这里为第3个epoch的第1875个step的模型参数
     ├── lenet-4_1875.ckpt
     ├── lenet-5_1875.ckpt
     └── lenet-6_1875.ckpt
```

> 如果用户使用相同的前缀名，运行多次训练脚本，可能会生成同名Checkpoint文件。MindSpore为方便用户区分每次生成的文件，会在用户定义的前缀后添加”_”和数字加以区分。如果想要删除.ckpt文件时，请同步删除.meta 文件。例如：`lenet_3-2_1875.ckpt` 表示运行第4次脚本生成的第2个epoch的第1875个step的Checkpoint文件。

用户可以使用自定义脚本找到最新保存的Checkpoint文件。

```python
ckpt_path = "./lenet"
filenames = os.listdir(ckpt_path)
# 筛选所有的Checkpoint文件名
ckptnames = [ckpt for ckpt in filenames if ckpt.endswith(".ckpt")]
# 按照创建顺序从旧到新对Checkpoint文件名进行排序
ckptnames.sort(key=lambda ckpt: os.path.getctime(ckpt_path + "/" + ckpt))
# 获取最新的Checkpoint文件路径
ckpt_file = ckpt_path + "/" + ckptnames[-1]
```

## 恢复训练

### 加载Checkpoint文件

使用 `load_checkpoint` 和 `load_param_into_net` 方法加载最新保存的Checkpoint文件。

- `load_checkpoint` 方法会把Checkpoint文件中的网络参数加载到字典param_dict中。
- `load_param_into_net` 方法会把字典param_dict中的参数加载到网络或者优化器中，加载后网络中的参数就是Checkpoint文件中保存的。

```python
# 将模型参数加载到param_dict中，这里加载的是训练过程中保存的模型参数和用户自定义保存的数据
param_dict = mindspore.load_checkpoint(ckpt_file)
net = LeNet5()
# 将参数加载模型中
mindspore.load_param_into_net(net, param_dict)
```

### 获取用户自定义数据

用户可以从Checkpoint文件中获取训练时的epoch数和自定义保存的数据。注意，此时获取的数据是Parameter类型。

```python
epoch_num = int(param_dict["epoch_num"].asnumpy())
step_num = int(param_dict["step_num"].asnumpy())
lr = float(param_dict["lr"].asnumpy())
momentum = float(param_dict["momentum"].asnumpy())
```

### 设置继续训练的epoch

向 `Model.train` 的 `initial_epoch` 参数传入获取的epoch数，网络即可从该epoch继续训练。此时，`Model.train` 的 `epoch` 参数表示训练的最后一个epoch数。

```python
model.train(10, train_dataset, callbacks=ckpoint_cb, initial_epoch=epoch_num, dataset_sink_mode=True)
```

### 训练结束

训练结束， `./lenet` 目录下新生成4个Checkpoint文件。根据Checkpoint文件名可以看出，在故障发生后，模型重新在第7个epoch进行训练，并在第10个epoch结束。故障恢复成功。

```text
└── lenet
     ├── lenet-graph.meta
     ├── lenet-2_1875.ckpt
     ├── lenet-3_1875.ckpt
     ├── lenet-4_1875.ckpt
     ├── lenet-5_1875.ckpt
     ├── lenet-6_1875.ckpt
     ├── lenet-1-7_1875.ckpt
     ├── lenet-1-8_1875.ckpt
     ├── lenet-1-9_1875.ckpt
     ├── lenet-1-10_1875.ckpt
     └── lenet-1-graph.meta
```
