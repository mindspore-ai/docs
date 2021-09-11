# LossScale

<!-- TOC -->

- [LossScale](#LossScale)
    - [概述](#概述)
    - [FixedLossScaleManager](#FixedLossScaleManager)
    - [DynamicLossScaleManager](#DynamicLossScaleManager)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/programming_guide/source_zh_cn/lossscale.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## 概述

在混合精度中，会使用float16类型来替代float32类型存储数据，从而达到减少内存和提高计算速度的效果。但是由于float16类型要比float32类型表示的范围小很多，所以当某些参数（比如说梯度）在训练过程中变得很小时，就会发生数据下溢的情况。而LossScale正是为了解决float16类型数据下溢问题的，LossScale的主要思想是在计算loss时，将loss扩大一定的倍数，由于链式法则的存在，梯度也会相应扩大，然后在优化器更新权重时再缩小相应的倍数，从而避免了数据下溢的情况又不影响计算结果。

MindSpore中提供了两种LossScale的方式，分别是`FixedLossScaleManager`和`DynamicLossScaleManager`，一般需要和Model配合使用。

## FixedLossScaleManager

`FixedLossScaleManager`在进行scale的时候，不会改变scale的大小，scale的值由入参loss_scale控制，可以由用户指定，不指定则取默认值。`FixedLossScaleManager`的另一个参数是`drop_overflow_update`，用来控制发生溢出时是否更新参数。一般情况下LossScale功能不需要和优化器配合使用，但使用`FixedLossScaleManager`时，如果`drop_overflow_update`为False，那么优化器需设置`loss_scale`的值，且`loss_scale`的值要与`FixedLossScaleManager`的相同。

`FixedLossScaleManager`具体用法如下：

```python
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore.nn import Accuracy
from mindspore import context, Model, FixedLossScaleManager, DynamicLossScaleManager, Tensor
from mindspore.train.callback import LossMonitor
from mindspore.common.initializer import Normal
from mindspore import dataset as ds

mindspore.set_seed(0)
context.set_context(mode=context.GRAPH_MODE)

class LeNet5(nn.Cell):
    """
    Lenet network

    Args:
        num_class (int): Number of classes. Default: 10.
        num_channel (int): Number of channels. Default: 1.

    Returns:
        Tensor, output tensor


    """
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
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

# create dataset
def get_data(num, img_size=(1, 32, 32), num_classes=10, is_onehot=True):
    for _ in range(num):
        img = np.random.randn(*img_size)
        target = np.random.randint(0, num_classes)
        target_ret = np.array([target]).astype(np.float32)
        if is_onehot:
            target_onehot = np.zeros(shape=(num_classes,))
            target_onehot[target] = 1
            target_ret = target_onehot.astype(np.float32)
        yield img.astype(np.float32), target_ret

def create_dataset(num_data=1024, batch_size=32, repeat_size=1):
    input_data = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data','label'])
    input_data = input_data.batch(batch_size, drop_remainder=True)
    input_data = input_data.repeat(repeat_size)
    return input_data

ds_train = create_dataset()

# Initialize network
network = LeNet5(10)

# Define Loss and Optimizer
net_loss = nn.SoftmaxCrossEntropyWithLogits(reduction="mean")

# Define Loss Scale, optimizer and model
#1) Drop the parameter update if there is an overflow
loss_scale_manager = FixedLossScaleManager()
net_opt = nn.Momentum(network.trainable_params(),learning_rate=0.01, momentum=0.9)
model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()}, amp_level="O0", loss_scale_manager=loss_scale_manager)

#2) Execute parameter update even if overflow occurs
loss_scale = 1024.0
loss_scale_manager = FixedLossScaleManager(loss_scale, False)
net_opt = nn.Momentum(network.trainable_params(),learning_rate=0.01, momentum=0.9, loss_scale=loss_scale)
model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()}, amp_level="O0", loss_scale_manager=loss_scale_manager)

# Run training
model.train(epoch=10, train_dataset=ds_train, callbacks=[LossMonitor()])
```

## DynamicLossScaleManager

`DynamicLossScaleManager`在训练过程中可以动态改变scale的大小，在没有发生溢出的情况下，要尽可能保持较大的scale。`DynamicLossScaleManager`会首先将scale设置为一个初始值，该值由入参init_loss_scale控制。在训练过程中，如果不发生溢出，在更新scale_window次参数后，会尝试扩大scale的值，如果发生了溢出，则跳过参数更新，并缩小scale的值，入参scale_factor是控制扩大或缩小的步数，scale_window控制没有发生溢出时，最大的连续更新步数。

`DynamicLossScaleManager`的具体用法如下，仅需将`FixedLossScaleManager`样例中定义LossScale，优化器和模型部分的代码改成如下代码：

```python
# Define Loss Scale, optimizer and model
scale_factor = 4
scale_window = 3000
loss_scale_manager = DynamicLossScaleManager(scale_factor, scale_window)
net_opt = nn.Momentum(network.trainable_params(),learning_rate=0.01, momentum=0.9)
model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()}, amp_level="O0", loss_scale_manager=loss_scale_manager)
```