# 训练

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/docs/programming_guide/source_zh_cn/train.md)

## 概述

MindSpore在Model_zoo也已经提供了大量的目标检测、自然语言处理等多种网络模型，供用户直接使用，但是对于某些高级用户而言可能想要自行设计网络或者自定义训练循环，下面就对自定义训练网络、自定义训练循环和边训练边推理三种场景进行介绍，另外对On device执行方式进行详细介绍。

## 自定义训练网络

在自定义训练网络前，需要先了解下MindSpore的网络支持、Python源码构造网络约束和算子支持情况。

- 网络支持：当前MindSpore已经支持多种网络，按类型分为计算机视觉、自然语言处理、推荐和图神经网络，可以通过[网络支持](https://www.mindspore.cn/doc/note/zh-CN/r1.0/network_list.html)查看具体支持的网络情况。如果现有网络无法满足用户需求，用户可以根据实际需要定义自己的网络。

- Python源码构造网络约束：MindSpore暂不支持将任意Python源码转换成计算图，所以对于用户源码支持的写法有所限制，主要包括语法约束和网络定义约束两方面。详细情况可以查看[Python源码构造网络约束](https://www.mindspore.cn/doc/note/zh-CN/r1.0/constraints_on_network_construction.html)了解。随着MindSpore的演进，这些约束可能会发生变化。

- 算子支持：顾名思义，网络的基础是算子，所以用户自定义训练网络前要对MindSpore当前支持的算子有所了解，可以通过查看[算子支持](https://www.mindspore.cn/doc/note/zh-CN/r1.0/operator_list.html)了解不同的后端（Ascend、GPU和CPU）的算子实现情况。

> 当开发网络遇到内置算子不足以满足需求时，用户也可以参考[自定义算子](https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/advanced_use/custom_operator_ascend.html)，方便快捷地扩展昇腾AI处理器的自定义算子。

代码样例如下：

```python
import numpy as np

from mindspore.common.tensor import Tensor
from mindspore.nn import Cell, Dense, SoftmaxCrossEntropyWithLogits, Momentum, TrainOneStepCell, WithLossCell
import mindspore.ops as ops


class ReLUReduceMeanDense(Cell):
    def __init__(self, kernel, bias, in_channel, num_class):
        super().__init__()
        self.relu = ops.ReLU()
        self.mean = ops.ReduceMean(keep_dims=False)
        self.dense = Dense(in_channel, num_class, kernel, bias)

    def construct(self, x):
        x = self.relu(x)
        x = self.mean(x, (2, 3))
        x = self.dense(x)
        return x


if __name__ == "__main__":
    weight_np = np.ones((1000, 2048)).astype(np.float32)
    weight = Tensor(weight_np.copy())
    bias_np = np.ones((1000,)).astype(np.float32)
    bias = Tensor(bias_np.copy())
    net = ReLUReduceMeanDense(weight, bias, 2048, 1000)
    criterion = SoftmaxCrossEntropyWithLogits(sparse=False)
    optimizer = Momentum(learning_rate=0.1, momentum=0.1,
                         params=filter(lambda x: x.requires_grad, net.get_parameters()))
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()
    input_np = np.random.randn(32, 2048, 7, 7).astype(np.float32)
    input = Tensor(input_np.copy())
    label_np_onehot = np.zeros(shape=(32, 1000)).astype(np.float32)
    label = Tensor(label_np_onehot.copy())
    for i in range(1):
        loss = train_network(input, label)
        print("-------loss------", loss)
```

输出如下：

```python
-------loss------ [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0.]
```

## 自定义训练循环

用户如果不想使用MindSpore提供的Model接口，可以将模仿Model的train接口自由控制循环的迭代次数和每个epoch的step数量。

代码样例如下：

```python
import os

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as CT
import mindspore.dataset.vision.c_transforms as CV
import mindspore.nn as nn
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.common.initializer import TruncatedNormal
from mindspore.common.parameter import ParameterTuple
from mindspore.dataset.vision import Inter
from mindspore.nn.wrap.cell_wrapper import WithLossCell
import mindspore.ops as ops
from mindspore.train.dataset_helper import DatasetHelper, connect_network_with_dataset


def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    """
    create dataset for train or test
    """
    # define dataset
    mnist_ds = ds.MnistDataset(data_path)

    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)  # Bilinear mode
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = CT.TypeCast(mstype.int32)

    # apply map operations on images
    mnist_ds = mnist_ds.map(input_columns="label", operations=type_cast_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=resize_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=rescale_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=rescale_nml_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=hwc2chw_op, num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)  # 10000 as in LeNet train script
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


class LeNet5(nn.Cell):
    """
    Lenet network
    Args:
        num_class (int): Num classes. Default: 10.

    Returns:
        Tensor, output tensor

    Examples:
        >>> LeNet(num_class=10)
    """

    def __init__(self, num_class=10):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.batch_size = 32
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = ops.Reshape()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.reshape(x, (self.batch_size, -1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class TrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def set_sens(self, value):
        self.sens = value

    def construct(self, data, label):
        weights = self.weights
        loss = self.network(data, label)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(data, label, sens)
        return ops.depend(loss, self.optimizer(grads))


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    ds_train = create_dataset(os.path.join("/home/workspace/mindspore_dataset/MNIST_Data/", "train"), 32)

    network = LeNet5(10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)
    net = WithLossCell(network, net_loss)
    net = TrainOneStepCell(net, net_opt)
    dataset_helper = DatasetHelper(ds_train, dataset_sink_mode=True, sink_size=100, epoch_num=10)
    net = connect_network_with_dataset(net, dataset_helper)
    network.set_train()
    print("============== Starting Training ==============")
    epoch = 10
    for step in range(epoch):
        for inputs in dataset_helper:
            output = net(*inputs)
            print("epoch: {0}/{1}, losses: {2}".format(step + 1, epoch, output.asnumpy(), flush=True))
```

> 示例中用到的MNIST数据集的获取方法，可以参照[实现一个图片分类应用](https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/quick_start/quick_start.html)的下载数据集部分，下同。

输出如下：

```python
epoch: 1/10, losses: 2.294034719467163
epoch: 2/10, losses: 2.3150298595428467
epoch: 3/10, losses: 2.3107073307037354
epoch: 4/10, losses: 2.3155436515808105
epoch: 5/10, losses: 2.28973388671875
epoch: 6/10, losses: 2.3108928203582764
epoch: 7/10, losses: 2.293713092803955
epoch: 8/10, losses: 2.29837703704834
epoch: 9/10, losses: 2.305952548980713
epoch: 10/10, losses: 1.4282708168029785
```

> 典型的使用场景是梯度累积，详细查看[梯度累积](https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/advanced_use/apply_gradient_accumulation.html)。

## 边训练边推理

对于某些数据量较大、训练时间较长的复杂网络，为了能掌握训练的不同阶段模型精度的指标变化情况，可以通过边训练边推理的方式跟踪精度的变化情况。具体可以参考[同步训练和验证模型](https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/advanced_use/evaluate_the_model_during_training.html)。

## on-device执行

当前MindSpore支持的后端包括Ascend、GPU、CPU，所谓On Device中的Device通常指Ascend（昇腾）AI处理器。

昇腾芯片上集成了AICORE、AICPU和CPU。其中，AICORE负责大型Tensor Vector运算，AICPU负责标量运算，CPU负责逻辑控制和任务分发。

Host侧CPU负责将图或算子下发到昇腾芯片。昇腾芯片由于具备了运算、逻辑控制和任务分发的功能，所以不需要与Host侧的CPU进行频繁的交互，只需要将计算完的最终结果返回给Host侧，实现整图下沉到Device执行，避免Host-Device频繁交互，减小了开销。

以下是Device的主要组成结构：

- 片上32G内存：5G(parameter) + 26G(feature map) + 1G(HCCL)
- 多流水线并行：6条流水线
- AICORE&带宽：32Cores、读写带宽128GBps
- 通信协议：HCCS、PCIe4.0、RoCEv2

### 计算图下沉

计算图整图下沉到Device上执行，减少Host-Device交互开销。可以结合循环下沉实现多个Step下沉，进一步减少Host和Device的交互次数。

循环下沉是在On Device执行的基础上的优化，目的是进一步减少Host侧和Device侧之间的交互次数。通常情况下，每个step都返回一个结果，循环下沉是控制每隔多少个step返回一次结果。

默认配置下是每一个epoch返回一次结果，这样每个epoch里，Host侧和Device侧只需要进行一次数据交互。

也可以结合`train`接口的`dataset_sink_mode`和`sink_size`控制每个epoch的下沉数据量。

### 数据下沉

`Model`的`train`接口参数`dataset_sink_mode`可以控制数据是否下沉。`dataset_sink_mode`为True表示数据下沉，否则为非下沉。所谓下沉即数据通过通道直接传送到Device上。

dataset_sink_mode参数可以配合`sink_size`控制每个`epoch`下沉的数据量大小。当`dataset_sink_mode`设置为True，即数据下沉模式时：

如果`sink_size`为默认值-1，则每一个`epoch`下沉的数据量为原始的整个数据集大小；

如果`sink_size`>0，此时原始数据集可以被无限次遍历，每个`epoch`下沉`sink_size`大小的数据量，下一个`epoch`继续从上次遍历的结束位置继续遍历。

下沉的总数据量由`epoch`和`sink_size`两个变量共同控制，即总数据量=`epoch`*`sink_size`。

代码样例如下：

```python
import os

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as CT
import mindspore.dataset.vision.c_transforms as CV
import mindspore.nn as nn
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.common.initializer import TruncatedNormal
from mindspore.dataset.vision import Inter
from mindspore.nn.metrics import Accuracy
import mindspore.ops as ops
from mindspore.train import Model
from mindspore.train.callback import LossMonitor


def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    """
    create dataset for train or test
    """
    # define dataset
    mnist_ds = ds.MnistDataset(data_path)

    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)  # Bilinear mode
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = CT.TypeCast(mstype.int32)

    # apply map operations on images
    mnist_ds = mnist_ds.map(input_columns="label", operations=type_cast_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=resize_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=rescale_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=rescale_nml_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=hwc2chw_op, num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)  # 10000 as in LeNet train script
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


class LeNet5(nn.Cell):
    """
    Lenet network
    Args:
        num_class (int): Num classes. Default: 10.

    Returns:
        Tensor, output tensor

    Examples:
        >>> LeNet(num_class=10)
    """

    def __init__(self, num_class=10):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.batch_size = 32
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = ops.Reshape()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.reshape(x, (self.batch_size, -1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    ds_train = create_dataset(os.path.join("/home/workspace/mindspore_dataset/MNIST_Data/", "train"), 32)

    network = LeNet5(10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)
    model = Model(network, net_loss, net_opt)

    print("============== Starting Training ==============")
    model.train(epoch=10, train_dataset=ds_train, callbacks=[LossMonitor()], dataset_sink_mode=True, sink_size=1000)
```

`batch_size`为32的情况下，数据集的大小为1875，当`sink_size`设置为1000时，表示每个`epoch`下沉1000个batch的数据，下沉次数为`epoch`=10，下沉的总数据量为：`epoch`*`sink_size`=10000。

输出如下：

```python
epoch: 1 step: 1000, loss is 0.5399815
epoch: 2 step: 1000, loss is 0.033433747
epoch: 3 step: 1000, loss is 0.054761313
epoch: 4 step: 1000, loss is 0.007882872
epoch: 5 step: 1000, loss is 0.00658499
epoch: 6 step: 1000, loss is 0.0413095
epoch: 7 step: 1000, loss is 0.13373856
epoch: 8 step: 1000, loss is 0.015793817
epoch: 9 step: 1000, loss is 0.00017951085
epoch: 10 step: 1000, loss is 0.01490275
```

> `dataset_sink_mode`为False时，`sink_size`参数设置无效。
