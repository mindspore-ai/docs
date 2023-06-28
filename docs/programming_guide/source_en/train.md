# Training

<a href="https://gitee.com/mindspore/docs/blob/r1.1/docs/programming_guide/source_en/train.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Overview

MindSpore provides a large number of network models such as object detection and natural language processing in ModelZoo for users to directly use. However, some senior users may want to design networks or customize training cycles. The following describes how to customize a training network, how to customize a training cycle, and how to conduct inference while training. In addition, the on-device execution mode is also described in detail.

## Customizing a Training Network

Before customizing a training network, you need to understand the network support of MindSpore, constraints on network construction using Python, and operator support.

- Network support: Currently, MindSpore supports multiple types of networks, including computer vision, natural language processing, recommender, and graph neural network. For details, see [Network List](https://www.mindspore.cn/doc/note/en/r1.1/network_list.html). If the existing networks cannot meet your requirements, you can define your own network as required.

- Constraints on network construction using Python: MindSpore does not support the conversion of any Python source code into computational graphs. Therefore, the source code has the syntax and network definition constraints. These constraints may change as MindSpore evolves.

- Operator support: As the name implies, the network is based on operators. Therefore, before customizing a training network, you need to understand the operators supported by MindSpore. For details about operator implementation on different backends (Ascend, GPU, and CPU), see [Operator List](https://www.mindspore.cn/doc/note/en/r1.1/operator_list.html).

> When the built-in operators of the network cannot meet the requirements, you can refer to [Custom Operators(Ascend)](https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/custom_operator_ascend.html) to quickly expand the custom operators of the Ascend AI processor.

The following is a code example:

```python
import numpy as np

from mindspore import Tensor
from mindspore.nn import Cell, Dense, SoftmaxCrossEntropyWithLogits, Momentum, TrainOneStepCell, WithLossCell
from mindspore.ops import operations as P


class ReLUReduceMeanDense(Cell):
    def __init__(self, kernel, bias, in_channel, num_class):
        super().__init__()
        self.relu = P.ReLU()
        self.mean = P.ReduceMean(keep_dims=False)
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

The output is as follows:

```python
-------loss------ [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0.]
```

## Customizing a Training Cycle

If you do not want to use the `Model` interface provided by MindSpore, you can use the `train` interface to control the number of iterations and the number of steps for each epoch.

The following is a code example:

```python
import os

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as CT
import mindspore.dataset.vision.c_transforms as CV
import mindspore.nn as nn
from mindspore import context, DatasetHelper, connect_network_with_dataset
from mindspore import dtype as mstype
from mindspore.common.initializer import TruncatedNormal
from mindspore import ParameterTuple
from mindspore.dataset.vision import Inter
from mindspore.nn import WithLossCell
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P


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
        self.reshape = P.Reshape()

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
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def set_sens(self, value):
        self.sens = value

    def construct(self, data, label):
        weights = self.weights
        loss = self.network(data, label)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(data, label, sens)
        return F.depend(loss, self.optimizer(grads))


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

> For details about how to obtain the MNIST dataset used in the example, see [Downloading the Dataset](https://www.mindspore.cn/tutorial/training/en/r1.1/quick_start/quick_start.html#downloading-the-dataset).

The output is as follows:

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

> The typical application scenario is gradient accumulation. For details, see [Applying Gradient Accumulation Algorithm](https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/apply_gradient_accumulation.html).

## Conducting Inference While Training

For some complex networks with a large data volume and a relatively long training time, to learn the change of model accuracy in different training phases, the model accuracy may be traced in a manner of inference while training. For details, see [Evaluating the Model during Training](https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/evaluate_the_model_during_training.html).

## On-Device Execution

Currently, the backends supported by MindSpore include Ascend, GPU, and CPU. The device in the "On-Device" refers to the Ascend AI processor.

The Ascend AI processor integrates the AI core, AI CPU, and CPU. The AI core is responsible for large Tensor Vector computing, the AI CPU is responsible for scalar computing, and the CPU is responsible for logic control and task distribution.

The CPU on the host side delivers graphs or operators to the Ascend AI processor. The Ascend AI processor has the functions of computing, logic control, and task distribution. Therefore, it does not need to frequently interact with the CPU on the host side. It only needs to return the final calculation result to the host. In this way, the entire graph is sunk to the device for execution, avoiding frequent interaction between the host and device and reducing overheads.

### Computational Graphs on Devices

The entire graph is executed on the device to reduce the interaction overheads between the host and device. Multiple steps can be moved downwards together with cyclic sinking to further reduce the number of interactions between the host and device.

Cyclic sinking is optimized based on on-device execution to further reduce the number of interactions between the host and device. Generally, each step returns a result. Cyclic sinking is used to control the number of steps at which a result is returned.

By default, the result is returned for each epoch. In this way, the host and device need to exchange data only once in each epoch.

You can also use `dataset_sink_mode` and `sink_size` of the `train` interface to control the sunk data volume of each epoch.

### Data Sinking

The `train` interface parameter `dataset_sink_mode` of `Model` can be used to control whether data sinks. If the value of `dataset_sink_mode` is True, data sinking is enabled. Otherwise, data sinking is disabled. Sinking means that data is directly transmitted to the device through a channel.

The `dataset_sink_mode` parameter can be used with `sink_size` to control the amount of data sunk by each `epoch`. When `dataset_sink_mode` is set to True, that is, the data sinking mode is used:

If `sink_size` is set to the default value –1, the amount of data sunk by each `epoch` is the size of the original entire dataset.

If `sink_size` is greater than 0, the raw dataset can be traversed for an unlimited number of times. Each `epoch` sinks the data volume of `sink_size`, and the next `epoch` continues to traverse from the end position of the previous traversal.

The total sunk data volume is controlled by the `epoch` and `sink_size` variables. That is, the total data volume is calculated as follows: Total data volume = `epoch` x `sink_size`.

> The CPU and pynative mode cannot support dataset sink mode currently.

The following is a code example:

```python
import os

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as CT
import mindspore.dataset.vision.c_transforms as CV
import mindspore.nn as nn
from mindspore import context, Model
from mindspore import dtype as mstype
from mindspore.common.initializer import TruncatedNormal
from mindspore.dataset.vision import Inter
from mindspore.nn import Accuracy
from mindspore.ops import operations as P
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
        self.reshape = P.Reshape()

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

When `batch_size` is 32, the size of the dataset is 1875. When `sink_size` is set to 1000, each `epoch` sinks 1000 batches of data, the number of sinks is `epoch` (=10), and the total sunk data volume is `epoch` x `sink_size` = 10000.

The output is as follows:

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

> When `dataset_sink_mode` is set to False, the `sink_size` parameter is invalid.
