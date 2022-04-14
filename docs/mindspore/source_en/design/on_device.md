# On-Device Execution

`Ascend` `GPU` `CPU` `Model Running`

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/design/on_device.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

## Overview

The backends supported by MindSpore include Ascend, GPU, and CPU. The device in the "On-Device" refers to the Ascend AI processor.

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

If `sink_size` is set to the default value –1, each `epoch` trains the whole dataset. Ideally, the speed of sinking data is faster than hardware calculation, so as to ensure that the time spent in processing data is hidden in the network calculation time.

If `sink_size` is greater than 0, the raw dataset can be traversed for an unlimited number of times, sinking data flow is still the same as `sink_size` = -1, except that each `epoch` only trains `sink_size` amount of data. If there is `LossMonitor`, it will train `sink_size` amount of data and print the loss value once, and the next `epoch` continues to traverse from the end position of the previous traversal.

The total sunk data volume is controlled by the `epoch` and `sink_size` variables. That is, the total data volume is calculated as follows: Total data volume = `epoch` x `sink_size`.

When using `LossMonitor`, `TimeMonitor` or other `Callback` interfaces, if the `dataset_sink_mode` is set to False, each `step` between the Host side and the Device side interacts once, so each `step` will return a result. If `dataset_sink_mode` is True, because the data is transmitted through the channel on the Device, there is one data interaction between the Host side and the Device side for each `epoch`, so each `epoch` only returns one result.

> Currently dataset sink mode is not supported on CPU target.
> If `fault kernel_name=GetNext` or `GetNext... task error` or `outputs = self.get_next()` error info occurs, it may be that some sample processing in the data processing process is too time-consuming, resulting in the failure of the network computing side to get the data for a long time and report an error. At this time, you can set `dataset_sink_mode` to False to verify again, or use `create_dict_iterator()` interface separate cyclic dataset and refer to [Optimizing the Data Processing](https://www.mindspore.cn/tutorials/experts/en/r1.7/dataset/optimize.html) optimize data processing to ensure high performance of data processing.

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
import mindspore.ops as ops
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
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    ds_train_path = "./datasets/MNIST_Data/train/"
    ds_train = create_dataset(ds_train_path, 32)

    network = LeNet5(10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)
    model = Model(network, net_loss, net_opt)

    print("============== Starting Training ==============")
    model.train(epoch=10, train_dataset=ds_train, callbacks=[LossMonitor()], dataset_sink_mode=True, sink_size=1000)
```

The output is as follows:

```text
============== Starting Training ==============
epoch: 1 step: 1000, loss is 0.110185064
epoch: 2 step: 1000, loss is 0.12088283
epoch: 3 step: 1000, loss is 0.15903473
epoch: 4 step: 1000, loss is 0.030054657
epoch: 5 step: 1000, loss is 0.013846226
epoch: 6 step: 1000, loss is 0.052161213
epoch: 7 step: 1000, loss is 0.0050197737
epoch: 8 step: 1000, loss is 0.17207858
epoch: 9 step: 1000, loss is 0.010310417
epoch: 10 step: 1000, loss is 0.000672762
```

When `batch_size` is 32, the size of the dataset is 1875. When `sink_size` is set to 1000, each `epoch` sinks 1000 batches of data, the number of sinks is `epoch` (=10), and the total sunk data volume is `epoch` x `sink_size` = 10000.

`dataset_sink_mode` is True, so every `epoch` returns a result.

`DatasetHelper` is a class to process the dataset and provide information of the dataset. In sink mode, `mindspore.connect_network_with_dataset` function is used to connect the current training network or evaluate network `network` and `DatasetHelper`, this function wraps the input `network` with `GetNext`  so that the data can be fetched automatically from the data channel with the corresponding name `queue_name` on the device side during forward computation, and pass the data to the input `network`.  In the no-sink mode, the data set is fetched at host side by iterating through the dataset.

> When `dataset_sink_mode` is set to False, the `sink_size` parameter is invalid.
