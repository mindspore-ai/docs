# Running Mode

<a href="https://gitee.com/mindspore/docs/blob/r1.1/docs/programming_guide/source_en/run.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Overview

There are three execution modes: single operator, common function, and network training model.

## Executing a Single Operator

Execute a single operator and output the result.

A code example is as follows:

```python
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

conv = nn.Conv2d(3, 4, 3, bias_init='zeros')
input_data = Tensor(np.ones([1, 3, 5, 5]).astype(np.float32))
output = conv(input_data)
print(output.asnumpy())
```

The output is as follows:

```python
[[[[ 0.06022915  0.06149777  0.06149777  0.06149777  0.01145121]
   [ 0.06402162  0.05889071  0.05889071  0.05889071 -0.00933781]
   [ 0.06402162  0.05889071  0.05889071  0.05889071 -0.00933781]
   [ 0.06402162  0.05889071  0.05889071  0.05889071 -0.00933781]
   [ 0.02712326  0.02096302  0.02096302  0.02096302 -0.01119636]]

  [[-0.0258286  -0.03362969 -0.03362969 -0.03362969 -0.00799183]
   [-0.0513729  -0.06778982 -0.06778982 -0.06778982 -0.03168458]
   [-0.0513729  -0.06778982 -0.06778982 -0.06778982 -0.03168458]
   [-0.0513729  -0.06778982 -0.06778982 -0.06778982 -0.03168458]
   [-0.04186669 -0.07266843 -0.07266843 -0.07266843 -0.04836193]]

  [[-0.00840744 -0.03043237 -0.03043237 -0.03043237  0.00172079]
   [ 0.00401019 -0.03755453 -0.03755453 -0.03755453 -0.00851137]
   [ 0.00401019 -0.03755453 -0.03755453 -0.03755453 -0.00851137]
   [ 0.00401019 -0.03755453 -0.03755453 -0.03755453 -0.00851137]
   [ 0.00270888 -0.03718876 -0.03718876 -0.03718876 -0.03043662]]

  [[-0.00982172  0.02009856  0.02009856  0.02009856  0.03327979]
   [ 0.02529106  0.04035065  0.04035065  0.04035065  0.01782833]
   [ 0.02529106  0.04035065  0.04035065  0.04035065  0.01782833]
   [ 0.02529106  0.04035065  0.04035065  0.04035065  0.01782833]
   [ 0.01015155  0.00781826  0.00781826  0.00781826 -0.02884173]]]]
```

## Executing a Common Function

Combine multiple operators into a function, execute these operators by calling the function, and output the result.

A code example is as follows:

```python
import numpy as np
from mindspore import context, Tensor
from mindspore.ops import functional as F

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

def tensor_add_func(x, y):
    z = F.tensor_add(x, y)
    z = F.tensor_add(z, x)
    return z

x = Tensor(np.ones([3, 3], dtype=np.float32))
y = Tensor(np.ones([3, 3], dtype=np.float32))
output = tensor_add_func(x, y)
print(output.asnumpy())
```

The output is as follows:

```python
[[3. 3. 3.]
 [3. 3. 3.]
 [3. 3. 3.]]
```

## Executing a Network Model

The [Model API](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/mindspore.html#mindspore.Model) of MindSpore is an advanced API used for training and validation. Layers with the training or inference function can be combined into an object. The training, inference, and prediction functions can be implemented by calling the train, eval, and predict APIs, respectively.

You can transfer the initialized Model APIs such as the network, loss function, and optimizer as required. You can also configure amp_level to implement mixed precision and configure metrics to implement model evaluation.

> Executing a network model will produce a `kernel_meta` directory in the current directory, and all the cache of operations compiled during the executing process will be stored in it. If user executes the same model again, or a model with some differences, MindSpore will automatically call the reusable cache in `kernel_meta` to reduce the compilation time of the whole model. It has a significant improvement in performance. The cache usually cannot be shared between different situations, for example, single device and multiple devices, training and inference, etc.
>
> Please note that, if users only delete the cache on part of the devices when executing models on several devices, may lead to a timeout of the waiting time between devices, because only some of them need to recompile the operations. To avoid this situation, users could set the environment variable `HCCL_CONNECT_TIMEOUT` to a reasonable waiting time. However, in this way, the time consuming is the same as deleting all the cache and recompiling. If users interrupt the process of compilation, there is a possibility that the cache file in `kernel_meta` will be generated incorrectly, and the subsequent re-execution process will fail. In this case, users need to delete the `kernel_mata` folder and recompile the network.

### Executing a Training Model

Call the train API of Model to implement training.

A code example is as follows:

```python
import os

import mindspore.dataset.vision.c_transforms as CV
from mindspore.dataset.vision import Inter

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as CT
import mindspore.dataset.vision.c_transforms as CV
import mindspore.nn as nn
from mindspore import context, Model
from mindspore import dtype as mstype
from mindspore.common.initializer import Normal
from mindspore.common.initializer import TruncatedNormal
from mindspore.dataset.vision import Inter
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
        num_channel (int): Num channels. Default: 1.

    Returns:
        Tensor, output tensor
    Examples:
        >>> LeNet(num_class=10)

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


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    ds_train = create_dataset(os.path.join("/home/workspace/mindspore_dataset/MNIST_Data/", "train"), 32)

    network = LeNet5(10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)
    model = Model(network, net_loss, net_opt)

    print("============== Starting Training ==============")
    model.train(1, ds_train, callbacks=[LossMonitor()], dataset_sink_mode=False)
```

> For details about how to obtain the MNIST dataset used in the example, see [Downloading the Dataset](https://www.mindspore.cn/tutorial/training/en/r1.1/quick_start/quick_start.html#downloading-the-dataset).

The output is as follows:

```python
epoch: 1 step: 1, loss is 2.300784
epoch: 1 step: 2, loss is 2.3076947
epoch: 1 step: 3, loss is 2.2993166
...
epoch: 1 step: 1873, loss is 0.13014838
epoch: 1 step: 1874, loss is 0.0346688
epoch: 1 step: 1875, loss is 0.017264696
```

> Use the PyNative mode for debugging, including the execution of single operator, common function, and network training model. For details, see [Debugging in PyNative Mode](https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/debug_in_pynative_mode.html).

### Executing an Inference Model

Call the eval API of Model to implement inference. To facilitate model evaluation, you can set metrics when the Model API is initialized.

Metrics are used to evaluate models. Common metrics include Accuracy, Fbeta, Precision, Recall, and TopKCategoricalAccuracy. Generally, the comprehensive model quality cannot be evaluated by one model metric. Therefore, multiple metrics are often used together to evaluate the model.

Common built-in evaluation metrics are as follows:

- `Accuracy`: evaluates a classification model. Generally, accuracy refers to the percentage of results correctly predicted by the model to all results. Formula: $$Accuracy = (TP + TN)/(TP + TN + FP + FN)$$

- `Precision`: percentage of correctly predicted positive results to all predicted positive results. Formula: $$Precision = TP/(TP + FP)$$

- `Recall`: percentage of correctly predicted positive results to all actual positive results. Formula: $$Recall = TP/(TP + FN)$$

- `Fbeta`: harmonic mean of precision and recall.

Formula: $$F_\beta = (1 + \beta^2) \cdot \frac{precisiont \cdot recall}{(\beta^2 \cdot precision) + recall}$$

- `TopKCategoricalAccuracy`: calculates the top K categorical accuracy.

A code example is as follows:

```python
import os

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as CT
import mindspore.dataset.vision.c_transforms as CV
import mindspore.nn as nn
from mindspore import context, Model, load_checkpoint, load_param_into_net
from mindspore import dtype as mstype
from mindspore.common.initializer import Normal
from mindspore.dataset.vision import Inter
from mindspore.nn import Accuracy, Precision


class LeNet5(nn.Cell):
    """
    Lenet network

    Args:
        num_class (int): Num classes. Default: 10.
        num_channel (int): Num channels. Default: 1.

    Returns:
        Tensor, output tensor
    Examples:
        >>> LeNet(num_class=10)

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


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    network = LeNet5(10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    repeat_size = 1
    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy(), "Precision": Precision()})

    print("============== Starting Testing ==============")
    param_dict = load_checkpoint("./ckpt/checkpoint_lenet-1_1875.ckpt")
    load_param_into_net(network, param_dict)
    ds_eval = create_dataset(os.path.join("/home/workspace/mindspore_dataset/MNIST_Data", "test"), 32, repeat_size)
    acc = model.eval(ds_eval, dataset_sink_mode=True)
    print("============== {} ==============".format(acc))
```

In the preceding information:

- `load_checkpoint`: loads the checkpoint model parameter file and returns a parameter dictionary.
- `checkpoint_lenet-1_1875.ckpt`: name of the saved checkpoint model file.
- `load_param_into_net`: loads parameters to the network.

> For details about how to save the `checkpoint_lenet-1_1875.ckpt` file, see [Training the Network](https://www.mindspore.cn/tutorial/training/en/r1.1/quick_start/quick_start.html#training-the-network).

The output is as follows:

```python
============== {'Accuracy': 0.96875, 'Precision': array([0.97782258, 0.99451052, 0.98031496, 0.92723881, 0.98352214,
       0.97165533, 0.98726115, 0.9472196 , 0.9394551 , 0.98236515])} ==============
```
