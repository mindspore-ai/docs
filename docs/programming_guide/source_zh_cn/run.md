# 运行方式

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/programming_guide/source_zh_cn/run.md)
&nbsp;&nbsp;
[![查看notebook](./_static/logo_notebook.png)](https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/r1.1/programming_guide/mindspore_run.ipynb)
&nbsp;&nbsp;
[![在线运行](./_static/logo_modelarts.png)](https://console.huaweicloud.com/modelarts/?region=cn-north-4#/notebook/loading?share-url-b64=aHR0cHM6Ly9vYnMuZHVhbHN0YWNrLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vbWluZHNwb3JlLXdlYnNpdGUvbm90ZWJvb2svbW9kZWxhcnRzL3Byb2dyYW1taW5nX2d1aWRlL21pbmRzcG9yZV9ydW4uaXB5bmI=&image_id=65f636a0-56cf-49df-b941-7d2a07ba8c8c)

## 概述

执行主要有三种方式：单算子、普通函数和网络训练模型。

## 执行单算子

执行单个算子，并打印相关结果。

代码样例如下：

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

> 由于weight初始化存在随机因素，实际输出结果可能不同，仅供参考。

输出如下：

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

## 执行普通函数

将若干算子组合成一个函数，然后直接通过函数调用的方式执行这些算子，并打印相关结果，如下例所示。

代码样例如下：

```python
import numpy as np
from mindspore import context, Tensor
import mindspore.ops as ops

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

def tensor_add_func(x, y):
    z = ops.tensor_add(x, y)
    z = ops.tensor_add(z, x)
    return z

x = Tensor(np.ones([3, 3], dtype=np.float32))
y = Tensor(np.ones([3, 3], dtype=np.float32))
output = tensor_add_func(x, y)
print(output.asnumpy())
```

输出如下：

```python
[[3. 3. 3.]
 [3. 3. 3.]
 [3. 3. 3.]]
```

## 执行网络模型

MindSpore的[Model接口](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/mindspore.html#mindspore.Model)是用于训练和验证的高级接口。可以将有训练或推理功能的layers组合成一个对象，通过调用train、eval、predict接口可以分别实现训练、推理和预测功能。

用户可以根据实际需要传入网络、损失函数和优化器等初始化Model接口，还可以通过配置amp_level实现混合精度，配置metrics实现模型评估。

> 执行网络模型会在执行目录下生成`kernel_meta`目录，并在执行过程中保存网络编译生成的算子缓存文件到此目录，包括`.o`文件和`.json`文件。若用户再次执行相同的网络模型，或者仅有部分变化，MindSpore会自动调用`kernel_meta`目录下可复用的算子缓存文件，显著减少网络编译时间，提升执行性能。不同场景下缓存文件通常不能共用，例如多卡与单卡、训练与推理等。
>
> 请注意，在多卡运行的情况下，如果仅删除部分卡的`kernel_meta`下的算子缓存文件后重复执行相同的网络模型，可能会引起不需重新编译算子的部分卡等候超时，导致执行失败。在这种情况下，可以通过设置环境变量`HCCL_CONNECT_TIMEOUT`，即多卡间等待时间来避免失败，但该方式耗时等同于全部删除缓存重新编译。如果在网络编译的过程中打断进程，有概率会导致`kernel_meta`中的缓存文件生成错误，并使得后续重新执行的过程失败。此时需要用户去删除`kernel_mata`文件夹，重新编译网络。

### 执行训练模型

通过调用Model的train接口可以实现训练。

代码样例如下：

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

> 示例中用到的MNIST数据集的获取方法，可以参照[实现一个图片分类应用](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/quick_start/quick_start.html)的下载数据集部分，下同。

输出如下：

```text
============== Starting Training ==============
epoch: 1 step: 1, loss is 2.300784
epoch: 1 step: 2, loss is 2.3076947
epoch: 1 step: 3, loss is 2.2993166
...
epoch: 1 step: 1873, loss is 0.13014838
epoch: 1 step: 1874, loss is 0.0346688
epoch: 1 step: 1875, loss is 0.017264696
```

> 使用PyNative模式调试， 请参考[使用PyNative模式调试](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/advanced_use/debug_in_pynative_mode.html)， 包括单算子、普通函数和网络训练模型的执行。

### 执行推理模型

通过调用Model的eval接口可以实现推理。为了方便评估模型的好坏，可以在Model接口初始化的时候设置评估指标Metric。

Metric是用于评估模型好坏的指标。常见的主要有Accuracy、Fbeta、Precision、Recall和TopKCategoricalAccuracy等，通常情况下，一种模型指标无法全面的评估模型的好坏，一般会结合多个指标共同作用对模型进行评估。

常用的内置评估指标：

- `Accuracy`（准确率）：是一个用于评估分类模型的指标。通俗来说，准确率是指我们的模型预测正确的结果所占的比例。 公式：$$Accuracy = （TP+TN）/（TP+TN+FP+FN）$$

- `Precision`（精确率）：在被识别为正类别的样本中，确实为正类别的比例。公式：$$Precision = TP/(TP+FP)$$

- `Recall`（召回率）：在所有正类别样本中，被正确识别为正类别的比例。 公式：$$Recall = TP/(TP+FN)$$

- `Fbeta`（调和均值）：综合考虑precision和recall的调和均值。

    公式：$$F_\beta = (1 + \beta^2) \cdot \frac{precisiont \cdot recall}{(\beta^2 \cdot precision) + recall}$$

- `TopKCategoricalAccuracy`（多分类TopK准确率）：计算TopK分类准确率。

代码样例如下：

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
from mindspore.nn.metrics import Accuracy, Precision


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

其中：

- `load_checkpoint`：通过该接口加载CheckPoint模型参数文件，返回一个参数字典。
- `checkpoint_lenet-1_1875.ckpt`：保存的CheckPoint模型文件名称。
- `load_param_into_net`：通过该接口把参数加载到网络中。

> `checkpoint_lenet-1_1875.ckpt`文件的保存方法，可以参考[实现一个图片分类应用](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/quick_start/quick_start.html)的训练网络部分。

输出如下：

```python
============== {'Accuracy': 0.96875, 'Precision': array([0.97782258, 0.99451052, 0.98031496, 0.92723881, 0.98352214,
       0.97165533, 0.98726115, 0.9472196 , 0.9394551 , 0.98236515])} ==============
```
