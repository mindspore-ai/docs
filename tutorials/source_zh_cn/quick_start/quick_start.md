# 实现一个图片分类应用


[![查看源文件](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r0.6/tutorials/source_zh_cn/quick_start/quick_start.md)
&nbsp;&nbsp;
[![查看Notebook](../_static/logo_notebook.png)](https://gitee.com/mindspore/docs/blob/r0.6/tutorials/notebook/quick_start.ipynb)

## 概述

下面我们通过一个实际样例，带领大家体验MindSpore基础的功能，对于一般的用户而言，完成整个样例实践会持续20~30分钟。

本例子会实现一个简单的图片分类的功能，整体流程如下：
1. 处理需要的数据集，这里使用了MNIST数据集。
2. 定义一个网络，这里我们使用LeNet网络。
3. 定义损失函数和优化器。
4. 加载数据集并进行训练，训练完成后，查看结果及保存模型文件。
5. 加载保存的模型，进行推理。
6. 验证模型，加载测试数据集和训练后的模型，验证结果精度。

> 你可以在这里找到完整可运行的样例代码：<https://gitee.com/mindspore/docs/blob/r0.6/tutorials/tutorial_code/lenet.py> 。


这是简单、基础的应用流程，其他高级、复杂的应用可以基于这个基本流程进行扩展。

## 准备环节

在动手进行实践之前，确保，你已经正确安装了MindSpore。如果没有，可以通过[MindSpore安装页面](https://www.mindspore.cn/install)将MindSpore安装在你的电脑当中。  

同时希望你拥有Python编码基础和概率、矩阵等基础数学知识。

那么接下来，就开始MindSpore的体验之旅吧。

### 下载数据集

我们示例中用到的`MNIST`数据集是由10类28*28的灰度图片组成，训练数据集包含60000张图片，测试数据集包含10000张图片。

> MNIST数据集下载页面：<http://yann.lecun.com/exdb/mnist/>。页面提供4个数据集下载链接，其中前2个文件是训练数据需要，后2个文件是测试结果需要。

将数据集下载并解压到本地路径下，这里将数据集解压分别存放到工作区的`./MNIST_Data/train`、`./MNIST_Data/test`路径下。

目录结构如下：

```
└─MNIST_Data
    ├─test
    │      t10k-images.idx3-ubyte
    │      t10k-labels.idx1-ubyte
    │
    └─train
            train-images.idx3-ubyte
            train-labels.idx1-ubyte
```
> 为了方便样例使用，我们在样例脚本中添加了自动下载数据集的功能。

### 导入Python库&模块

在使用前，需要导入需要的Python库。

目前使用到`os`库，为方便理解，其他需要的库，我们在具体使用到时再说明。
 
 
```python
import os
```

详细的MindSpore的模块说明，可以在[MindSpore API页面](https://www.mindspore.cn/api/zh-CN/r0.6/index.html)中搜索查询。

### 配置运行信息

在正式编写代码前，需要了解MindSpore运行所需要的硬件、后端等基本信息。

可以通过`context.set_context`来配置运行需要的信息，譬如运行模式、后端信息、硬件等信息。

导入`context`模块，配置运行需要的信息。

```python
import argparse
from mindspore import context

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: CPU)')
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    dataset_sink_mode = not args.device_target == "CPU"
    ...
```

在样例中我们配置样例运行使用图模式。根据实际情况配置硬件信息，譬如代码运行在Ascend AI处理器上，则`--device_target`选择`Ascend`，代码运行在CPU、GPU同理。详细参数说明，请参见`context.set_context`接口说明。

## 数据处理

数据集对于训练非常重要，好的数据集可以有效提高训练精度和效率。在加载数据集前，我们通常会对数据集进行一些处理。

### 定义数据集及数据操作

我们定义一个函数`create_dataset`来创建数据集。在这个函数中，我们定义好需要进行的数据增强和处理操作：

1. 定义数据集。
2. 定义进行数据增强和处理所需要的一些参数。
3. 根据参数，生成对应的数据增强操作。
4. 使用`map`映射函数，将数据操作应用到数据集。
5. 对生成的数据集进行处理。

```python
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.transforms.vision.c_transforms as CV
from mindspore.dataset.transforms.vision import Inter
from mindspore.common import dtype as mstype

def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    """ create dataset for train or test
    Args:
        data_path: Data path
        batch_size: The number of data records in each group
        repeat_size: The number of replicated data records
        num_parallel_workers: The number of parallel workers
    """
    # define dataset
    mnist_ds = ds.MnistDataset(data_path)

    # define operation parameters
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)  # resize images to (32, 32)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)  # normalize images
    rescale_op = CV.Rescale(rescale, shift)  # rescale images
    hwc2chw_op = CV.HWC2CHW()  # change shape from (height, width, channel) to (channel, height, width) to fit network.
    type_cast_op = C.TypeCast(mstype.int32)  # change data type of label to int32 to fit network
    
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

```

其中，  
`batch_size`：每组包含的数据个数，现设置每组包含32个数据。  
`repeat_size`：数据集复制的数量。

先进行shuffle、batch操作，再进行repeat操作，这样能保证1个epoch内数据不重复。

> MindSpore支持进行多种数据处理和增强的操作，各种操作往往组合使用，具体可以参考[数据处理与数据增强](https://www.mindspore.cn/tutorial/zh-CN/r0.6/use/data_preparation/data_processing_and_augmentation.html)章节。


## 定义网络

我们选择相对简单的LeNet网络。LeNet网络不包括输入层的情况下，共有7层：2个卷积层、2个下采样层（池化层）、3个全连接层。每层都包含不同数量的训练参数，如下图所示：

![LeNet-5](./images/LeNet_5.jpg)
  
> 更多的LeNet网络的介绍不在此赘述，希望详细了解LeNet网络，可以查询<http://yann.lecun.com/exdb/lenet/>。

我们需要对全连接层以及卷积层进行初始化。 

`TruncatedNormal`：参数初始化方法，MindSpore支持`TruncatedNormal`、`Normal`、`Uniform`等多种参数初始化方法，具体可以参考MindSpore API的`mindspore.common.initializer`模块说明。

初始化示例代码如下：

```python
import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal

def weight_variable():
    """
    weight initial
    """
    return TruncatedNormal(0.02)

def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """
    conv layer weight initial
    """
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")

def fc_with_initialize(input_channels, out_channels):
    """
    fc layer weight initial
    """
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)
```

使用MindSpore定义神经网络需要继承`mindspore.nn.cell.Cell`。`Cell`是所有神经网络（`Conv2d`等）的基类。

神经网络的各层需要预先在`__init__`方法中定义，然后通过定义`construct`方法来完成神经网络的前向构造。按照LeNet的网络结构，定义网络各层如下：

```python
class LeNet5(nn.Cell):
    """
    Lenet network structure
    """
    #define the operator required
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    #use the preceding operators to construct networks
    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
```

## 定义损失函数及优化器

### 基本概念

在进行定义之前，先简单介绍损失函数及优化器的概念。

- 损失函数：又叫目标函数，用于衡量预测值与实际值差异的程度。深度学习通过不停地迭代来缩小损失函数的值。定义一个好的损失函数，可以有效提高模型的性能。
- 优化器：用于最小化损失函数，从而在训练过程中改进模型。 

定义了损失函数后，可以得到损失函数关于权重的梯度。梯度用于指示优化器优化权重的方向，以提高模型性能。

### 定义损失函数

MindSpore支持的损失函数有`SoftmaxCrossEntropyWithLogits`、`L1Loss`、`MSELoss`等。这里使用`SoftmaxCrossEntropyWithLogits`损失函数。

```python
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
```

在`__main__`函数中调用定义好的损失函数：

```python
if __name__ == "__main__":
    ...
    #define the loss function
    net_loss = SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction='mean')
    ...
```

### 定义优化器

MindSpore支持的优化器有`Adam`、`AdamWeightDecay`、`Momentum`等。

这里使用流行的`Momentum`优化器。

```python
if __name__ == "__main__":
    ...
    #learning rate setting
    lr = 0.01
    momentum = 0.9
    #create the network
    network = LeNet5()
    #define the optimizer
    net_opt = nn.Momentum(network.trainable_params(), lr, momentum)
    ...
```

## 训练网络

### 配置模型保存

MindSpore提供了callback机制，可以在训练过程中执行自定义逻辑，这里使用框架提供的`ModelCheckpoint`为例。
`ModelCheckpoint`可以保存网络模型和参数，以便进行后续的fine-tuning（微调）操作。

```python
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

if __name__ == "__main__":
    ...
    # set parameters of check point
    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10) 
    # apply parameters of check point
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck) 
    ...
```

### 配置训练网络

通过MindSpore提供的`model.train`接口可以方便地进行网络的训练。`LossMonitor`可以监控训练过程中`loss`值的变化。
这里把`epoch_size`设置为1，对数据集进行1个迭代的训练。


```python
from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore.train import Model

...
def train_net(args, model, epoch_size, mnist_path, repeat_size, ckpoint_cb, sink_mode):
    """define the training method"""
    print("============== Starting Training ==============")
    #load training dataset
    ds_train = create_dataset(os.path.join(mnist_path, "train"), 32, repeat_size)
    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor()], dataset_sink_mode=sink_mode)
...

if __name__ == "__main__":
    ...
    
    epoch_size = 1    
    mnist_path = "./MNIST_Data"
    repeat_size = 1
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    train_net(args, model, epoch_size, mnist_path, repeat_size, ckpoint_cb, dataset_sink_mode)
    ...
```
其中，
在`train_net`方法中，我们加载了之前下载的训练数据集，`mnist_path`是MNIST数据集路径。

## 运行并查看结果

使用以下命令运行脚本：
```
python lenet.py --device_target=CPU
```
其中，  
`lenet.py`：为你根据教程编写的脚本文件。  
`--device_target CPU`：指定运行硬件平台，参数为`CPU`、`GPU`或者`Ascend`，根据你的实际运行硬件平台来指定。

训练过程中会打印loss值，类似下图。loss值会波动，但总体来说loss值会逐步减小，精度逐步提高。每个人运行的loss值有一定随机性，不一定完全相同。
训练过程中loss打印示例如下：

```bash
...
epoch: 1 step: 262, loss is 1.9212162
epoch: 1 step: 263, loss is 1.8498616
epoch: 1 step: 264, loss is 1.7990671
epoch: 1 step: 265, loss is 1.9492403
epoch: 1 step: 266, loss is 2.0305142
epoch: 1 step: 267, loss is 2.0657792
epoch: 1 step: 268, loss is 1.9582214
epoch: 1 step: 269, loss is 0.9459006
epoch: 1 step: 270, loss is 0.8167224
epoch: 1 step: 271, loss is 0.7432692
...
```

训练完后，即保存的模型文件，示例如下:

```bash
checkpoint_lenet-1_1875.ckpt
```

其中，  
`checkpoint_lenet-1_1875.ckpt`：指保存的模型参数文件。名称具体含义checkpoint_*网络名称*-*第几个epoch*_*第几个step*.ckpt。

## 验证模型

在得到模型文件后，通过模型运行测试数据集得到的结果，验证模型的泛化能力。

1. 使用`model.eval`接口读入测试数据集。
2. 使用保存后的模型参数进行推理。

```python
from mindspore.train.serialization import load_checkpoint, load_param_into_net

...
def test_net(args,network,model,mnist_path):
    """define the evaluation method"""
    print("============== Starting Testing ==============")
    #load the saved model for evaluation
    param_dict = load_checkpoint("checkpoint_lenet-1_1875.ckpt")
    #load parameter to the network
    load_param_into_net(network, param_dict)
    #load testing dataset
    ds_eval = create_dataset(os.path.join(mnist_path, "test"))
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("============== Accuracy:{} ==============".format(acc))

if __name__ == "__main__":
    ...
    test_net(args, network, model, mnist_path)
```

其中，   
`load_checkpoint`：通过该接口加载CheckPoint模型参数文件，返回一个参数字典。  
`checkpoint_lenet-1_1875.ckpt`：之前保存的CheckPoint模型文件名称。  
`load_param_into_net`：通过该接口把参数加载到网络中。


使用运行命令，运行你的代码脚本。
```bash
python lenet.py --device_target=CPU
```
其中，  
`lenet.py`：为你根据教程编写的脚本文件。  
`--device_target CPU`：指定运行硬件平台，参数为`CPU`、`GPU`或者`Ascend`，根据你的实际运行硬件平台来指定。

运行结果示例如下：

```
...
============== Starting Testing ==============
============== Accuracy:{'Accuracy': 0.9742588141025641} ==============
```

可以在打印信息中看出模型精度数据，示例中精度数据达到97.4%，模型质量良好。
