# Cell构建及其子类

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/docs/programming_guide/source_zh_cn/cell.md)

## 概述

MindSpore的`Cell`类是构建所有网络的基类，也是网络的基本单元。当用户需要自定义网络时，需要继承`Cell`类，并重写`__init__`方法和`contruct`方法。

损失函数、优化器和模型层等本质上也属于网络结构，也需要继承`Cell`类才能实现功能，同样用户也可以根据业务需求自定义这部分内容。

本节内容首先将会介绍`Cell`类的关键成员函数，然后介绍基于`Cell`实现的MindSpore内置损失函数、优化器和模型层及使用方法，最后通过实例介绍如何利用`Cell`类构建自定义网络。

## 关键成员函数

### construct方法

`Cell`类重写了`__call__`方法，在`Cell`类的实例被调用时，会执行`construct`方法。网络结构在`construct`方法里面定义。

下面的样例中，我们构建了一个简单的网络实现卷积计算功能。构成网络的算子在`__init__`中定义，在`construct`方法里面使用，用例的网络结构为`Conv2d`->`BiasAdd`。

在`construct`方法中，`x`为输入数据，`output`是经过网络结构计算后得到的计算结果。

```python
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer

class Net(nn.Cell):
    def __init__(self, in_channels=10, out_channels=20, kernel_size=3):
        super(Net, self).__init__()
        self.conv2d = ops.Conv2D(out_channels, kernel_size)
        self.bias_add = ops.BiasAdd()
        self.weight = Parameter(
            initializer('normal', [out_channels, in_channels, kernel_size, kernel_size]),
            name='conv.weight')

    def construct(self, x):
        output = self.conv2d(x, self.weight)
        output = self.bias_add(output, self.bias)
        return output
```

### parameters_dict

`parameters_dict`方法识别出网络结构中所有的参数，返回一个以key为参数名，value为参数值的`OrderedDict`。

`Cell`类中返回参数的方法还有许多，例如`get_parameters`、`trainable_params`等，具体使用方法可以参见[API文档](https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.nn.html#mindspore.nn.Cell)。

代码样例如下：

```python
net = Net()
result = net.parameters_dict()
print(result.keys())
print(result['conv.weight'])
```

样例中的`Net`采用上文构造网络的用例，打印了网络中所有参数的名字和`conv.weight`参数的结果。

输出如下：

```text
odict_keys(['conv.weight'])
Parameter (name=conv.weight, value=[[[[-3.95042636e-03  1.08830128e-02 -6.51786150e-03]
   [ 8.66129529e-03  7.36288540e-03 -4.32638079e-03]
   [-1.47628486e-02  8.24100431e-03 -2.71035335e-03]]
   ......
   [ 1.58852488e-02 -1.03505487e-02  1.72988791e-02]]]])
```

### cells_and_names

`cells_and_names`方法是一个迭代器，返回网络中每个`Cell`的名字和它的内容本身。

用例简单实现了获取与打印每个`Cell`名字的功能，其中根据网络结构可知，存在1个`Cell`为`nn.Conv2d`。

其中`nn.Conv2d`是MindSpore以`Cell`为基类封装好的一个卷积层，其具体内容将在“模型层”中进行介绍。

代码样例如下：

```python
import mindspore.nn as nn

class Net1(nn.Cell):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal')

    def construct(self, x):
        out = self.conv(x)
        return out

net = Net1()
names = []
for m in net.cells_and_names():
    print(m)
    names.append(m[0]) if m[0] else None
print('-------names-------')
print(names)
```

输出如下：

```text
('', Net1<
  (conv): Conv2d<input_channels=3, output_channels=64, kernel_size=(3, 3),stride=(1, 1),  pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False,weight_init=normal, bias_init=zeros>
  >)
('conv', Conv2d<input_channels=3, output_channels=64, kernel_size=(3, 3),stride=(1, 1),  pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False,weight_init=normal, bias_init=zeros>)
-------names-------
['conv']
```

### set_grad

`set_grad`接口功能是使用户构建反向网络，在不传入参数调用时，默认设置`requires_grad`为True，需要在计算网络反向的场景中使用。

以`TrainOneStepCell`为例，其接口功能是使网络进行单步训练，需要计算网络反向，因此初始化方法里需要使用`set_grad`。

`TrainOneStepCell`部分代码如下：

```python
class TrainOneStepCell(Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        ......
```

如果用户使用`TrainOneStepCell`等类似接口无需使用`set_grad`， 内部已封装实现。

若用户需要自定义此类训练功能的接口，需要在其内部调用，或者在外部设置`network.set_grad`。

## nn模块与ops模块的关系

MindSpore的nn模块是Python实现的模型组件，是对低阶API的封装，主要包括各种模型层、损失函数、优化器等。

同时nn也提供了部分与`Primitive`算子同名的接口，主要作用是对`Primitive`算子进行进一步封装，为用户提供更友好的API。

重新分析上文介绍`construct`方法的用例，此用例是MindSpore的`nn.Conv2d`源码简化内容，内部会调用`ops.Conv2D`。`nn.Conv2d`卷积API增加输入参数校验功能并判断是否`bias`等，是一个高级封装的模型层。

```python
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer

class Net(nn.Cell):
    def __init__(self, in_channels=10, out_channels=20, kernel_size=3):
        super(Net, self).__init__()
        self.conv2d = ops.Conv2D(out_channels, kernel_size)
        self.bias_add = ops.BiasAdd()
        self.weight = Parameter(
            initializer('normal', [out_channels, in_channels, kernel_size, kernel_size]),
            name='conv.weight')

    def construct(self, x):
        output = self.conv2d(x, self.weight)
        output = self.bias_add(output, self.bias)
        return output
```

## 模型层

在讲述了`Cell`的使用方法后可知，MindSpore能够以`Cell`为基类构造网络结构。

为了方便用户的使用，MindSpore框架内置了大量的模型层，用户可以通过接口直接调用。

同样，用户也可以自定义模型，此内容在“构建自定义网络”中介绍。

### 内置模型层

MindSpore框架在`mindspore.nn`的layer层内置了丰富的接口，主要内容如下：

- 激活层

  激活层内置了大量的激活函数，在定义网络结构中经常使用。激活函数为网络加入了非线性运算，使得网络能够拟合效果更好。

  主要接口有`Softmax`、`Relu`、`Elu`、`Tanh`、`Sigmoid`等。

- 基础层

  基础层实现了网络中一些常用的基础结构，例如全连接层、Onehot编码、Dropout、平铺层等都在此部分实现。

  主要接口有`Dense`、`Flatten`、`Dropout`、`Norm`、`OneHot`等。

- 容器层

  容器层主要功能是实现一些存储多个Cell的数据结构。

  主要接口有`SequentialCell`、`CellList`等。

- 卷积层

  卷积层提供了一些卷积计算的功能，如普通卷积、深度卷积和卷积转置等。

  主要接口有`Conv2d`、`Conv1d`、`Conv2dTranspose`、`Conv1dTranspose`等。

- 池化层

  池化层提供了平均池化和最大池化等计算的功能。

  主要接口有`AvgPool2d`、`MaxPool2d`和`AvgPool1d`。

- 嵌入层

  嵌入层提供word embedding的计算功能，将输入的单词映射为稠密向量。

  主要接口有`Embedding`、`EmbeddingLookup`、`EmbeddingLookUpSplitMode`等。

- 长短记忆循环层

  长短记忆循环层提供LSTM计算功能。其中`LSTM`内部会调用`LSTMCell`接口，`LSTMCell`是一个LSTM单元，对一个LSTM层做运算，当涉及多LSTM网络层运算时，使用`LSTM`接口。

  主要接口有`LSTM`和`LSTMCell`。

- 标准化层

  标准化层提供了一些标准化的方法，即通过线性变换等方式将数据转换成均值和标准差。

  主要接口有`BatchNorm1d`、`BatchNorm2d`、`LayerNorm`、`GroupNorm`、`GlobalBatchNorm`等。

- 数学计算层

  数学计算层提供一些算子拼接而成的计算功能，例如数据生成和一些数学计算等。

  主要接口有`ReduceLogSumExp`、`Range`、`LinSpace`、`LGamma`等。

- 图片层

  图片计算层提供了一些矩阵计算相关的功能，将图片数据进行一些变换与计算。

  主要接口有`ImageGradients`、`SSIM`、`MSSSIM`、`PSNR`、`CentralCrop`等。

- 量化层

  量化是指将数据从float的形式转换成一段数据范围的int类型，所以量化层提供了一些数据量化的方法和模型层结构封装。

  主要接口有`Conv2dBnAct`、`DenseBnAct`、`Conv2dBnFoldQuant`、`LeakyReLUQuant`等。

### 应用实例

MindSpore的模型层在`mindspore.nn`下，使用方法如下所示：

```python
import mindspore.nn as nn

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal')
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(64 * 222 * 222, 3)

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        out = self.fc(x)
        return out
```

依然是上述网络构造的用例，从这个用例中可以看出，程序调用了`Conv2d`、`BatchNorm2d`、`ReLU`、`Flatten`和`Dense`模型层的接口。

在`Net`初始化方法里被定义，然后在`construct`方法里真正运行，这些模型层接口有序的连接，形成一个可执行的网络。

## 损失函数

目前MindSpore主要支持的损失函数有`L1Loss`、`MSELoss`、`SmoothL1Loss`、`SoftmaxCrossEntropyWithLogits`和`CosineEmbeddingLoss`。

MindSpore的损失函数全部是`Cell`的子类实现，所以也支持用户自定义损失函数，其构造方法在“构建自定义网络”中进行介绍。

### 内置损失函数

- L1Loss

  计算两个输入数据的绝对值误差，用于回归模型。`reduction`参数默认值为mean，返回loss平均值结果，若`reduction`值为sum，返回loss累加结果，若`reduction`值为none，返回每个loss的结果。

- MSELoss

  计算两个输入数据的平方误差，用于回归模型。`reduction`参数同`L1Loss`。

- SmoothL1Loss

  `SmoothL1Loss`为平滑L1损失函数，用于回归模型，阈值`sigma`默认参数为1。
`
- SoftmaxCrossEntropyWithLogits

  交叉熵损失函数，用于分类模型。当标签数据不是one-hot编码形式时，需要输入参数`sparse`为True。`reduction`参数默认值为none，其参数含义同`L1Loss`。

- CosineEmbeddingLoss

  `CosineEmbeddingLoss`用于衡量两个输入相似程度，用于分类模型。`margin`默认为0.0，`reduction`参数同`L1Loss`。

### 应用实例

MindSpore的损失函数全部在mindspore.nn下，使用方法如下所示：

```python
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor

loss = nn.L1Loss()
input_data = Tensor(np.array([[1, 2, 3], [2, 3, 4]]).astype(np.float32))
target_data = Tensor(np.array([[0, 2, 5], [3, 1, 1]]).astype(np.float32))
print(loss(input_data, target_data))
```

输出结果：

```python
1.5
```

此用例构造了两个Tensor数据，利用`nn.L1Loss`接口定义了loss，将`input_data`和`target_data`传入loss，执行L1Loss的计算，结果为1.5。若loss = nn.L1Loss(reduction='sum')，则结果为9.0。若loss = nn.L1Loss(reduction='none')，结果为[[1. 0. 2.] [1. 2. 3.]]。

## 优化算法

`mindspore.nn.optim`是MindSpore框架中实现各种优化算法的模块，详细说明参见[优化算法](https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.0/optim.html)。

## 构建自定义网络

无论是网络结构，还是前文提到的模型层、损失函数和优化器等，本质上都是一个`Cell`，因此都可以自定义实现。

首先构造一个继承`Cell`的子类，然后在`__init__`方法里面定义算子和模型层等，在`construct`方法里面构造网络结构。

以LeNet网络为例，在`__init__`方法中定义了卷积层，池化层和全连接层等结构单元，然后在`construct`方法将定义的内容连接在一起，形成一个完整LeNet的网络结构。

LeNet网络实现方式如下所示：

```python
import mindspore.nn as nn

class LeNet5(nn.Cell):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, pad_mode="valid")
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode="valid")
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 3)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
