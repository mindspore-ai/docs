# Cell

<!-- TOC -->

- [Cell](#cell)
    - [概念用途](#概念用途)
    - [关键成员函数](#关键成员函数)
        - [construct方法](#construct方法)
        - [parameters_dict](#parameters_dict)
        - [cells_and_names](#cells_and_names)
    - [模型层](#模型层)
        - [内置模型层](#内置模型层)
        - [应用实例](#应用实例)
    - [损失函数](#损失函数)
        - [内置损失函数](#内置损失函数)
        - [应用实例](#应用实例-1)
    - [构建自定义网络](#构建自定义网络)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/api/source_zh_cn/programming_guide/cell.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概念用途

MindSpore的`Cell`类是构建所有网络的基类，也是网络的基本单元。当用户需要自定义网络时，需要继承Cell类，并重写`__init_`方法和`contruct`方法。

损失函数、优化器和模型层等本质上也属于网络结构，也需要继承`Cell`类才能实现功能，同样用户也可以根据业务需求自定义这部分内容。

本节内容首先将会介绍`Cell`类的关键成员函数，然后介绍基于`Cell`实现的MindSpore内置损失函数、优化器和模型层及使用方法，最后通过实例介绍如何利用`Cell`类构建自定义网络。

## 关键成员函数

### construct方法

`Cell`类重写了`__call__`方法，在`Cell`类的实例被调用时，会执行`contruct`方法。网络结构在`contruct`方法里面定义。

下面的样例中，我们构建了一个简单的网络。用例的网络结构为Conv2d->BatchNorm2d->ReLU->Flatten->Dense。
在`construct`方法中，`x`为输入数据，`out`是经过网络的每层计算后得到的计算结果。

```
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

### parameters_dict

`parameters_dict`方法识别出网络结构中所有的参数，返回一个以key为参数名，value为参数值的`OrderedDict`。

`Cell`类中返回参数的方法还有许多，例如`get_parameters`、`trainable_params`等，具体使用方法可以参见[API文档](https://www.mindspore.cn/api/zh-CN/master/api/python/mindspore/mindspore.nn.html#mindspore.nn.Cell)。

代码样例如下：

```
net = Net()
result = net.parameters_dict()
print(result.keys())
print(result['conv.weight'])
```

样例中的`Net`采用上文构造网络的用例，打印了网络中所有参数的名字和`conv.weight`参数的结果。

输出如下：
```
odict_keys(['conv.weight', 'bn.moving_mean', 'bn.moving_variance', 'bn.gamma', 'bn.beta', 'fc.weight', 'fc.bias'])
Parameter (name=conv.weight, value=[[[[ 1.07402597e-02  7.70052336e-03  5.55867562e-03]
   [-3.21971579e-03 -3.75304517e-04 -8.73021083e-04]
...
[-1.81201510e-02 -1.31190736e-02 -4.27651079e-03]]]])
```

### cells_and_names

`cells_and_names`方法是一个迭代器，返回网络中每个`Cell`的名字和它的内容本身。

用例简单实现了获取与打印每个`Cell`名字的功能，其中根据上文网络结构可知，存在五个`Cell`分别是'conv'、'bn'、'relu'、'flatten'和'fc'。

代码样例如下：
```
net = Net()
names = []
for m in net.cells_and_names():
    names.append(m[0]) if m[0] else None
print(names)
```

输出如下：
```
['conv', 'bn', 'relu', 'flatten', 'fc']
```

## 模型层

在讲述了`Cell`的使用方法后可知，MindSpore能够以`Cell`为基类构造网络结构。

为了方便用户的使用，MindSpore框架内置了大量的模型层，用户可以通过接口直接调用。

同样，用户也可以自定义模型，此内容在“构建自定义网络”中介绍。

### 内置模型层

MindSpore框架在nn的layer层内置了丰富的接口，主要内容如下：

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

  数据计算层提供一些算子拼接而成的计算功能，例如数据生成和一些数学计算等。
  
  主要接口有`ReduceLogSumExp`、`Range`、`LinSpace`、`LGamma`等。

- 图片层

  图片计算层提供了一些矩阵计算相关的功能，将图片数据进行一些变换与计算。
  
  主要接口有`ImageGradients`、`SSIM`、`MSSSIM`、`PSNR`、`CentralCrop`等。

- 量化层

  量化是指将数据从float的形式转换成一段数据范围的int类型，所以量化层提供了一些数据量化的方法和模型层结构封装。
  
  主要接口有`Conv2dBnAct`、`DenseBnAct`、`Conv2dBnFoldQuant`、`LeakyReLUQuant`等。
  
### 应用实例

MindSpore的模型层在mindspore.nn下，使用方法如下所示：

```
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

目前MindSpore主要支持的损失函数有`L1Loss`、`MSELoss`、`SmoothL1Loss`、`SoftmaxCrossEntropyWithLogits`、`SoftmaxCrossEntropyExpand`和`CosineEmbeddingLoss`。

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

  交叉熵损失函数，用于分类模型。当标签数据不是one-hot编码形式时，需要输入参数`sparse`为True。`reduction`参数同`L1Loss`。
   
- SoftmaxCrossEntropyExpand

  交叉熵扩展损失函数，用于分类模型。当标签数据不是one-hot编码形式时，需要输入参数`sparse`为True。

- CosineEmbeddingLoss

  CosineEmbeddingLoss用于衡量两个输入相似程度，用于分类模型。`margin`默认为0.0，`reduction`参数同`L1Loss`。
  
### 应用实例

MindSpore的损失函数全部在mindspore.nn下，使用方法如下所示：

```
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor

loss = nn.L1Loss()
input_data = Tensor(np.array([[1, 2, 3], [2, 3, 4]]).astype(np.float32))
target_data = Tensor(np.array([[0, 2, 5], [3, 1, 1]]).astype(np.float32))
print(loss(input_data, target_data))
```

此用例构造了两个Tensor数据，利用`nn.L1Loss`接口定义了L1Loss，将`input_data`和`target_data`传入loss，执行L1Loss的计算，结果为1.5。若`loss = nn.L1Loss(reduction='sum')`，则结果为9.0。若`loss = nn.L1Loss(reduction='none')`，结果为[[1. 0. 2.] [1. 2. 3.]]。


## 构建自定义网络

无论是网络结构，还是前文提到的模型层、损失函数和优化器等，本质上都是一个Cell，因此都可以自定义实现。

首先构造一个继承`Cell`的子类，然后在`__init__`方法里面定义算子和模型层等，在`construct`方法里面构造网络结构。

以LeNet网络为例，在`__init__`方法中定义了卷积层，池化层和全连接层等结构单元，然后在`construct`方法将定义的内容连接在一起，形成一个完整LeNet的网络结构。

LeNet网络实现方式如下所示：
```
import mindspore.nn as nn

class LeNet5(nn.Cell):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, pad_mode="valid")
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
