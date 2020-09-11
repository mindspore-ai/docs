# cell模块概述

<!-- TOC -->

- [cell模块概述](#cell模块概述)
    - [概念用途](#概念用途)
    - [关键成员函数](#关键成员函数)
    - [模型层](#模型层)
    - [损失函数](#损失函数)
    - [网络构造](#Cell构造自定义网络)

<!-- /TOC -->

## 概念用途

MindSpose的Cell类是构建所有网络的基类，也是网络的基本单元。当用户需要自定义网络时，需要继承Cell类，并重写__init_方法和contruct方法。

损失函数，优化器和模型层等本质上也属于网络结构，也需要继承Cell类才能实现功能，同样用户也可以根据业务需求自定义这部分内容。

本节内容首先将会介绍Cell类的关键成员函数，然后介绍基于Cell实现的MindSpore内置损失函数，优化器和模型层及使用方法，最后通过实例介绍
如何利用Cell类构建自定义网络。

## 关键成员函数

### construct方法

Cell类重写了__call__方法，在cell类的实例被调用时，会执行contruct方法。网络结构在contruct方法里面定义。

下面的样例中，我们构建了一个简单的网络。用例的网络结构为Conv2d->BatchNorm2d->ReLU->Flatten->Dense。
在construct方法中，x为输入数据， out是经过网络的每层计算后得到的计算结果。

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

parameters_dict方法识别出网络结构中所有的参数，返回一个以key为参数名，value为参数值的OrderedDict()。

Cell类中返回参数的方法还有许多，例如get_parameters()，trainable_params()等， 具体使用方法可以参见MindSpore API手册。

代码样例如下：

```
net = Net()
result = net.parameters_dict()
print(result.keys())
print(result['conv.weight'])
```

样例中的Net()采用上文构造网络的用例，打印了网络中是所有参数的名字和conv.weight参数的结果。

运行结果如下：
```
odict_keys(['conv.weight', 'bn.moving_mean', 'bn.moving_variance', 'bn.gamma', 'bn.beta', 'fc.weight', 'fc.bias'])
Parameter (name=conv.weight, value=[[[[ 1.07402597e-02  7.70052336e-03  5.55867562e-03]
   [-3.21971579e-03 -3.75304517e-04 -8.73021083e-04]
...
[-1.81201510e-02 -1.31190736e-02 -4.27651079e-03]]]])
```

### cells_and_names

cells_and_names方法是一个迭代器，返回网络中每个cell的名字和它的内容本身。

用例简单实现了网络的cell获取与打印每个cell名字的功能，其中根据上文网络结构可知，存在五个cell分别是'conv'，'bn'，'relu'，'flatten'，'fc'。

代码样例如下：
```
net = Net()
names = []
for m in net.cells_and_names():
    names.append(m[0]) if m[0] else None
print(names)
```

运行结果：
```
['conv', 'bn', 'relu', 'flatten', 'fc']
```
## 模型层

在讲述了Cell的使用方法后可知，MindSpore能够以Cell为基类构造网络结构。

为了方便业界需求及用户使用方便，MindSpore框架内置了大量的模型层，用户可以通过接口直接调用。

同样，用户也可以自定义模型，此内容在cell自定义构建中介绍。

### 内置模型层

MindSpore框架在nn的layer层内置了丰富的接口，主要内容如下：

- 激活层：

  激活层内置了大量的激活函数，在定义网络结构中经常使用。激活函数为网络加入了非线性运算，使得网络能够拟合效果更好。
  
  主要接口有Softmax，Relu，Elu，Tanh，Sigmoid等。

- 基础层：
  
  基础层实现了网络中一些常用的基础结构，例如全连接层，Onehot编码，Dropout，平铺层等都在此部分实现。
  
  主要接口有Dense，Flatten，Dropout，Norm，OneHot等。
  
- 容器层：

  容器层主要功能是实现一些存储多个cell的数据结构。
  
  主要接口有SequentialCell，CellList等。

- 卷积层：

  卷积层提供了一些卷积计算的功能，如普通卷积，深度卷积和卷积转置等。
  
  主要接口有Conv2d，Conv1d，Conv2dTranspose，DepthwiseConv2d，Conv1dTranspose等。

- 池化层：
  
  池化层提供了平均池化和最大池化等计算的功能。
  
  主要接口有AvgPool2d，MaxPool2d，AvgPool1d。

- 嵌入层：

  嵌入层提供word embedding的计算功能，将输入的单词映射为稠密向量。

  主要接口有：Embedding，EmbeddingLookup，EmbeddingLookUpSplitMode等。

- 长短记忆循环层：
  
  长短记忆循环层提供LSTM计算功能。其中LSTM内部会调用LSTMCell接口， LSTMCell是一个LSTM单元，
  对一个LSTM层做运算，当涉及多LSTM网络层运算时，使用LSTM接口。
  
  主要接口有：LSTM，LSTMCell。

- 标准化层：

  标准化层提供了一些标准化的方法，即通过线性变换等方式将数据转换成均值和标准差。
  
  主要接口有：BatchNorm1d，BatchNorm2d，LayerNorm，GroupNorm，GlobalBatchNorm等。

- 数学计算层：

  数据计算层提供一些算子拼接而成的计算功能，例如数据生成和一些数学计算等。
  
  主要接口有ReduceLogSumExp，Range，LinSpace，LGamma等。

- 图片层：

  图片计算层提供了一些矩阵计算相关的功能，将图片数据进行一些变换与计算。
  
  主要接口有ImageGradients，SSIM，MSSSIM，PSNR，CentralCrop等。

- 量化层：

  量化是指将数据从float的形式转换成一段数据范围的int类型，所以量化层提供了一些数据量化的方法和模型层结构封装。
  
  主要接口有Conv2dBnAct，DenseBnAct，Conv2dBnFoldQuant，LeakyReLUQuant等。
  
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

依然是上述网络构造的用例，从这个用例中可以看出，程序调用了Conv2d，BatchNorm2d，ReLU，Flatten和Dense模型层的接口。
在Net初始化方法里面被定义，然后在construct方法里面真正运行，这些模型层接口有序的连接，形成一个可执行的网络。

## 损失函数

目前MindSpore主要支持的损失函数有L1Loss，MSELoss，SmoothL1Loss，SoftmaxCrossEntropyWithLogits，SoftmaxCrossEntropyExpand
和CosineEmbeddingLoss。

MindSpore的损失函数全部是Cell的子类实现，所以也支持用户自定义损失函数，其构造方法在cell自定义构建中进行介绍。

### 内置损失函数

- L1Loss：

  计算两个输入数据的绝对值误差，用于回归模型。reduction参数默认值为mean，返回loss平均值结果，
若reduction值为sum，返回loss累加结果，若reduction值为none，返回每个loss的结果。

- MSELoss:

  计算两个输入数据的平方误差，用于回归模型。reduction参数默认值为mean，返回loss平均值结果，
若reduction值为sum，返回loss累加结果，若reduction值为none，返回每个loss的结果。

- SmoothL1Loss：

  SmoothL1Loss为平滑L1损失函数，用于回归模型，阈值sigma默认参数为1。

- SoftmaxCrossEntropyWithLogits：

  交叉熵损失函数，用于分类模型。当标签数据不是one-hot编码形式时，需要输入参数sparse为True。reduction参数
  与L1Loss一致。
   
- SoftmaxCrossEntropyExpand：

  交叉熵扩展损失函数，用于分类模型。当标签数据不是one-hot编码形式时，需要输入参数sparse为True。

- CosineEmbeddingLoss：

  CosineEmbeddingLoss用于衡量两个输入相似程度，用于分类模型。margin默认为0.0，reduction参数与L1Loss一致
  
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

此用例构造了两个Tensor数据，利用nn.L1Loss()接口定义了L1Loss，将input_data和target_data传入loss,
执行L1Loss的计算，结果为1.5。若loss = nn.L1Loss(reduction='sum')，则结果为9.0。
若loss = nn.L1Loss(reduction='none')，结果为[[1. 0. 2.] [1. 2. 3.]]


## Cell构造自定义网络

无论是网络结构，还是前文提到的模型层，损失函数和优化器等，本质上都是一个Cell，因此都可以自定义实现。

首先构造一个继承cell的子类，然后在__init__方法里面定义算子和模型层等，然后在construct方法里面构造网络结构。

以lenet5网络为例，在__init__方法中定义了卷积层，池化层和全连接层等结构单元，然后在construct方法将定义的内容连接在一起，
形成一个完整lenet5的网络结构。

lenet5网络实现方式如下所示：
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
