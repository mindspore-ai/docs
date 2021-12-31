# 构建单算子网络和多层网络

`Ascend` `GPU` `CPU` `模型开发`

<!-- TOC -->

- [网络构建](#网络构建)
    - [概述](#概述)
    - [运行基础算子](#运行基础算子)
    - [使用Cell构建和执行网络](#使用Cell构建和执行网络)
        - [Cell的基础使用](#ell的基础使用])
        - [基础算子的nn封装](#基础算子的nn封装)
        - [CellList和SequentialCell](#CellList和SequentialCell)

<!-- TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/programming_guide/source_zh_cn/build_net.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## 概述

MindSpore的Cell类是构建所有网络的基类，也是网络的基本单元。定义网络时，可以继承Cell类，并重写`__init__`方法和`construct`方法。MindSpore的ops模块提供了基础算子的实现，nn模块实现了对基础算子的进一步封装，用户可以根据需要，灵活使用不同的算子。

Cell本身具备模块管理能力，一个Cell可以由多个Cell组成，便于组成更复杂的网络。同时，为了更好地构建和管理复杂的网络，`mindspore.nn`提供了容器对网络中的子模块或模型层进行管理，分为CellList和SequentialCell两种方式。

## 运行基础算子

网络的构建离不开基础算子的使用。operations模块是MindSpore的基础运算单元，封装了不同类型的算子，例如：

- array_ops: 数组相关的算子

- math_ops: 数学相关的算子

- nn_ops: 网络类算子

> 更多算子使用方式参考文档[算子](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.5/operators.html)。

直接运行两个基础算子，`ops.Mul()`和`ops.Add()`：

```python
import numpy as np
import mindspore
from mindspore import Tensor, ops

x = Tensor(np.array([1, 2, 3]), mindspore.float32)
y = Tensor(np.array([2, 2, 2]), mindspore.float32)

mul = ops.Mul()
add = ops.Add()
output = add(mul(x, x), y)
print(output)
```

运行结果如下：

```text
[3. 6. 11.]
```

## 使用Cell构建和执行网络

### Cell的基础使用

MindSpore提供了Cell类来方便用户定义和执行自己的网络，用户通过继承nn.Cell，在`__init__`构造函数中申明各个层的定义，在`construct`中实现层之间的连接关系，完成神经网络的前向构造。需要注意的是，construct中存在一定的限制，无法使用第三方库的方法，一般使用MindSpore的Tensor和Cell实例。

使用简单的ops算子，组合一个Cell：

```python
import numpy as np
import mindspore
from mindspore import Parameter, ops, Tensor, nn

class Net(nn.Cell):

  def __init__(self):
    super(Net, self).__init__()
    self.mul = ops.Mul()
    self.add = ops.Add()
    self.weight = Parameter(Tensor(np.array([2, 2, 2]), mindspore.float32))

  def construct(self, x):
    return self.add(self.mul(x, x), self.weight)

net = Net()
input = Tensor(np.array([1, 2, 3]))
output = net(input)
print(output)
```

运行结果如下：

```text
[3. 6. 11.]
```

### 基础算子的nn封装

尽管ops模块提供的多样算子可以基本满足网络构建的诉求，但为了在复杂的深度网络中提供更方便易用的接口，MindSpore对复杂算子进行了进一步的nn层封装。nn模块包括了各种模型层、损失函数、优化器等，为用户的使用提供了便利。

基于nn提供的模型层，使用Cell构建一个网络：

```python
import numpy as np
import mindspore
from mindspore import Tensor, nn

class ConvBNReLU(nn.Cell):
    def __init__(self):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out

net = ConvBNReLU()
input = Tensor(np.ones([1, 3, 64, 32]), mindspore.float32)
output = net(input)
```

### CellList和SequentialCell

为了便于管理和组成更复杂的网络，`mindspore.nn`提供了容器对网络中的子模型块或模型层进行管理，有CellList和SequentialCell两种方式。

- mindspore.nn.CellList：存储Cell的列表，存储的Cell既可以是模型层，也可以是构建的网络子块。CellList支持append，extend，insert方法。在执行网络时，可以在construct方法里，使用for循环，运行输出结果。

- mindspore.nn.SequentialCell：顺序容器，支持子模块以list或OrderedDict格式作为输入。不同于CellList的是，SequentialCell类内部实现了construct方法，可以直接输出结果。

使用CellList定义并执行一个网络，依次包含一个之前定义的模型子块ConvBNReLU，一个Conv2d层，一个BatchNorm2d层，一个ReLU层：

```python
import numpy as np
import mindspore
from mindspore import Tensor, nn

class MyNet(nn.Cell):

    def __init__(self):
        super(MyNet, self).__init__()
        # 将上一步中定义的ConvBNReLU加入一个列表
        layers = [ConvBNReLU()]
        # 使用CellList对网络进行管理
        self.build_block = nn.CellList(layers)
        # 使用append方法添加Conv2d层和ReLU层
        self.build_block.append(nn.Conv2d(64, 4, 4))
        self.build_block.append(nn.ReLU())
        # 使用insert方法在Conv2d层和ReLU层中间插入BatchNorm2d
        self.build_block.insert(-1,  nn.BatchNorm2d(4))

    def construct(self, x):
        # for循环执行网络
        for layer in self.build_block:
            x = layer(x)
        return x

net = MyNet()
print(net)

input = Tensor(np.ones([1, 3, 64, 32]), mindspore.float32)
output = net(input)
print(output.shape)
```

输出结果如下：

```text
MyNet<
  (build_block): CellList<
    (0): ConvBNReLU<
      (conv): Conv2d<input_channels=3, output_channels=64, kernel_size=(3, 3),stride=(1, 1),  pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
      (bn): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=build_block.0.bn.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=build_block.0.bn.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=build_block.0.bn.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=build_block.0.bn.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
      (relu): ReLU<>
      >
    (1): Conv2d<input_channels=64, output_channels=4, kernel_size=(4, 4),stride=(1, 1),  pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
    (2): BatchNorm2d<num_features=4, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=build_block.2.gamma, shape=(4,), dtype=Float32, requires_grad=True), beta=Parameter (name=build_block.2.beta, shape=(4,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=build_block.2.moving_mean, shape=(4,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=build_block.2.moving_variance, shape=(4,), dtype=Float32, requires_grad=False)>
    (3): ReLU<>
    >
  >
(1, 4, 64, 32)
```

使用SequentialCell构建一个网络，输入为list，网络结构依次包含一个之前定义的模型子块ConvBNReLU，一个Conv2d层，一个BatchNorm2d层，一个ReLU层：

```python
import numpy as np
import mindspore
from mindspore import Tensor, nn

class MyNet(nn.Cell):

    def __init__(self):
        super(MyNet, self).__init__()
        # 将上一步中定义的ConvBNReLU加入一个列表
        layers = [ConvBNReLU()]
       # 在列表中添加模型层
        layers.extend([
          nn.Conv2d(64, 4, 4),
          nn.BatchNorm2d(4),
          nn.ReLU()
        ])
        # 使用SequentialCell对网络进行管理
        self.build_block = nn.SequentialCell(layers)

    def construct(self, x):
      return self.build_block(x)

net = MyNet()
print(net)

input = Tensor(np.ones([1, 3, 64, 32]), mindspore.float32)
output = net(input)
print(output.shape)
```

输出结果如下：

```text
MyNet<
  (build_block): SequentialCell<
    (0): ConvBNReLU<
      (conv): Conv2d<input_channels=3, output_channels=64, kernel_size=(3, 3),stride=(1, 1),  pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
      (bn): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=build_block.0.bn.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=build_block.0.bn.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=build_block.0.bn.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=build_block.0.bn.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
      (relu): ReLU<>
      >
    (1): Conv2d<input_channels=64, output_channels=4, kernel_size=(4, 4),stride=(1, 1),  pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
    (2): BatchNorm2d<num_features=4, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=build_block.2.gamma, shape=(4,), dtype=Float32, requires_grad=True), beta=Parameter (name=build_block.2.beta, shape=(4,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=build_block.2.moving_mean, shape=(4,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=build_block.2.moving_variance, shape=(4,), dtype=Float32, requires_grad=False)>
    (3): ReLU<>
    >
  >
(1, 4, 64, 32)
```

SequentialCell也支持输入为OrderdDict类型：

```python
import numpy as np
import mindspore
from mindspore import Tensor, nn
from collections import OrderedDict

class MyNet(nn.Cell):

    def __init__(self):
        super(MyNet, self).__init__()
        layers = OrderedDict()
        # 将cells加入字典
        layers["ConvBNReLU"] = ConvBNReLU()
        layers["conv"] = nn.Conv2d(64, 4, 4)
        layers["norm"] = nn.BatchNorm2d(4)
        layers["relu"] = nn.ReLU()
        # 使用SequentialCell对网络进行管理
        self.build_block = nn.SequentialCell(layers)

    def construct(self, x):
      return self.build_block(x)

net = MyNet()
print(net)

input = Tensor(np.ones([1, 3, 64, 32]), mindspore.float32)
output = net(input)
print(output.shape)
```

输出结果如下：

```text
MyNet<
  (build_block): SequentialCell<
    (ConvBNReLU): ConvBNReLU<
      (conv): Conv2d<input_channels=3, output_channels=64, kernel_size=(3, 3),stride=(1, 1),  pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
      (bn): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=build_block.ConvBNReLU.bn.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=build_block.ConvBNReLU.bn.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=build_block.ConvBNReLU.bn.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=build_block.ConvBNReLU.bn.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
      (relu): ReLU<>
      >
    (conv): Conv2d<input_channels=64, output_channels=4, kernel_size=(4, 4),stride=(1, 1),  pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
    (norm): BatchNorm2d<num_features=4, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=build_block.norm.gamma, shape=(4,), dtype=Float32, requires_grad=True), beta=Parameter (name=build_block.norm.beta, shape=(4,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=build_block.norm.moving_mean, shape=(4,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=build_block.norm.moving_variance, shape=(4,), dtype=Float32, requires_grad=False)>
    (relu): ReLU<>
  >
>
(1, 4, 64, 32)
```
