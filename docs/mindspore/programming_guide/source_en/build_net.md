# Constructing Single Operator Network and Multi-layer Network

Translator: [ChanJiatao](https://gitee.com/ChanJiatao)

`Linux` `Ascend` `GPU` `CPU` `Model Development` `Primary`

<!-- TOC -->

- [Constructing Single Operator Network and Multi-layer Network](#constructing-single-operator-network-and-multi-layer-network)
    - [Overview](#overview)
    - [Running Basic Operators](#running-basic-operators)
    - [Using Cell to Construct and Run Networks](#using-cell-to-construct-and-run-networks)
        - [The Basic Use of Cell](#the-basic-use-of-cell)
        - [The nn Encapsulation of Basic Operators](#the-nn-encapsulation-of-basic-operators)
        - [CellList and SequentialCell](#celllist-and-sequentialcell)

<!-- TOC -->

<a href="https://gitee.com/mindspore/docs/tree/r1.5/docs/mindspore/programming_guide/source_en/build_net.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## Overview

The Cell class of MindSpore is the base class for constructing all networks, which is also the base unit of networks. When defining a network, you can inherit the Cell class and override the `__init__` and `construct` methods. The ops module of MindSpore provides the realization of basic operators, and the nn module further encapsulates basic operators. Users can flexibly use different operators according to their needs.

A Cell has the module management capability. A Cell can be composed of multiple Cells to form a complex network. Meanwhile, in order to better build and manage complex networks, `mindspore.nn` provides a container to manage submodules or model layers in the network, including CellList and SequentialCell.

## Running Basic Operators

The construction of network is inseparable from the use of basic operators. The base operation unit of MindSpore is the operations module, which encapsulates different types of operators, such as:

- array_ops: The operators associated with arrays.

- math_ops: The operators associated with mathematics.

- nn_ops: The operators associated with networks.

> For more information about operator usage, please refer to the documentation [operator](https://www.mindspore.cn/docs/programming_guide/en/r1.5/operators.html).

Running two basic operators directly, `ops.Mul()` and `ops.Add()`:

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

The result is as follow:

```text
[3. 6. 11.]
```

## Using Cell to Construct and Run Networks

### The Basic Use of Cell

MindSpore provides Cell class to facilitate users to define and run their own networks. By inheriting nn.Cell, users can declare the definition of each layer in `__init__` constructor, and realize the connection relationship between layers in `construct` to complete the forward construction of neural networks. Note that there is a limit to use the third party libraries in construct, which generally uses the Tensor and Cell instances of MindSpore.

Using simple ops operators to combine a Cell:

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

The result is as follow:

```text
[3. 6. 11.]
```

### The nn Encapsulation of Basic Operators

Although various operators provided by ops module can basically meet the needs of network construction, in order to provide more convenient and usable interfaces in complex deep networks, MindSpore carries out further nn layer encapsulation of complex operators. The nn module includes various model layers, loss functions, optimizers and so on, which provides convenience for users.

Based on the model layer provided by nn, you can use Cell to build a network:

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

### CellList and SequentialCell

To facilitate the management and formation of more complex networks, `mindspore.nn` provides containers for the management of sub-model blocks or model layers in the network, including CellList and SequentialCell.

- mindspore.nn.CellList: A list of stored cells that can be either model layers or constructed network subblocks. CellList supports append, extend, insert methods. When executing the network, you can run the output using a for loop in the construct method.

- mindspore.nn.SequentialCell：A sequential container that supports submodules in list or OrderedDict type as input. Unlike CellList, the SequentialCell class implements the construct method internally, which directly outputs the results.

Using SequentialCell to define and construct a network, which in turn contains a previously defined model block ConvBNReLU, a Conv2d layer, a BatchNorm2d layer, and a ReLU layer:

```python
import numpy as np
import mindspore
from mindspore import Tensor, nn

class MyNet(nn.Cell):

    def __init__(self):
        super(MyNet, self).__init__()
        # Adding the ConvBNReLU defined in the previous step to a list.
        layers = [ConvBNReLU()]
        # Using CellList to manage the network.
        self.build_block = nn.CellList(layers)
        # Using the append method to add the Conv2d layer and ReLU layer.
        self.build_block.append(nn.Conv2d(64, 4, 4))
        self.build_block.append(nn.ReLU())
        # Using the insert method to insert BatchNorm2d between the Conv2d and ReLU layers.
        self.build_block.insert(-1,  nn.BatchNorm2d(4))

    def construct(self, x):
        # Running the network with a for loop.
        for layer in self.build_block:
            x = layer(x)
        return x

net = MyNet()
print(net)

input = Tensor(np.ones([1, 3, 64, 32]), mindspore.float32)
output = net(input)
print(output.shape)
```

The result is as follow:

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

Using SequentialCell to construct a network with input as list. The network structure in turn contains a previously defined model block ConvBNReLU, a Conv2d layer, a BatchNorm2d layer, and a ReLU layer:

```python
import numpy as np
import mindspore
from mindspore import Tensor, nn

class MyNet(nn.Cell):

    def __init__(self):
        super(MyNet, self).__init__()
        # Adding the ConvBNReLU defined in the previous step to a list.
        layers = [ConvBNReLU()]
        # Adding a model layer to the list.
        layers.extend([
          nn.Conv2d(64, 4, 4),
          nn.BatchNorm2d(4),
          nn.ReLU()
        ])
        # Using SequentialCell to manage the network.
        self.build_block = nn.SequentialCell(layers)

    def construct(self, x):
      return self.build_block(x)

net = MyNet()
print(net)

input = Tensor(np.ones([1, 3, 64, 32]), mindspore.float32)
output = net(input)
print(output.shape)
```

The result is as follow:

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

SequentialCell also supports the input as OrderdDict type:

```python
import numpy as np
import mindspore
from mindspore import Tensor, nn
from collections import OrderedDict

class MyNet(nn.Cell):

    def __init__(self):
        super(MyNet, self).__init__()
        layers = OrderedDict()
        # Adding cells to the dictionary.
        layers["ConvBNReLU"] = ConvBNReLU()
        layers["conv"] = nn.Conv2d(64, 4, 4)
        layers["norm"] = nn.BatchNorm2d(4)
        layers["relu"] = nn.ReLU()
        # Using SequentialCell to manage the network.
        self.build_block = nn.SequentialCell(layers)

    def construct(self, x):
      return self.build_block(x)

net = MyNet()
print(net)

input = Tensor(np.ones([1, 3, 64, 32]), mindspore.float32)
output = net(input)
print(output.shape)
```

The result is as follow:

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
