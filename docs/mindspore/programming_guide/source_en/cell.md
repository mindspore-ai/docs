# Cell

`Ascend` `GPU` `CPU` `Beginner`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/cell.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

The `Cell` class of MindSpore is the base class for building all networks and the basic unit of a network. When you need to customize a network, you need to inherit the `Cell` class and override the `__init__` and `construct` methods.

Loss functions, optimizers, and model layers are parts of the network structure and can be implemented only by inheriting the `Cell` class. You can also customize them based on service requirements.

The following describes the key member functions of the `Cell` class, "Building a network" will introduce the built-in loss functions, optimizers, and model layers of MindSpore implemented based on the `Cell` class, and how to use them, as well as describes how to use the `Cell` class to build a customized network.

## Key Member Functions

### construct

The `Cell` class overrides the `__call__` method. When the `Cell` class instance is called, the `construct` method is executed. The network structure is defined in the `construct` method.

In the following example, a simple network is built to implement the convolution computing function. The operators in the network are defined in `__init__` and used in the `construct` method. The network structure of the case is as follows: `Conv2d` -> `BiasAdd`.

In the `construct` method, `x` is the input data, and `output` is the result obtained after the network structure computation.

```python
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter
from mindspore.common.initializer import initializer

class Net(nn.Cell):
    def __init__(self, in_channels=10, out_channels=20, kernel_size=3):
        super(Net, self).__init__()
        self.conv2d = ops.Conv2D(out_channels, kernel_size)
        self.bias_add = ops.BiasAdd()
        self.weight = Parameter(initializer('normal', [out_channels, in_channels, kernel_size, kernel_size]), name='conv.weight')
        self.bias = Parameter(initializer('normal', [out_channels]), name='conv.bias')

    def construct(self, x):
        output = self.conv2d(x, self.weight)
        output = self.bias_add(output, self.bias)
        return output
```

### parameters_dict

The `parameters_dict` method is used to identify all parameters in the network structure and return `OrderedDict` with key as the parameter name and value as the parameter value.

There are many other methods for returning parameters in the `Cell` class, such as `get_parameters` and `trainable_params`. For details, see [mindspore API](https://www.mindspore.cn/docs/api/en/master/api_python/nn/mindspore.nn.Cell.html).

A code example is as follows:

```python
net = Net()
result = net.parameters_dict()
print(result.keys())
print(result['conv.weight'])
```

The following information is displayed:

```text
odict_keys(['conv.weight', 'conv.bias'])
Parameter (name=conv.weight, shape=(20, 10, 3, 3), dtype=Float32, requires_grad=True)
```

In the example, `Net` uses the preceding network building case to print names of all parameters on the network and the result of the `weight` parameter.

### cells_and_names

The `cells_and_names` method is an iterator that returns the name and content of each `Cell` on the network.

The case simply implements the function of obtaining and printing the name of each `Cell`. According to the network structure, there is a `Cell` whose name is `nn.Conv2d`.

`nn.Conv2d` is a convolutional layer encapsulated by MindSpore using `Cell` as the base class. For details, see "Model Layers".

A code example is as follows:

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

```text
('', Net1<
  (conv): Conv2d<input_channels=3, output_channels=64, kernel_size=(3, 3),stride=(1, 1),  pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False,weight_init=normal, bias_init=zeros, format=NCHW>
  >)
('conv', Conv2d<input_channels=3, output_channels=64, kernel_size=(3, 3),stride=(1, 1),  pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False,weight_init=normal, bias_init=zeros, format=NCHW>)
-------names-------
['conv']
```

### set_grad

The `set_grad` API is used to specify whether the network requires gradient. If no parameter is transferred for calling the API, the default value of `requires_grad` is True, and the backward network needed to compute the gradients will be generated when the forward network is executed.

Take `TrainOneStepCell` as an example. Its API function is to perform single-step training on the network. The backward network needs to be computed. Therefore, `set_grad` needs to be used in the initialization method.

A part of the `TrainOneStepCell` code is as follows:

```python
class TrainOneStepCell(Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        ......
```

If using similar APIs such as `TrainOneStepCell` and `GradOperation`, you do not need to use `set_grad`. The internal encapsulation is implemented.

If you need to customize APIs of this training function, call APIs internally or set `network.set_grad` externally.

### set_train

The `set_train` interface recursively configures the training attributes of the current `Cell` and all sub-`Cell`. When called without parameters, the default training attribute is set to True.

When implementing networks with different training and inference structures, the training and inference scenarios can be distinguished by the `training` attribute, and the execution logic of the network can be switched by combining with `set_train` when the network is running.

For example, part of the code of `nn.Dropout` is as follows:

```python
class Dropout(Cell):
    def __init__(self, keep_prob=0.5, dtype=mstype.float32):
        """Initialize Dropout."""
        super(Dropout, self).__init__()
        self.dropout = ops.Dropout(keep_prob, seed0, seed1)
        ......

    def construct(self, x):
        if not self.training:
            return x

        if self.keep_prob == 1:
            return x

        out, _ = self.dropout(x)
        return out
```

In `nn.Dropout`, two execution logics are distinguished according to the training attribute of `Cell`. When training is False, the input is returned directly, and when training is True, the `Dropout` operator is executed. Therefore, when defining the network, you need to set the execution mode of the network according to the training and inference scenarios. Take `nn.Dropout` as an example:

```python
import mindspore.nn as nn
net = nn.Dropout()
# execute training
net.set_train()
# execute inference
net.set_train(False)
```

### to_float

The `to_float` interface recursively configures the coercion type of the current `Cell` and all sub-`Cell` so that the current network structure runs with a specific float type. Usually used in mixed precision scenes.

For details of `to_float` and mixed precision, please refer to [Enabling Mixed Precision](https://www.mindspore.cn/docs/programming_guide/en/master/enable_mixed_precision.html).

## Relationship Between the nn Module and the ops Module

The nn module of MindSpore is a model component implemented by Python. It encapsulates low-level APIs, including various model layers, loss functions, and optimizers.

In addition, nn provides some APIs with the same name as the `Primitive` operator to further encapsulate the `Primitive` operator and provide more friendly APIs.

Reanalyze the case of the `construct` method described above. This case is the simplified content of the `nn.Conv2d` source code of MindSpore, and `ops.Conv2D` is internally called. The `nn.Conv2d` convolution API adds the input parameter validation function and determines whether `bias` is used. It is an advanced encapsulated model layer.

```python
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter
from mindspore.common.initializer import initializer

class Net(nn.Cell):
    def __init__(self, in_channels=10, out_channels=20, kernel_size=3):
        super(Net, self).__init__()
        self.conv2d = ops.Conv2D(out_channels, kernel_size)
        self.bias_add = ops.BiasAdd()
        self.weight = Parameter(initializer('normal', [out_channels, in_channels, kernel_size, kernel_size]))

    def construct(self, x):
        output = self.conv2d(x, self.weight)
        output = self.bias_add(output, self.bias)
        return output
```
