# 参数传递

<!-- TOC -->

- [参数传递](#参数传递)
    - [概述](#概述)
    - [传入tuple类型的参数](#传入tuple类型的参数)
    - [传入Python的可变参数](#传入Python的可变参数)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/indefinite_parameter.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

本文介绍不定参数在网络构建中的使用，指的是在构建网络时，可以使用不定个数的参数来构造，有两种构造方式，一种是直接传入一个tuple类型的参数，另一种是传入Python的可变参数(*参数)来构造。下面以两个例子来说明这两种构造方式的用法。

## 传入tuple类型的参数

构造一个单Add算子网络，该网络需要有两个输入，可以传入一个tuple类型的参数来代替两个输入。网络构造如下：

```python
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.ops as op


class AddModel(Cell):
    def __init__(self):
        super().__init__()
        self.add = op.Add()

    def construct(self, inputs):
        return self.add(inputs[0], inputs[1])
```

AddModel网络的定义中，inputs表示的是一个tuple类型的参数，包含两个元素。调用方法如下：

```python
input_x = Tensor(np.ones((2, 3)), mindspore.float32)
input_y = Tensor(np.ones((2, 3)), mindspore.float32)
net = AddModel()

y = net((input_x, input_y))
print(y)
```

运行结果:

```text
[[2. 2. 2.]
 [2. 2. 2.]]
```

## 传入Python的可变参数

第二种用法是使用Python的可变参数(*参数)，网络构造如下：

```python
class AddModel(Cell):
    def __init__(self):
        super().__init__()
        self.add = op.Add()

    def construct(self, *inputs):
        return self.add(inputs[0], inputs[1])
```

第二种用法，网络定义中，*inputs表示的是Python中的可变参数，可以在函数定义时收集位置参数组成一个tuple对象，也可以在函数调用时解包tuple对象中的每个参数，调用方法有两种，如下：

```python
input_x = Tensor(np.ones((2, 3)), mindspore.float32)
input_y = Tensor(np.ones((2, 3)), mindspore.float32)
net = AddModel()

#1) The first call method
y = net(input_x, input_y)

#2) The second call method
inputs = (input_x, input_y)
y = net(*inputs)

print(y)
```

运行结果:

```text
[[2. 2. 2.]
 [2. 2. 2.]]
```