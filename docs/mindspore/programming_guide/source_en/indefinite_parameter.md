# Parameter Passing

`Ascend` `GPU` `CPU` `Model Development`

Translator: [lamuxiaoyu](https://gitee.com/xiaoxinniuniu)

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/programming_guide/source_en/indefinite_parameter.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

## Overview

This article describes the use of variable parameters in network construction, indicates that a variable number of parameters can be used to construct a network. There are two ways to pass in parameters, one is passing in parameters of tuple type, the other is passing in Python variable parameters. The following two examples explain the use of these two construction methods.

## Passing in Parameters of Tuple Type

Construct a single Add operator network with two inputs, and a tuple parameter can be passed to replace the two inputs. The network structure is as follows:

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

In the definition of the AddModel network, inputs indicates a parameter of the tuple type and contains two elements. The call method is as follows:

```python
input_x = Tensor(np.ones((2, 3)), mindspore.float32)
input_y = Tensor(np.ones((2, 3)), mindspore.float32)
net = AddModel()

y = net((input_x, input_y))
print(y)
```

Result:

```text
[[2. 2. 2.]
 [2. 2. 2.]]
```

## Passing in Python Variable Parameters

The second way is to use Python's variable parameters(*parameters), the network is constructed as follows:

```python
class AddModel(Cell):
    def __init__(self):
        super().__init__()
        self.add = op.Add()

    def construct(self, *inputs):
        return self.add(inputs[0], inputs[1])
```

In the network definition, *inputs indicates variable parameters in Python, Position parameters may be collected during function definition to form a tuple object, or each parameters in tuple object may be unpacked during function call. There are two call methods:

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

Result:

```text
[[2. 2. 2.]
 [2. 2. 2.]]
```