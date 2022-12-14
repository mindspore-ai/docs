# Construct Constants In the Network

<a href="https://gitee.com/mindspore/docs/blob/r1.10/tutorials/experts/source_en/network/constexpr.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.10/resource/_static/logo_source_en.png"></a>

A @constexpr python decorator is provided in `mindspore.ops.constexpr`, which can be used to decorate a function. The
function will be executed by the Python interpreter in the compiling stage, and finally will be collapsed by constants in the type derivation phase of MindSpore to become a ValueNode of the ANF graph.

Since the function is executed in the compiling state of MindSpore, when using @constexpr to decorate a function, the
parameter of this input function must be a constant value that can be determined in compiling state of MindSpore.

Otherwise, if the parameter is a value that cannot be determined in the compiling state, the parameter will be None, which may cause the function output to be different than expected.

When the input parameter can be determined, you can use @constexpr to implement some operations that are not supported in the construct function, for example, operations such as creating tensor based on a certain shape.

To avoid that @constexpr input is a value that cannot be determined in the compiling state, you can perform internal judgment processing on None to avoid some unknown errors.

A code example is as follows:

```python
import numpy as np
from mindspore.ops import constexpr
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor
import mindspore


@constexpr
def construct_tensor(x):
    if x is None:
        raise ValueError("input is an unknown value")
    return Tensor(np.array(x), dtype=mindspore.float32)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = ops.ReLU()

    def construct(self, x):
        return self.relu(construct_tensor(ops.shape(x)))


net = Net()
x = Tensor(np.random.random([7, 6, 3]))
out = net(x)
print(out)
```

```text
[7. 6. 3.]
```

As shown below, if we change Net to a value that cannot be determined in the compiling state, an exception will be thrown. Because the input of construct_tensor is a value that can be determined when ReLU is run. ValueError will be thrown in constexpr.

```python
@constexpr
def construct_tensor(x):
    if x is None:
        raise ValueError("input is an unknown value")
    return Tensor(np.array(x), dtype=mindspore.float32)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = ops.ReLU()

    def construct(self, x):
        return self.relu(construct_tensor(self.relu(x)))


net = Net()
x = Tensor(np.random.random([7, 6, 3]))
out = net(x)
print(out)
```

The following information is displayed:

```text
ValueError: input is an unknown value
```
