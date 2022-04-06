# Construct Constants In the Network

`Ascend` `GPU` `CPU` `Model Development`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/constexpr.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>
Overview

A @constexpr python decorator is provided in `mindspore.ops.constexpr`,which can be used to decorate a function.The
function will be executed by the Python interpreter in the compiling stage, and finally this function will be changed to
a ValueNode of ANF graph after the inference state of MindSpore.

Since the function is executed in the compiling state of MindSpore, so when using @constexpr to decorate a function,the
parameter of this function must be a constant value that can be determined in compiling state of MindSpore.

Otherwise, if the parameter cannot be determined in compiling state of mindspore, the value of parameter will be none,
that may make the return result of function does not match with expectations.

When the input parameter can be determined, you can use @constexpr to implement some operations that are not supported
in the construct function. For example, operations such as creating tensor based on a certain shape. You can use guard
clauses to check if the input arguments is None to avoid @constexpr's input arguments cannot be determined. A code
example is as follows:

```python
import numpy as np
from mindspore.ops import constexpr
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor


@constexpr
def construct_tensor(x):
    if x is None:
        raise ValueError("input is an unknown value")
    return Tensor(np.array(x, dtype=np.float32))


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

The following information is displayed:

```text
[7.0 6.0 3.0]
```

As shown below, if we change Net to a value that cannot be determined in MindSpore compiling state, an exception will be
thrown. Because the input of construct_tensor is a value that can be determined when ReLU must be run. ValueError will
be thrown in constexpr.

```python
@constexpr
def construct_tensor(x):
    if x is None:
        raise ValueError("input is an unknown value")
    return Tensor(np.array(x, dtype=np.float32))


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
