# contexpr

<!-- TOC -->

- [constexpr](#constexpr)
    - [Overview](#overview)

<!-- /TOC -->
<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/constexpr.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>
Overview

A @constexpr python decorator is provided in `mindspore.ops.constexpr`,which can be used to decorate a function.The function will be executed by the Python interpreter in the compiling stage, and finally this function will be changed to a ValueNode of ANF graph after the inference state of MindSpore.

Since the function is executed in the compiling state of MindSpore, so when using @constexpr to decorate a function,the parameter of this function must be a constant value that can be determined in compiling state of MindSpore.

Otherwise, if the parameter cannot be determined in compiling state of mindspore, the value of parameter will be none, that may make the return result of function does not match with expectations.

A code example is as follows:

```python
import numpy as np
from mindspore.ops import constexpr
from mindspore.ops import ops
import mindspore.nn as nn

@constexpr
def construct_tensor(x):
    return Tensor(np.array(x))

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = ops.ReLU()
    def construct(self):
        return self.relu(construct_tensor(x))

net = Net()
out = net()
print(out)

The following information is displayed:

```text

[1 2 0 4]