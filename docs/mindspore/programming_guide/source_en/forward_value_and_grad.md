# Forward Value And Grad

`Ascend` `GPU` `CPU` `Model Running`

<!-- TOC -->

- [Forward Value And Grad](#forward-value-and-grad)
    - [Overview](#overview)
    - [Examples](#first-order-derivation)
        - [Input Derivation](#input-derivation)
        - [Weight Derivation](#weight-derivation)
        - [Gradient Value Scaling](#gradient-value-scaling)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/forward_value_and_grad.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

ForwardValueAndGrad is used to generate the forward value and backend gradient of the input network. The `get_all`, `get_by_list`, and `sens_param` parameters are used to control the gradient calculation method. For details, see [mindspore API](https://www.mindspore.cn/docs/api/en/master/api_python/nn/mindspore.nn.ForwardValueAndGrad.html).

The following is an example of using ForwardValueAndGrad.

## First-order Derivation

The first-order derivative method of MindSpore is `mindspore.nn.ForwardValueAndGrad (network, weights=None, get_all=False, get_by_list=False, sens_param=False)`. When `get_all` is set to `False`, the first input derivative is computed. When `get_all` is set to `True`, all input derivatives are computed. When `get_by_list` is set to `False`, weight derivation is not performed. When `get_by_list` is set to `True`, weight derivation is performed. `sens_param` scales the output value of the network to change the final gradient. Therefore, its dimension is consistent with the output dimension. The following uses the first-order derivation of the MatMul operator for in-depth analysis.

For details about the complete sample code, see [First-order Derivation Sample Code](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/high_order_differentiation/first_order).

### Input Derivation

The input derivation code is as follows:

```python
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import ParameterTuple, Parameter
from mindspore import dtype as mstype
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()
        self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')
    def construct(self, x, y):
        x = x * self.z
        out = self.matmul(x, y)
        return out

class ForwardValueAndGradWrtX(nn.Cell):
    def __init__(self, net):
        super(ForwardValueAndGradWrtX, self).__init__()
        self.net = net
        self.grad = nn.ForwardValueAndGrad(self.net)
    def construct(self, x, y):
        ret = self.grad(x, y)
        return ret

x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
y = Tensor([[0.11, 3.3, 1.1], [1.1, 0.2, 1.4], [1.1, 2.2, 0.3]], dtype=mstype.float32)
output = ForwardValueAndGradWrtX(Net())(x, y)
print(output)
```

The output is as follows:

```text
(Tensor(shape=[2, 3], dtype=Float32, value=
[[9.68000054e-01, 3.20000029e+00, 1.78000009e+00],
 [2.83800006e+00, 8.61999989e+00, 4.13000011e+00]]), Tensor(shape=[2, 3], dtype=Float32, value=
[[4.50999975e+00, 2.70000005e+00, 3.60000014e+00],
 [4.50999975e+00, 2.70000005e+00, 3.60000014e+00]]))
```

If the derivatives of the `x` and `y` inputs are considered, you only need to set `self.grad = nn.ForwardValueAndGrad(self.net, get_all=True)` in `ForwardValueAndGradWrtX`.

### Weight Derivation

If the derivation of weights is considered, change `ForwardValueAndGradWrtX` to the following:

```python
class ForwardValueAndGradWrtX(nn.Cell):
    def __init__(self, net):
        super(ForwardValueAndGradWrtX, self).__init__()
        self.net = net
        self.params = ParameterTuple(net.trainable_params())
        self.grad = nn.ForwardValueAndGrad(self.net, weights=self.params, get_by_list=True)
    def construct(self, x, y):
        ret = self.grad(x, y)
        return ret
```

```python
output = ForwardValueAndGradWrtX(Net())(x, y)
print(output)
```

The output is as follows:

```text
(Tensor(shape=[2, 3], dtype=Float32, value=
[[9.68000054e-01, 3.20000029e+00, 1.78000009e+00]
 [2.83800006e+00, 8.61999989e+00, 4.13000011e+00]]), (Tensor(shape=[1], dtype=Float32, value= [2.15359993e+01]),))
```

### Gradient Value Scaling

You can use the `sens_param` parameter to control the scaling of the gradient value.

```python
class ForwardValueAndGradWrtX(nn.Cell):
    def __init__(self, net):
        super(ForwardValueAndGradWrtX, self).__init__()
        self.net = net
        self.grad = nn.ForwardValueAndGrad(self.net, sens_param=True)
        self.grad_wrt_output = Tensor([[0.1, 0.6, 0.2], [0.8, 1.3, 1.1]], dtype=mstype.float32)
    def construct(self, x, y):
        ret = self.grad(x, y, self.grad_wrt_output)
        return ret
```

```python
output = ForwardValueAndGradWrtX(Net())(x, y)
print(output)
```

The output is as follows:

```text
(Tensor(shape=[2, 3], dtype=Float32, value=
[[9.68000054e-01, 3.20000029e+00, 1.78000009e+00],
 [2.83800006e+00, 8.61999989e+00, 4.13000011e+00]]), Tensor(shape=[2, 3], dtype=Float32, value=
[[2.21099997e+00, 5.09999990e-01, 1.49000001e+00],
 [5.58799982e+00, 2.68000007e+00, 4.07000017e+00]]))
```

`self.grad_wrt_output` may be denoted as the following form:

```python
self.grad_wrt_output = Tensor([[s1, s2, s3], [s4, s5, s6]])
```

The output value after scaling is the product of the original output value and the element corresponding to `self.grad_wrt_output`.
