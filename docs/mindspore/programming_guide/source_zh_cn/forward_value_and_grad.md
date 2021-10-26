# ForwardValueAndGrad

`Ascend` `GPU` `CPU` `模型运行`

<!-- TOC -->

- [ForwardValueAndGrad](#forward-value-and-grad)
    - [概述](#概述)
    - [用例](#一阶求导)
        - [对输入求导](#对输入求导)
        - [对权重求导](#对权重求导)
        - [梯度值缩放](#梯度值缩放)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/forward_value_and_grad.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

`ForwardValueAndGrad` 用来返回输入网络或者函数的正向计算结果和参数的反向梯度值。`get_all`, `get_by_list`, 和 `sens_param` 参数用来控制梯度的计算范围。详细说明可见 [mindspore API](https://www.mindspore.cn/docs/api/en/master/api_python/nn/mindspore.nn.ForwardValueAndGrad.html).

下面是使用`ForwardValueAndGrad`的用例说明.

## 一阶求导

MindSpore可以用`mindspore.nn.ForwardValueAndGrad (network, weights=None, get_all=False, get_by_list=False, sens_param=False)`来计算正向结果和一阶导数。 当`get_all` 和`get_by_list` 都设置成 `False`，将会输出第一个入参的导数。 当 `get_all` 设置成 `True`，所有入参的导数将会被输出。当 `get_by_list` 设置成 `False`，权重的导数将不会被输出。当 `get_by_list` 被设置成 `True`，权重的导数将会被输出。`sens_param` 用来缩放网络的正向输出值，用而影响梯度的计算结果。因此，它的维度和网络的正向输出结果一致。下面将举例说明。

### 对输入求导

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

输出结果是:

```text
(Tensor(shape=[2, 3], dtype=Float32, value=
[[9.68000054e-01, 3.20000029e+00, 1.78000009e+00],
 [2.83800006e+00, 8.61999989e+00, 4.13000011e+00]]), Tensor(shape=[2, 3], dtype=Float32, value=
[[4.50999975e+00, 2.70000005e+00, 3.60000014e+00],
 [4.50999975e+00, 2.70000005e+00, 3.60000014e+00]]))
```

如果需要同时输出 `x` 和 `y` 的梯度，可以在 `ForwardValueAndGradWrtX` 设置 `self.grad = nn.ForwardValueAndGrad(self.net, get_all=True)`。

### 对权重求导

如果需要输出权重的梯度，可以参照下面的例子:

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

输出结果是:

```text
(Tensor(shape=[2, 3], dtype=Float32, value=
[[9.68000054e-01, 3.20000029e+00, 1.78000009e+00]
 [2.83800006e+00, 8.61999989e+00, 4.13000011e+00]]), (Tensor(shape=[1], dtype=Float32, value= [2.15359993e+01]),))
```

### 梯度值缩放

你可以使用参数 `sens_param` 来控制梯度的缩放值。

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

输出结果是:

```text
(Tensor(shape=[2, 3], dtype=Float32, value=
[[9.68000054e-01, 3.20000029e+00, 1.78000009e+00],
 [2.83800006e+00, 8.61999989e+00, 4.13000011e+00]]), Tensor(shape=[2, 3], dtype=Float32, value=
[[2.21099997e+00, 5.09999990e-01, 1.49000001e+00],
 [5.58799982e+00, 2.68000007e+00, 4.07000017e+00]]))
```

`self.grad_wrt_output` 可以统一写成以下的格式:

```python
self.grad_wrt_output = Tensor([[s1, s2, s3], [s4, s5, s6]])
```

原始的正向输出结果与参数 `self.grad_wrt_output` 对应的元素相乘，产生缩放之后的输出结果。
