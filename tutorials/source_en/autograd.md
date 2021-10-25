# Automatic Differentiation

`Ascend` `GPU` `CPU` `Beginner` `Model Development`

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/autograd.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

Backward propagation is the commonly used algorithm for training neural networks. In this algorithm, parameters (model weights) are adjusted based on a gradient of a loss function for a given parameter.

The first-order derivative method of MindSpore is `mindspore.ops.GradOperation (get_all=False, get_by_list=False, sens_param=False)`. When `get_all` is set to `False`, the first input derivative is computed. When `get_all` is set to `True`, all input derivatives are computed. When `get_by_list` is set to `False`, weight derivatives are not computed. When `get_by_list` is set to `True`, the weight derivative is computed. `sens_param` scales the output value of the network to change the final gradient. The following uses the MatMul operator derivative for in-depth analysis.

Import the required modules and APIs:

```python
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import ParameterTuple, Parameter
from mindspore import dtype as mstype
```

## First-order Derivative of the Input

To compute the input derivative, you need to define a network requiring a derivative. The following uses a network $f(x,y)=z *x* y$ formed by the MatMul operator as an example.

The network structure is as follows:

```python
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()
        self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

    def construct(self, x, y):
        x = x * self.z
        out = self.matmul(x, y)
        return out
```

Define the network requiring the derivative. In the `__init__` function, define the `self.net` and `ops.GradOperation` networks. In the `construct` function, compute the derivative of `self.net`.

The network structure is as follows:

```python
class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation()

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y)
```

Define the input and display the output:

```python
x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
y = Tensor([[0.11, 3.3, 1.1], [1.1, 0.2, 1.4], [1.1, 2.2, 0.3]], dtype=mstype.float32)
output = GradNetWrtX(Net())(x, y)
print(output)
```

```text
    [[4.5099998 2.7       3.6000001]
     [4.5099998 2.7       3.6000001]]
```

If the derivatives of the `x` and `y` inputs are considered, you only need to set `self.grad_op = GradOperation(get_all=True)` in `GradNetWrtX`.

## First-order Derivative of the Weight

To compute weight derivatives, you need to set `get_by_list` in `ops.GradOperation` to `True`.

The `GradNetWrtX` structure is as follows:

```python
class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.params = ParameterTuple(net.trainable_params())
        self.grad_op = ops.GradOperation(get_by_list=True)

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(x, y)
```

Run and display the output:

```python
output = GradNetWrtX(Net())(x, y)
print(output)
```

```text
(Tensor(shape=[1], dtype=Float32, value= [ 2.15359993e+01]),)
```

If computation of certain weight derivatives is not required, set `requirements_grad` to `False` when defining the network requiring derivatives.

```Python
self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z', requires_grad=False)
```

## Gradient Value Scaling

You can use the `sens_param` parameter to scale the output value of the network to change the final gradient. Set `sens_param` in `ops.GradOperation` to `True` and determine the scaling index. The dimension must be the same as the output dimension.

The scaling index `self.grad_wrt_output` may be in the following format:

```python
self.grad_wrt_output = Tensor([[s1, s2, s3], [s4, s5, s6]])
```

The `GradNetWrtX` structure is as follows:

```python
class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation(sens_param=True)
        self.grad_wrt_output = Tensor([[0.1, 0.6, 0.2], [0.8, 1.3, 1.1]], dtype=mstype.float32)

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y, self.grad_wrt_output)

output = GradNetWrtX(Net())(x, y)  
print(output)
```

```text
[[2.211 0.51  1.49 ]
 [5.588 2.68  4.07 ]]
```

## Stop Gradient

We can use `stop_gradient` to disable calculation of gradient for certain operators. For example:

```python
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import ParameterTuple, Parameter
from mindspore import dtype as mstype
from mindspore.ops import stop_gradient

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()

    def construct(self, x, y):
        out1 = self.matmul(x, y)
        out2 = self.matmul(x, y)
        out2 = stop_gradient(out2)
        out = out1 + out2
        return out

class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation()

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y)

x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
y = Tensor([[0.11, 3.3, 1.1], [1.1, 0.2, 1.4], [1.1, 2.2, 0.3]], dtype=mstype.float32)
output = GradNetWrtX(Net())(x, y)
print(output)
```

```text
    [[4.5, 2.7, 3.6],
     [4.5, 2.7, 3.6]]
```

Here, we set `stop_gradient` to `out2`, so this operator does not have any contribution to gradient. If we delete `out2 = stop_gradient(out2)`, the result is:

```text
    [[9.0, 5.4, 7.2],
     [9.0, 5.4, 7.2]]
```

After we do not set `stop_gradient` to `out2`, it will make the same contribution to gradient as `out1`. So we can see that each result has doubled.  