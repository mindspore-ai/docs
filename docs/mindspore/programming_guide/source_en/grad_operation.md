# Gradient Operation

`Ascend` `GPU` `CPU` `Model Development`

<!-- TOC -->

- [Gradient Operation](#gradient-operation)
    - [Overview](#overview)
    - [First-order Derivation](#first-order-derivation)
        - [Input Derivation](#input-derivation)
        - [Weight Derivation](#weight-derivation)
        - [Gradient Value Scaling](#gradient-value-scaling)
    - [Stop Gradient](#stop-gradient)
    - [High-order Derivation](#high-order-derivation)
        - [Single-input Single-output High-order Derivative](#single-input-single-output-high-order-derivative)
        - [Single-input Multi-output High-order Derivative](#single-input-multi-output-high-order-derivative)
        - [Multiple-Input Multiple-Output High-Order Derivative](#multiple-input-multiple-output-high-order-derivative)
    - [Support for Second-order Differential Operators](#support-for-second-order-differential-operators)
    - [References](#references)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/programming_guide/source_en/grad_operation.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## Overview

GradOperation is used to generate the gradient of the input function. The `get_all`, `get_by_list`, and `sens_param` parameters are used to control the gradient calculation method. For details, see [mindspore API](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.GradOperation.html).

The following is an example of using GradOperation.

## First-order Derivation

The first-order derivative method of MindSpore is `mindspore.ops.GradOperation (get_all=False, get_by_list=False, sens_param=False)`. When `get_all` is set to `False`, the first input derivative is computed. When `get_all` is set to `True`, all input derivatives are computed. When `get_by_list` is set to `False`, weight derivation is not performed. When `get_by_list` is set to `True`, weight derivation is performed. `sens_param` scales the output value of the network to change the final gradient. Therefore, its dimension is consistent with the output dimension. The following uses the first-order derivation of the MatMul operator for in-depth analysis.

For details about the complete sample code, see [First-order Derivation Sample Code](https://gitee.com/mindspore/docs/tree/r1.5/docs/sample_code/high_order_differentiation/first_order).

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

The output is as follows:

```text
[[4.5099998 2.7 3.6000001]
 [4.5099998 2.7 3.6000001]]
```

To facilitate analysis, inputs `x`, `y`, and `z` can be expressed as follows:

```python
x = Tensor([[x1, x2, x3], [x4, x5, x6]])  
y = Tensor([[y1, y2, y3], [y4, y5, y6], [y7, y8, y9]])
z = Tensor([z])
```

The following forward result can be obtained based on the definition of the MatMul operator:

$output = [[(x1 \cdot y1 + x2 \cdot y4 + x3 \cdot y7) \cdot z, (x1 \cdot y2 + x2 \cdot y5 + x3 \cdot y8) \cdot z, (x1 \cdot y3 + x2 \cdot y6 + x3 \cdot y9) \cdot z]$,

$[(x4 \cdot y1 + x5 \cdot y4 + x6 \cdot y7) \cdot z, (x4 \cdot y2 + x5 \cdot y5 + x6 \cdot y8) \cdot z, (x4 \cdot y3 + x5 \cdot y6 + x6 \cdot y9) \cdot z]]$

MindSpore uses the Reverse[3] automatic differentiation mechanism during gradient computation. The output result is summed and then the derivative of the input `x` is computed.

(1) Summation formula:

$\sum{output} = [(x1 \cdot y1 + x2 \cdot y4 + x3 \cdot y7) + (x1 \cdot y2 + x2 \cdot y5 + x3 \cdot y8) + (x1 \cdot y3 + x2 \cdot y6 + x3 \cdot y9) +$

$(x4 \cdot y1 + x5 \cdot y4 + x6 \cdot y7) + (x4 \cdot y2 + x5 \cdot y5 + x6 \cdot y8) + (x4 \cdot y3 + x5 \cdot y6 + x6 \cdot y9)] \cdot z$

(2) Derivation formula:

$\frac{\mathrm{d}(\sum{output})}{\mathrm{d}x} = [[(y1 + y2 + y3) \cdot z, (y4 + y5 + y6) \cdot z, (y7 + y8 + y9) \cdot z], [(y1 + y2 + y3) \cdot z, (y4 + y5 + y6) \cdot z, (y7 + y8 + y9) \cdot z]]$

(3) Computation result:

$\frac{\mathrm{d}(\sum{output})}{\mathrm{d}x} = [[4.5099998 \quad 2.7 \quad 3.6000001] [4.5099998 \quad 2.7 \quad 3.6000001]]$

If the derivatives of the `x` and `y` inputs are considered, you only need to set `self.grad_op = GradOperation(get_all=True)` in `GradNetWrtX`.

### Weight Derivation

If the derivation of weights is considered, change `GradNetWrtX` to the following:

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

```python
output = GradNetWrtX(Net())(x, y)
print(output)
```

The output is as follows:

```text
(Tensor(shape=[1], dtype=Float32, value= [ 2.15359993e+01]),)
```

The derivation formula is changed to:

$\frac{\mathrm{d}(\sum{output})}{\mathrm{d}z} = (x1 \cdot y1 + x2 \cdot y4 + x3 \cdot y7) + (x1 \cdot y2 + x2 \cdot y5 + x3 \cdot y8) + (x1 \cdot y3 + x2 \cdot y6 + x3 \cdot y9) + $

$(x4 \cdot y1 + x5 \cdot y4 + x6 \cdot y7) + (x4 \cdot y2 + x5 \cdot y5 + x6 \cdot y8) + (x4 \cdot y3 + x5 \cdot y6 + x6 \cdot y9)$

Computation result

$\frac{\mathrm{d}(\sum{output})}{\mathrm{d}z} = [2.15359993e+01]$

### Gradient Value Scaling

You can use the `sens_param` parameter to control the scaling of the gradient value.

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
```

```python
output = GradNetWrtX(Net())(x, y)  
print(output)
```

The output is as follows:

```text
[[2.211 0.51 1.49 ]
 [5.588 2.68 4.07 ]]
```

`self.grad_wrt_output` may be denoted as the following form:

```python
self.grad_wrt_output = Tensor([[s1, s2, s3], [s4, s5, s6]])
```

The output value after scaling is the product of the original output value and the element corresponding to `self.grad_wrt_output`.

$output = [[(x1 \cdot y1 + x2 \cdot y4 + x3 \cdot y7) \cdot z \cdot s1, (x1 \cdot y2 + x2 \cdot y5 + x3 \cdot y8) \cdot z \cdot s2, (x1 \cdot y3 + x2 \cdot y6 + x3 \cdot y9) \cdot z \cdot s3], $

$[(x4 \cdot y1 + x5 \cdot y4 + x6 \cdot y7) \cdot z \cdot s4, (x4 \cdot y2 + x5 \cdot y5 + x6 \cdot y8) \cdot z \cdot s5, (x4 \cdot y3 + x5 \cdot y6 + x6 \cdot y9) \cdot z \cdot s6]$

The derivation formula is changed to compute the derivative of the sum of the output values to each element of `x`.

$\frac{\mathrm{d}(\sum{output})}{\mathrm{d}x} = [[(s1 \cdot y1 + s2 \cdot y2 + s3 \cdot y3) \cdot z, (s1 \cdot y4 + s2 \cdot y5 + s3 \cdot y6) \cdot z, (s1 \cdot y7 + s2 \cdot y8 + s3 \cdot y9) \cdot z], $

$[(s4 \cdot y1 + s5 \cdot y2 + s6 \cdot y3) \cdot z, (s4 \cdot y4 + s5 \cdot y5 + s6 \cdot y6) \cdot z, (s4 \cdot y7 + s5 \cdot y8 + s6 \cdot y9) \cdot z]$

To compute the derivative of a single output (for example, `output[0][0]`) to the input, set the scaling value of the corresponding position to 1, and set the scaling values of other positions to 0. You can also change the network structure as follows:

```python
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()
        self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')
    def construct(self, x, y):
        x = x * self.z
        out = self.matmul(x, y)
        return out[0][0]
```

```python
output = GradNetWrtX(Net())(x, y)  
print(output)
```

The output is as follows:

```text
[[0.11 1.1 1.1]
 [0.   0.  0. ]]
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

## High-order Derivation

High-order differentiation is used in domains such as AI-supported scientific computing and second-order optimization. For example, in the molecular dynamics simulation, when the potential energy is trained using the neural network[1], the derivative of the neural network output to the input needs to be computed in the loss function, and then the second-order cross derivative of the loss function to the input and the weight exists in backward propagation. In addition, the second-order derivatives of the output to the input exist in differential equations solved by AI (such as PINNs[2]). Another example is that in order to enable the neural network to converge quickly in the second-order optimization, the second-order derivative of the loss function to the weight needs to be computed using the Newton method.

MindSpore can support high-order derivatives by computing derivatives for multiple times. The following uses several examples to describe how to compute derivatives.

For details about the complete sample code, see [High-order Derivation Sample Code](https://gitee.com/mindspore/docs/tree/r1.5/docs/sample_code/high_order_differentiation/second_order).

### Single-input Single-output High-order Derivative

For example, the second-order derivative (-Sin) of the Sin operator is implemented as follows:

```python
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.sin = ops.Sin()
    def construct(self, x):
        out = self.sin(x)
        return out

class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation()
        self.network = network
    def construct(self, x):
        gout= self.grad(self.network)(x)
        return gout
class GradSec(nn.Cell):
    def __init__(self, network):
        super(GradSec, self).__init__()
        self.grad = ops.GradOperation()
        self.network = network
    def construct(self, x):
        gout= self.grad(self.network)(x)
        return gout

net=Net()
firstgrad = Grad(net) # first order
secondgrad = GradSec(firstgrad) # second order
x_train = Tensor(np.array([1.0], dtype=np.float32))
output = secondgrad(x_train)
print(output)
```

The output is as follows:

```text
[-0.841471]
```

### Single-input Multi-output High-order Derivative

For example, for a multiplication operation with multiple outputs, a high-order derivative of the multiplication operation is as follows:

```python
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import dtype as mstype
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = ops.Mul()
    def construct(self, x):
        out = self.mul(x, x)
        return out

class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation(sens_param=False)
        self.network = network
    def construct(self, x):
        gout = self.grad(self.network)(x)
        return gout
class GradSec(nn.Cell):
    def __init__(self, network):
        super(GradSec, self).__init__()
        self.grad = ops.GradOperation(sens_param=False)
        self.network = network
    def construct(self, x):
        gout = self.grad(self.network)(x)
        return gout

net=Net()
firstgrad = Grad(net) # first order
secondgrad = GradSec(firstgrad) # second order
x = Tensor([0.1, 0.2, 0.3], dtype=mstype.float32)
output = secondgrad(x)
print(output)
```

The output is as follows:

```text
[2. 2. 2.]
```

### Multiple-Input Multiple-Output High-Order Derivative

For example, if a neural network has multiple inputs `x` and `y`, second-order derivatives `dxdx`, `dydy`, `dxdy`, and `dydx` may be obtained by using a gradient scaling mechanism as follows:

```python
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = ops.Mul()

    def construct(self, x, y):
        x_square = self.mul(x, x)
        x_square_y = self.mul(x_square, y)
        return x_square_y

class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation(get_all=True, sens_param=False)
        self.network = network
    def construct(self, x, y):
        gout = self.grad(self.network)(x, y) # return dx, dy
        return gout

class GradSec(nn.Cell):
    def __init__(self, network):
        super(GradSec, self).__init__()
        self.grad = ops.GradOperation(get_all=True, sens_param=True)
        self.network = network
        self.sens1 = Tensor(np.array([1]).astype('float32'))
        self.sens2 = Tensor(np.array([0]).astype('float32'))
    def construct(self, x, y):
        dxdx, dxdy = self.grad(self.network)(x, y, (self.sens1,self.sens2))
        dydx, dydy = self.grad(self.network)(x, y, (self.sens2,self.sens1))
        return dxdx, dxdy, dydx, dydy

net = Net()
firstgrad = Grad(net) # first order
secondgrad = GradSec(firstgrad) # second order
x_train = Tensor(np.array([4],dtype=np.float32))
y_train = Tensor(np.array([5],dtype=np.float32))
dxdx, dxdy, dydx, dydy = secondgrad(x_train, y_train)
print(dxdx, dxdy, dydx, dydy)
```

The output is as follows:

```text
[10] [8.] [8.] [0.]
```

Specifically, results of computing the first-order derivatives are `dx` and `dy`. If `dxdx` is computed, only the first-order derivative `dx` needs to be retained, and scaling values corresponding to `x` and `y` are set to 1 and 0 respectively, that is, `self.grad(self.network)(x, y, (self.sens1,self.sens2))`. Similarly, if `dydy` is computed, only the first-order derivative `dy` is retained, and `sens_param` corresponding to `x` and `y` is set to 0 and 1, respectively, that is, `self.grad(self.network)(x, y, (self.sens2,self.sens1))`.

## Support for Second-order Differential Operators

CPU supports the following operators: [Square](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Square.html#mindspore.ops.Square),
[Exp](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Exp.html#mindspore.ops.Exp), [Neg](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Neg.html#mindspore.ops.Neg), [Mul](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Mul.html#mindspore.ops.Mul), and [MatMul](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.MatMul.html#mindspore.ops.MatMul).

GPU supports the following operators: [Pow](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Pow.html#mindspore.ops.Pow), [Log](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Log.html#mindspore.ops.Log), [Square](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Square.html#mindspore.ops.Square), [Exp](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Exp.html#mindspore.ops.Exp), [Neg](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Neg.html#mindspore.ops.Neg), [Mul](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Mul.html#mindspore.ops.Mul), [Div](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Div.html#mindspore.ops.Div), [MatMul](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.MatMul.html#mindspore.ops.MatMul), [Sin](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Sin.html#mindspore.ops.Sin), [Cos](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Cos.html#mindspore.ops.Cos), [Tan](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Tan.html#mindspore.ops.Tan) and [Atanh](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Atanh.html#mindspore.ops.Atanh).

Ascend supports the following operators: [Pow](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Pow.html#mindspore.ops.Pow), [Log](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Log.html#mindspore.ops.Log), [Square](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Square.html#mindspore.ops.Square), [Exp](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Exp.html#mindspore.ops.Exp), [Neg](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Neg.html#mindspore.ops.Neg), [Mul](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Mul.html#mindspore.ops.Mul), [Div](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Div.html#mindspore.ops.Div), [MatMul](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.MatMul.html#mindspore.ops.MatMul), [Sin](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Sin.html#mindspore.ops.Sin), [Cos](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Cos.html#mindspore.ops.Cos), [Tan](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Tan.html#mindspore.ops.Tan), [Sinh](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Sinh.html#mindspore.ops.Sinh), [Cosh](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Cosh.html#mindspore.ops.Cosh) and [Atanh](https://www.mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.Atanh.html#mindspore.ops.Atanh).

## References

[1] Zhang L, Han J, Wang H, et al. [Deep potential molecular dynamics: a scalable model with the accuracy of quantum mechanics[J]](https://arxiv.org/pdf/1707.09571v2.pdf). Physical review letters, 2018, 120(14): 143001.

[2] Raissi M, Perdikaris P, Karniadakis G E. [Physics informed deep learning (part i): Data-driven solutions of nonlinear partial differential equations[J]](https://arxiv.org/pdf/1711.10561.pdf). arXiv preprint arXiv:1711.10561, 2017.

[3] Baydin A G, Pearlmutter B A, Radul A A, et al. [Automatic differentiation in machine learning: a survey[J]](https://jmlr.org/papers/volume18/17-468/17-468.pdf). The Journal of Machine Learning Research, 2017, 18(1): 5595-5637.
