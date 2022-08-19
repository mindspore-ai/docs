# Automatic Derivation

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/advanced/network/derivation.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

The `GradOperation` API provided by the `mindspore.ops` module can be used to generate a gradient of a network model. The following describes how to use the `GradOperation` API to perform first-order and second-order derivations and how to stop gradient computation.

> For details about `GradOperation`, see [API](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.GradOperation.html#mindspore.ops.GradOperation).

## First-order Derivation

Method: `mindspore.ops.GradOperation()`. The parameter usage is as follows:

- `get_all`: If this parameter is set to `False`, the derivation is performed only on the first input. If this parameter is set to `True`, the derivation is performed on all inputs.
- `get_by_list`: If this parameter is set to `False`, the weight derivation is not performed. If this parameter is set to `True`, the weight derivation is performed.
- `sens_param`: The output value of the network is scaled to change the final gradient. Therefore, the dimension is the same as the output dimension.

The [MatMul](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.MatMul.html#mindspore.ops.MatMul) operator is used to build a customized network model `Net`, and then perform first-order derivation on the model. The following formula is an example to describe how to use the `GradOperation` API:

$$f(x, y)=(x * z) * y \tag{1}$$

First, define the network model `Net`, input `x`, and input `y`.

```python
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms

# Define the inputs x and y.
x = ms.Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=ms.float32)
y = ms.Tensor([[0.11, 3.3, 1.1], [1.1, 0.2, 1.4], [1.1, 2.2, 0.3]], dtype=ms.float32)

class Net(nn.Cell):
    """Define the matrix multiplication network Net."""

    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()
        self.z = ms.Parameter(ms.Tensor(np.array([1.0], np.float32)), name='z')

    def construct(self, x, y):
        x = x * self.z
        out = self.matmul(x, y)
        return out
```

### Computing the Input Derivative

Compute the input derivative. The code is as follows:

```python
class GradNetWrtX(nn.Cell):
    """Define the first-order derivation of network input."""

    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation()

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y)

output = GradNetWrtX(Net())(x, y)
print(output)
```

```text
    [[4.5099998 2.7       3.6000001]
     [4.5099998 2.7       3.6000001]]
```

The preceding result is explained as follows. To facilitate analysis, the preceding inputs `x` and `y`, and weight `z` are expressed as follows:

```text
x = ms.Tensor([[x1, x2, x3], [x4, x5, x6]])
y = ms.Tensor([[y1, y2, y3], [y4, y5, y6], [y7, y8, y9]])
z = ms.Tensor([z])
```

The following forward result can be obtained based on the definition of the MatMul operator:

$$output = [[(x_1 \cdot y_1 + x_2 \cdot y_4 + x_3 \cdot y_7) \cdot z, (x_1 \cdot y_2 + x_2 \cdot y_5 + x_3 \cdot y_8) \cdot z, (x_1 \cdot y_3 + x_2 \cdot y_6 + x_3 \cdot y_9) \cdot z],$$

$$[(x_4 \cdot y_1 + x_5 \cdot y_4 + x_6 \cdot y_7) \cdot z, (x_4 \cdot y_2 + x_5 \cdot y_5 + x_6 \cdot y_8) \cdot z, (x_4 \cdot y_3 + x_5 \cdot y_6 + x_6 \cdot y_9) \cdot z]] \tag{2}$$

MindSpore uses the reverse-mode automatic differentiation mechanism during gradient computation. The output result is summed and then the derivative of the input `x` is computed.

1. Sum formula:

    $$\sum{output} = [(x_1 \cdot y_1 + x_2 \cdot y_4 + x_3 \cdot y_7) + (x_1 \cdot y_2 + x_2 \cdot y_5 + x_3 \cdot y_8) + (x_1 \cdot y_3 + x_2 \cdot y_6 + x_3 \cdot y_9)$$

    $$+ (x_4 \cdot y_1 + x_5 \cdot y_4 + x_6 \cdot y_7) + (x_4 \cdot y_2 + x_5 \cdot y_5 + x_6 \cdot y_8) + (x_4 \cdot y_3 + x_5 \cdot y_6 + x_6 \cdot y_9)] \cdot z \tag{3}$$

2. Derivation formula:

    $$\frac{\mathrm{d}(\sum{output})}{\mathrm{d}x} = [[(y_1 + y_2 + y_3) \cdot z, (y_4 + y_5 + y_6) \cdot z, (y_7 + y_8 + y_9) \cdot z],$$

    $$[(y_1 + y_2 + y_3) \cdot z, (y_4 + y_5 + y_6) \cdot z, (y_7 + y_8 + y_9) \cdot z]] \tag{4}$$

3. Computation result:

    $$\frac{\mathrm{d}(\sum{output})}{\mathrm{d}x} = [[4.51 \quad 2.7 \quad 3.6] [4.51 \quad 2.7 \quad 3.6]] \tag{5}$$

    > If the derivatives of the `x` and `y` inputs are considered, you only need to set `self.grad_op = GradOperation(get_all=True)` in `GradNetWrtX`.

### Computing the Weight Derivative

Compute the weight derivative. The sample code is as follows:

```python
class GradNetWrtZ(nn.Cell):
    """Define the first-order derivation of network weight.""

    def __init__(self, net):
        super(GradNetWrtZ, self).__init__()
        self.net = net
        self.params = ms.ParameterTuple(net.trainable_params())
        self.grad_op = ops.GradOperation(get_by_list=True)

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(x, y)

output = GradNetWrtZ(Net())(x, y)
print(output[0])
```

```text
    [21.536]
```

The following formula is used to explain the preceding result. A derivation formula for the weight is:

$$\frac{\mathrm{d}(\sum{output})}{\mathrm{d}z} = (x_1 \cdot y_1 + x_2 \cdot y_4 + x_3 \cdot y_7) + (x_1 \cdot y_2 + x_2 \cdot y_5 + x_3 \cdot y_8) + (x_1 \cdot y_3 + x_2 \cdot y_6 + x_3 \cdot y_9)$$

$$+ (x_4 \cdot y_1 + x_5 \cdot y_4 + x_6 \cdot y_7) + (x_4 \cdot y_2 + x_5 \cdot y_5 + x_6 \cdot y_8) + (x_4 \cdot y_3 + x_5 \cdot y_6 + x_6 \cdot y_9) \tag{6}$$

Computation result:

$$\frac{\mathrm{d}(\sum{output})}{\mathrm{d}z} = [2.1536e+01] \tag{7}$$

### Gradient Value Scaling

You can use the `sens_param` parameter to control the scaling of the gradient value.

```python
class GradNetWrtN(nn.Cell):
    """Define the first-order derivation of the network and control gradient value scaling."""
    def __init__(self, net):
        super(GradNetWrtN, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation(sens_param=True)

        # Define gradient value scaling.
        self.grad_wrt_output = ms.Tensor([[0.1, 0.6, 0.2], [0.8, 1.3, 1.1]], dtype=ms.float32)

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y, self.grad_wrt_output)

output = GradNetWrtN(Net())(x, y)
print(output)
```

```text
    [[2.211 0.51  1.49 ]
     [5.588 2.68  4.07 ]]
```

To facilitate the explanation of the preceding result, `self.grad_wrt_output` is recorded as follows:

```text
self.grad_wrt_output = ms.Tensor([[s1, s2, s3], [s4, s5, s6]])
```

The output value after scaling is the product of the original output value and the element corresponding to `self.grad_wrt_output`. The formula is as follows:

$$output = [[(x_1 \cdot y_1 + x_2 \cdot y_4 + x_3 \cdot y_7) \cdot z \cdot s_1, (x_1 \cdot y_2 + x_2 \cdot y_5 + x_3 \cdot y_8) \cdot z \cdot s_2, (x_1 \cdot y_3 + x_2 \cdot y_6 + x_3 \cdot y_9) \cdot z \cdot s_3], $$

$$[(x_4 \cdot y_1 + x_5 \cdot y_4 + x_6 \cdot y_7) \cdot z \cdot s_4, (x_4 \cdot y_2 + x_5 \cdot y_5 + x_6 \cdot y_8) \cdot z \cdot s_5, (x_4 \cdot y_3 + x_5 \cdot y_6 + x_6 \cdot y_9) \cdot z \cdot s_6]] \tag{8}$$

The derivation formula is changed to compute the derivative of the sum of the output values to each element of `x`.

$$\frac{\mathrm{d}(\sum{output})}{\mathrm{d}x} = [[(s_1 \cdot y_1 + s_2 \cdot y_2 + s_3 \cdot y_3) \cdot z, (s_1 \cdot y_4 + s_2 \cdot y_5 + s_3 \cdot y_6) \cdot z, (s_1 \cdot y_7 + s_2 \cdot y_8 + s_3 \cdot y_9) \cdot z],$$

$$[(s_4 \cdot y_1 + s_5 \cdot y_2 + s_6 \cdot y_3) \cdot z, (s_4 \cdot y_4 + s_5 \cdot y_5 + s_6 \cdot y_6) \cdot z, (s_4 \cdot y_7 + s_5 \cdot y_8 + s_6 \cdot y_9) \cdot z]] \tag{9}$$

Computation result:

$$\frac{\mathrm{d}(\sum{output})}{\mathrm{d}x} = [[2.211 \quad 0.51 \quad 1.49][5.588 \quad 2.68 \quad 4.07]] \tag{10}$$

### Stopping Gradient Computation

You can use `stop_gradient` to stop computing the gradient of a specified operator to eliminate the impact of the operator on the gradient.

Based on the matrix multiplication network model used for the first-order derivation, add an operator `out2` and disable the gradient computation to obtain the customized network `Net2`. Then, check the derivation result of the input.

The sample code is as follows:

```python
class Net(nn.Cell):

    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()

    def construct(self, x, y):
        out1 = self.matmul(x, y)
        out2 = self.matmul(x, y)
        out2 = ops.stop_gradient(out2) # Stop computing the gradient of the out2 operator.
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

output = GradNetWrtX(Net())(x, y)
print(output)
```

```text
    [[4.5099998 2.7       3.6000001]
     [4.5099998 2.7       3.6000001]]
```

According to the preceding information, `stop_gradient` is set for `out2`. Therefore, `out2` does not contribute to gradient computation. The output result is the same as that when `out2` is not added.

Delete `out2 = stop_gradient(out2)` and check the output. An example of the code is as follows:

```python
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()

    def construct(self, x, y):
        out1 = self.matmul(x, y)
        out2 = self.matmul(x, y)
        # out2 = stop_gradient(out2)
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

output = GradNetWrtX(Net())(x, y)
print(output)
```

```text
    [[9.0199995 5.4       7.2000003]
     [9.0199995 5.4       7.2000003]]
```

According to the printed result, after the gradient of the `out2` operator is computed, the gradients generated by the `out2` and `out1` operators are the same. Therefore, the value of each item in the result is twice the original value (accuracy error exists).

### Customized Backward Propagation Function

When MindSpore is used to build a neural network, the `nn.Cell` class needs to be inherited. When there are some operations that do not define backward propagation rules on the network, or when you want to control the gradient computation process of the entire network, you can use the function of customizing the backward propagation function of the `nn.Cell` object. The format is as follows:

```python
def bprop(self, ..., out, dout):
    return ...
```

- Input parameters: Input parameters in the forward porpagation plus `out` and `dout`. `out` indicates the computation result of the forward porpagation, and `dout` indicates the gradient returned to the `nn.Cell` object.
- Return values: Gradient of each input in the forward porpagation. The number of return values must be the same as the number of inputs in the forward porpagation.

A complete example is as follows:

```python
import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()

    def construct(self, x, y):
        out = self.matmul(x, y)
        return out

    def bprop(self, x, y, out, dout):
        dx = x + 1
        dy = y + 1
        return dx, dy


class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation(get_all=True)

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y)


x = ms.Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=ms.float32)
y = ms.Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=ms.float32)
out = GradNet(Net())(x, y)
print(out)
```

```text
    (Tensor(shape=[2, 3], dtype=Float32, value=
    [[ 1.50000000e+00,  1.60000002e+00,  1.39999998e+00],
     [ 2.20000005e+00,  2.29999995e+00,  2.09999990e+00]]), Tensor(shape=[3, 3], dtype=Float32, value=
    [[ 1.00999999e+00,  1.29999995e+00,  2.09999990e+00],
     [ 1.10000002e+00,  1.20000005e+00,  2.29999995e+00],
     [ 3.09999990e+00,  2.20000005e+00,  4.30000019e+00]]))
```

Constraints

- If the number of return values of the `bprop` function is 1, the return value must be written in the tuple format, that is, `return (dx,)`.
- In graph mode, the `bprop` function needs to be converted into a graph IR. Therefore, the static graph syntax must be complied with. For details, see [Static Graph Syntax Support](https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html).
- Only the gradient of the forward porpagation input can be returned. The gradient of the `Parameter` cannot be returned.
- `Parameter` cannot be used in `bprop`.

## High-order Derivation

High-order differentiation is used in domains such as AI-supported scientific computing and second-order optimization. For example, in the molecular dynamics simulation, when the potential energy is trained using the neural network, the derivative of the neural network output to the input needs to be computed in the loss function, and then the second-order cross derivative of the loss function to the input and the weight exists in backward propagation.

In addition, the second-order derivative of the output to the input exists in a differential equation solved by AI (such as PINNs). Another example is that in order to enable the neural network to converge quickly in the second-order optimization, the second-order derivative of the loss function to the weight needs to be computed using the Newton method.

MindSpore can support high-order derivatives by computing derivatives for multiple times. The following uses several examples to describe how to compute derivatives.

### Single-input Single-output High-order Derivative

For example, the formula of the Sin operator is as follows:

$$f(x) = sin(x) \tag{1}$$

The first derivative is:

$$f'(x) = cos(x) \tag{2}$$

The second derivative is:

$$f''(x) = cos'(x) = -sin(x) \tag{3}$$

The second derivative (-Sin) is implemented as follows:

```python
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms

class Net(nn.Cell):
    """Feedforward network model"""
    def __init__(self):
        super(Net, self).__init__()
        self.sin = ops.Sin()

    def construct(self, x):
        out = self.sin(x)
        return out

class Grad(nn.Cell):
    """First-order derivation"""
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation()
        self.network = network

    def construct(self, x):
        gout = self.grad(self.network)(x)
        return gout

class GradSec(nn.Cell):
    """Second order derivation"""
    def __init__(self, network):
        super(GradSec, self).__init__()
        self.grad = ops.GradOperation()
        self.network = network

    def construct(self, x):
        gout = self.grad(self.network)(x)
        return gout

x_train = ms.Tensor(np.array([3.1415926]), dtype=ms.float32)

net = Net()
firstgrad = Grad(net)
secondgrad = GradSec(firstgrad)
output = secondgrad(x_train)

# Print the result.
result = np.around(output.asnumpy(), decimals=2)
print(result)
```

```text
    [-0.]
```

The preceding print result shows that the value of $-sin(3.1415926)$ is close to $0$.

### Single-input Multi-output High-order Derivative

Compute the derivation of the following formula:

$$f(x) = (f_1(x), f_2(x)) \tag{1}$$

Where:

$$f_1(x) = sin(x) \tag{2}$$

$$f_2(x) = cos(x) \tag{3}$$

MindSpore uses the reverse-mode automatic differentiation mechanism during gradient computation. The output result is summed and then the derivative of the input is computed. Therefore, the first derivative is:

$$f'(x) = cos(x)  -sin(x) \tag{4}$$

The second derivative is:

$$f''(x) = -sin(x) - cos(x) \tag{5}$$

```python
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms

class Net(nn.Cell):
    """Feedforward network model"""
    def __init__(self):
        super(Net, self).__init__()
        self.sin = ops.Sin()
        self.cos = ops.Cos()

    def construct(self, x):
        out1 = self.sin(x)
        out2 = self.cos(x)
        return out1, out2

class Grad(nn.Cell):
    """First-order derivation"""
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation()
        self.network = network

    def construct(self, x):
        gout = self.grad(self.network)(x)
        return gout

class GradSec(nn.Cell):
    """Second order derivation"""
    def __init__(self, network):
        super(GradSec, self).__init__()
        self.grad = ops.GradOperation()
        self.network = network

    def construct(self, x):
        gout = self.grad(self.network)(x)
        return gout

x_train = ms.Tensor(np.array([3.1415926]), dtype=ms.float32)

net = Net()
firstgrad = Grad(net)
secondgrad = GradSec(firstgrad)
output = secondgrad(x_train)

# Print the result.
result = np.around(output.asnumpy(), decimals=2)
print(result)
```

```text
    [1.]
```

The preceding print result shows that the value of $-sin(3.1415926) - cos(3.1415926)$ is close to $1$.

### Multiple-Input Multiple-Output High-Order Derivative

Compute the derivation of the following formula:

$$f(x, y) = (f_1(x, y), f_2(x, y)) \tag{1}$$

Where:

$$f_1(x, y) = sin(x) - cos(y)  \tag{2}$$

$$f_2(x, y) = cos(x) - sin(y)  \tag{3}$$

MindSpore uses the reverse-mode automatic differentiation mechanism during gradient computation. The output result is summed and then the derivative of the input is computed.

Sum:

$$\sum{output} = sin(x) + cos(x) - sin(y) - cos(y) \tag{4}$$

The first derivative of output sum with respect to input $x$ is:

$$\dfrac{\mathrm{d}\sum{output}}{\mathrm{d}x} = cos(x) - sin(x) \tag{5}$$

The second derivative of output sum with respect to input $x$ is:

$$\dfrac{\mathrm{d}\sum{output}^{2}}{\mathrm{d}^{2}x} = -sin(x) - cos(x) \tag{6}$$

The first derivative of output sum with respect to input $y$ is:

$$\dfrac{\mathrm{d}\sum{output}}{\mathrm{d}y} = -cos(y) + sin(y) \tag{7}$$

The second derivative of output sum with respect to input $y$ is:

$$\dfrac{\mathrm{d}\sum{output}^{2}}{\mathrm{d}^{2}y} = sin(y) + cos(y) \tag{8}$$

```python
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms

class Net(nn.Cell):
    """Feedforward network model"""
    def __init__(self):
        super(Net, self).__init__()
        self.sin = ops.Sin()
        self.cos = ops.Cos()

    def construct(self, x, y):
        out1 = self.sin(x) - self.cos(y)
        out2 = self.cos(x) - self.sin(y)
        return out1, out2

class Grad(nn.Cell):
    """First-order derivation"""
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation(get_all=True)
        self.network = network

    def construct(self, x, y):
        gout = self.grad(self.network)(x, y)
        return gout

class GradSec(nn.Cell):
    """Second order derivation"""
    def __init__(self, network):
        super(GradSec, self).__init__()
        self.grad = ops.GradOperation(get_all=True)
        self.network = network

    def construct(self, x, y):
        gout = self.grad(self.network)(x, y)
        return gout

x_train = ms.Tensor(np.array([3.1415926]), dtype=ms.float32)
y_train = ms.Tensor(np.array([3.1415926]), dtype=ms.float32)

net = Net()
firstgrad = Grad(net)
secondgrad = GradSec(firstgrad)
output = secondgrad(x_train, y_train)

# Print the result.
print(np.around(output[0].asnumpy(), decimals=2))
print(np.around(output[1].asnumpy(), decimals=2))
```

```text
    [1.]
    [-1.]
```

According to the preceding result, the value of the second derivative $-sin(3.1415926) - cos(3.1415926)$ of the output to the input $x$ is close to $1$, and the value of the second derivative $sin(3.1415926) + cos(3.1415926)$ of the output to the input $y$ is close to $-1$.

> The accuracy may vary depending on the computing platform. Therefore, the execution results of the code in this section vary slightly on different platforms.
