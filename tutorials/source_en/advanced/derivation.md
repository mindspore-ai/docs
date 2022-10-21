# Advanced Automatic Differentiation

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/advanced/derivation.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

The `grad` and `value_and_grad` provided by the `mindspore.ops` module generate the gradients of the network model. `grad` computes the network gradient, and `value_and_grad` computes both the forward output and the gradient of the network. This article focuses on how to use the main functions of the `grad`, including first-order and second-order derivations, derivation of the input or network weights separately, returning auxiliary variables, and stopping calculating the gradient.

> For more information about the derivative interface, please refer to the [API documentation](https://mindspore.cn/docs/en/master/api_python/mindspore/mindspore.grad.html).

## First-order Derivation

Method: `mindspore.ops.grad`. The parameter usage is as follows:

- `fn`: the function or network to be derived.
- `grad_position`: specifies the index of the input position to be derived. If the index is int type, it means to derive for a single input; if tuple type, it means to derive for the position of the index within the tuple, where the index starts from 0; and if None, it means not to derive for the input. In this scenario, `weights` is non-None. Default: 0.
- `weights`: the network variables that need to return the gradients in the training network. Generally the network variables can be obtained by `weights = net.trainable_params()`. Default: None.
- `has_aux`: symbol for whether to return auxiliary arguments. If True, the number of `fn` outputs must be more than one, where only the first output of `fn` is involved in the derivation and the other output values will be returned directly. Default: False.

The following is a brief introduction to the use of the `grad` by first constructing a customized network model `Net` and then performing a first-order derivative on it:

$$f(x, y)=(x * z) * y \tag{1}$$

First, define the network model `Net`, input `x`, and input `y`.

```python
import numpy as np
from mindspore import ops, Tensor
import mindspore.nn as nn
import mindspore as ms

# Define the inputs x and y.
x = Tensor([3.0], dtype=ms.float32)
y = Tensor([5.0], dtype=ms.float32)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.z = ms.Parameter(ms.Tensor(np.array([1.0], np.float32)), name='z')

    def construct(self, x, y):
        out = x * x * y * self.z
        return out
```

### Computing the First-order Derivative for Input

To derive the inputs `x` and `y`, set `grad_position` to (0, 1):

$$\frac{\partial f}{\partial x}=2 * x * y * z \tag{2}$$

$$\frac{\partial f}{\partial y}=x * x * z \tag{3}$$

```python
net = Net()
grad_fn = ops.grad(net, grad_position=(0, 1))
gradients = grad_fn(x, y)
print(gradients)
```

```text
    (Tensor(shape=[1], dtype=Float32, value= [ 3.00000000e+01]), Tensor(shape=[1], dtype=Float32, value= [ 9.00000000e+00]))
```

### Computing the Derivative for Weight

Derive for the weight `z`, where it is not necessary to derive for the inputs, and set `grad_position` to None:

$$\frac{\partial f}{\partial z}=x * x * y \tag{4}$$

```python
params = ms.ParameterTuple(net.trainable_params())

output = ops.grad(net, grad_position=None, weights=params)(x, y)
print(output)
```

```text
    (Tensor(shape=[1], dtype=Float32, value= [ 4.50000000e+01]),)
```

### Returning Auxiliary Variables

Simultaneous derivation for the inputs and weights, where only the first output is involved in the derivation, with the following sample code:

```python
net = nn.Dense(10, 1)
loss_fn = nn.MSELoss()


def forward(inputs, labels):
    logits = net(inputs)
    loss = loss_fn(logits, labels)
    return loss, logits


inputs = Tensor(np.random.randn(16, 10).astype(np.float32))
labels = Tensor(np.random.randn(16, 1).astype(np.float32))
weights = net.trainable_params()

# Aux value does not contribute to the gradient.
grad_fn = ops.grad(forward, grad_position=0, weights=None, has_aux=True)
inputs_gradient, (aux_logits,) = grad_fn(inputs, labels)
print(len(inputs_gradient), aux_logits.shape)
```

```text
    16, (16, 1)
```

### Stopping Gradient Computation

You can use `stop_gradient` to stop computing the gradient of a specified operator to eliminate the impact of the operator on the gradient.

Based on the matrix multiplication network model used for the first-order derivation, add an operator `out2` and disable the gradient computation to obtain the customized network `Net2`. Then, check the derivation result of the input.

The sample code is as follows:

```python
class Net(nn.Cell):

    def __init__(self):
        super(Net, self).__init__()

    def construct(self, x, y):
        out1 = x * y
        out2 = x * y
        out2 = ops.stop_gradient(out2) # Stop computing the gradient of the out2 operator.
        out = out1 + out2
        return out


net = Net()
grad_fn = ops.grad(net)
output = grad_fn(x, y)
print(output)
```

```text
    [5.0]
```

According to the preceding information, `stop_gradient` is set for `out2`. Therefore, `out2` does not contribute to gradient computation. The output result is the same as that when `out2` is not added.

Delete `out2 = stop_gradient(out2)` and check the output result. An example of the code is as follows:

```python
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()

    def construct(self, x, y):
        out1 = x * y
        out2 = x * y
        # out2 = stop_gradient(out2)
        out = out1 + out2
        return out


net = Net()
grad_fn = ops.grad(net)
output = grad_fn(x, y)
print(output)
```

```text
    [10.0]
```

According to the printed result, after the gradient of the `out2` operator is computed, the gradients generated by the `out2` and `out1` operators are the same. Therefore, the value of each item in the result is twice the original value (accuracy error exists).

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

x_train = ms.Tensor(np.array([3.1415926]), dtype=ms.float32)
y_train = ms.Tensor(np.array([3.1415926]), dtype=ms.float32)

net = Net()
firstgrad = ops.grad(net, grad_position=(0, 1))
secondgrad = ops.grad(firstgrad, grad_position=(0, 1))
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
