# Automatic Differentiation

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/autograd.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

Automatic differentiation is able to calculate the derivative value of a derivative function at a certain point, which is a generalization of backpropagation algorithms. The main problem solved by automatic differentiation is to decompose a complex mathematical operation into a series of simple basic operations, which shields the user from a large number of details and processes of differentiation, which greatly reduces the threshold for the use of the framework.

MindSpore uses `ops.GradOperation` to calculate a first-order derivative, and the attributes of the first-order derivative are as the following:

- `get_all`：Whether to derive the input parameters, the default value is False.
- `get_by_list`：Whether to derive the weight parameters, the default value is False.
- `sens_param`：Whether to scale the output value of the network to change the final gradient, the default value is False.

This chapter uses `ops.GradOperation` in MindSpore to find first-order derivatives of the function $f(x)=wx+b$.

## First-order Derivative of the Input

The formula needs to be defined before the input can be derived:$f(x)=wx+b \tag {1}$

The example code below is an expression of Equation (1), and since MindSpore is functionally programmed, all expressions of computational formulas are represented as functions.

```python
import numpy as np
import mindspore.nn as nn
from mindspore import Parameter, Tensor

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.w = Parameter(np.array([6.0]), name='w')
        self.b = Parameter(np.array([1.0]), name='b')

    def construct(self, x):
        f = self.w * x + self.b
        return f
```

Define the derivative class `GradNet`. In the `__init__` function, define the `self.net` and `ops.GradOperation` networks. In the `construct` function, compute the derivative of `self.net`. Its corresponding MindSpore internally produces the following formula (2):$f^{'}(x)=w\tag {2}$

```python
from mindspore import dtype as mstype
import mindspore.ops as ops

class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation()

    def construct(self, x):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x)
```

At last, define the weight parameter as w and a first-order derivative is found for the input parameter x in the input formula (1). From the running result, the input in formula (1) is 6, that is:
$$
f(x)=wx+b=6*x+1 \tag {3}
$$
 To derive the above equation, there is:
$$
f^{'}(x)=w=6 \tag {4}
$$

```python
x = Tensor([100], dtype=mstype.float32)
output = GradNet(Net())(x)

print(output)
```

```text
[6.]
```

MindSpore calculates the first derivative method `ops.GradOperation (get_all=False, get_by_lsit=False, sens_param=False)`, where when `get_all` is `False`, only the first input is evaluated, and when `True` is set, all inputs are evaluated.

## First-order Derivative of the Weight

To compute weight derivatives, you need to set `get_by_list` in `ops.GradOperation` to `True`.

```python
from mindspore import ParameterTuple

class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.net = net
        self.params = ParameterTuple(net.trainable_params())
        self.grad_op = ops.GradOperation(get_by_list=True)  # Set the first-order derivative of the weight parameters

    def construct(self, x):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(x)
```

Next, derive the function:

```python
# Perform a derivative calculation on the function
x = Tensor([100], dtype=mstype.float32)
fx = GradNet(Net())(x)

# Print the results
print(fx)
print(f"wgrad: {fx[0]}\nbgrad: {fx[1]}")
```

```text
(Tensor(shape=[1], dtype=Float32, value= [ 6.00000000e+00]), Tensor(shape=[1], dtype=Float32, value= [ 1.00000000e+00]))
wgrad: [6.]
bgrad: [1.]
```

If computation of certain weight derivatives is not required, set `requirements_grad` to `False` when defining the network requiring derivatives.

```Python
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.w = Parameter(Tensor(np.array([6], np.float32)), name='w')
        self.b = Parameter(Tensor(np.array([1.0], np.float32)), name='b', requires_grad=False)

    def construct(self, x):
        out = x * self.w + self.b
        return out

class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.net = net
        self.params = ParameterTuple(net.trainable_params())
        self.grad_op = ops.GradOperation(get_by_list=True)

    def construct(self, x):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(x)

# Construct a derivative network
x = Tensor([5], dtype=mstype.float32)
fw = GradNet(Net())(x)

print(fw)
```

```text
(Tensor(shape=[1], dtype=Float32, value= [ 5.00000000e+00]),)
```

## Gradient Value Scaling

You can use the `sens_param` parameter to scale the output value of the network to change the final gradient. Set `sens_param` in `ops.GradOperation` to `True` and determine the scaling index. The dimension must be the same as the output dimension.

```python
class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.net = net
        # Derivative operation
        self.grad_op = ops.GradOperation(sens_param=True)
        # Scale index
        self.grad_wrt_output = Tensor([0.1], dtype=mstype.float32)

    def construct(self, x):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, self.grad_wrt_output)

x = Tensor([6], dtype=mstype.float32)
output = GradNet(Net())(x)

print(output)
```

```text
[0.6]
```

## Stopping Gradient

We can use `stop_gradient` to disable calculation of gradient for certain operators. For example:

```python
from mindspore.ops import stop_gradient

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.w = Parameter(Tensor(np.array([6], np.float32)), name='w')
        self.b = Parameter(Tensor(np.array([1.0], np.float32)), name='b')

    def construct(self, x):
        out = x * self.w + self.b
        # Stops updating the gradient, and out does not contribute to gradient calculations
        out = stop_gradient(out)
        return out

class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.net = net
        self.params = ParameterTuple(net.trainable_params())
        self.grad_op = ops.GradOperation(get_by_list=True)

    def construct(self, x):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(x)

x = Tensor([100], dtype=mstype.float32)
output = GradNet(Net())(x)

print(f"wgrad: {output[0]}\nbgrad: {output[1]}")
```

```text
wgrad: [0.]
bgrad: [0.]
```
