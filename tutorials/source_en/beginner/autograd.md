# Automatic Differentiation

<a href="https://gitee.com/mindspore/docs/blob/r1.7/tutorials/source_en/beginner/autograd.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

Automatic differentiation can calculate a derivative value of a derivative function at a certain point, which is a generalization of backpropagation algorithms. The main problem solved by automatic differentiation is to decompose a complex mathematical operation into a series of simple basic operations. This function shields a large number of derivative details and processes from users, greatly reducing the threshold for using the framework.

MindSpore uses `ops.GradOperation` to calculate the first-order derivative. The `ops.GradOperation` attributes are as follows:

+ `get_all`: determines whether to derive the input parameters. The default value is False.
+ `get_by_list`: determines whether to derive the weight parameters. The default value is False.
+ `sens_param`: determines whether to scale the output value of the network to change the final gradient. The default value is False.

This chapter uses `ops.GradOperation` in MindSpore to find first-order derivatives of the function $f(x)=wx+b$.

## First-order Derivative of the Input

Define the formula before deriving the input:

$$f(x)=wx+b \tag {1} $$

The example code below is an expression of Equation (1). Since MindSpore is functionally programmed, all expressions of computational formulas are represented as functions.

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

Define the derivative class `GradNet`. In the `__init__` function, define the `self.net` and `ops.GradOperation` networks. In the `construct` function, compute the derivative of `self.net`. The following formula (2) is generated in MindSpore:

$$f^{'}(x)=w\tag {2}$$

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

Finally, the weight parameter is defined as w, and a first-order derivative is found for the input parameter x in the input formula (1). According to the running result, the input in formula (1) is 6, that is:

$$f(x)=wx+b=6*x+1 \tag {3}$$

Derive the above equation:

$$f^{'}(x)=w=6 \tag {4}$$

```python
x = Tensor([100], dtype=mstype.float32)
output = GradNet(Net())(x)

print(output)
```

```text
[6.]
```

MindSpore calculates the first-order derivative using `ops.GradOperation (get_all=False, get_by_lsit=False, sens_param=False)`. If `get_all` is set to `False`, the derivative of only the first input is calculated. If `get_all` is set to `True`, the derivative of all inputs is calculated.

## First-order Derivative of the Weight

To compute weight derivatives, you need to set `get_by_list` in `ops.GradOperation` to `True`.

```python
from mindspore import ParameterTuple

class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.net = net
        self.params = ParameterTuple(net.trainable_params())
        self.grad_op = ops.GradOperation(get_by_list=True)  # Set the first-order derivative of the weight parameters.

    def construct(self, x):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(x)
```

Next, derive the function:

```python
# Perform a derivative calculation on the function.
x = Tensor([100], dtype=mstype.float32)
fx = GradNet(Net())(x)

# Print the result.
print(f"wgrad: {fx[0]}\nbgrad: {fx[1]}")
```

```text
wgrad: [100.]
bgrad: [1.]
```

If derivation is not required for some weights, set `requires_grad` to `False` when defining the derivation network and declaring the corresponding weight parameters.

```python
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

# Construct a derivative network.
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
        # Derivative operation.
        self.grad_op = ops.GradOperation(sens_param=True)
        # Scale an index.
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

## Stopping Gradient Calculation

You can use `ops.stop_gradient` to stop calculating gradients. The following is an example:

```python
from mindspore.ops import stop_gradient

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.w = Parameter(Tensor(np.array([6], np.float32)), name='w')
        self.b = Parameter(Tensor(np.array([1.0], np.float32)), name='b')

    def construct(self, x):
        out = x * self.w + self.b
        # Stop updating the gradient. The out does not contribute to gradient calculations.
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