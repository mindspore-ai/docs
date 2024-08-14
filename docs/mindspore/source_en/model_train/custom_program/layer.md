# Cell and Parameter

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/model_train/custom_program/layer.md)

Cell, as the basic unit of neural network construction, corresponds to the concept of neural network layer, and the abstract encapsulation of Tensor computation operation can represent the neural network structure more accurately and clearly. In addition to the basic Tensor computation flow definition, the neural network layer contains functions such as parameter management and state management. Parameter is the core of neural network training and is usually used as an internal member variable of the neural network layer. In this section, we systematically introduce parameters, neural network layers and their related usage.

## Parameter

Parameter is a special class of Tensor, which is a variable whose value can be updated during model training. MindSpore provides the `mindspore.Parameter` class for Parameter construction. In order to distinguish between Parameter for different purposes, two different categories of Parameter are defined below. In order to distinguish between Parameter for different purposes, two different categories of Parameter are defined below:

- Trainable parameter. Tensor that is updated after the gradient is obtained according to the backward propagation algorithm during model training, and `required_grad` needs to be set to `True`.
- Untrainable parameters. Tensor that does not participate in backward propagation needs to update values (e.g. `mean` and `var` variables in BatchNorm), when `requires_grad` needs to be set to `False`.

> Parameter is set to `required_grad=True` by default.

We construct a simple fully-connected layer as follows:

```python
import numpy as np
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import Tensor, Parameter

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.w = Parameter(Tensor(np.random.randn(5, 3), mindspore.float32), name='w') # weight
        self.b = Parameter(Tensor(np.random.randn(3,), mindspore.float32), name='b') # bias

    def construct(self, x):
        z = ops.matmul(x, self.w) + self.b
        return z

net = Network()
```

In the `__init__` method of `Cell`, we define two parameters `w` and `b` and configure `name` for namespace management. Use `self.attr` in the `construct` method to call directly to participate in Tensor operations.

### Obtaining Parameter

After constructing the neural network layer by using Cell+Parameter, we can use various methods to obtain the Parameter managed by Cell.

#### Obtaining a Single Parameter

To get a particular parameter individually, just call a member variable of a Python class directly.

```python
print(net.b.asnumpy())
```

```text
[-1.2192779  -0.36789745  0.0946381 ]
```

#### Obtaining a Trainable Parameter

Trainable parameters can be obtained by using the `Cell.trainable_params` method, and this interface is usually called when configuring the optimizer.

```python
print(net.trainable_params())
```

```text
[Parameter (name=w, shape=(5, 3), dtype=Float32, requires_grad=True), Parameter (name=b, shape=(3,), dtype=Float32, requires_grad=True)]
```

#### Obtaining All Parameters

Use the `Cell.get_parameters()` method to get all parameters, at which point a Python iterator will be returned.

```python
print(type(net.get_parameters()))
```

```text
<class 'generator'>
```

Or you can call `Cell.parameters_and_names` to return the parameter names and parameters.

```python
for name, param in net.parameters_and_names():
    print(f"{name}:\n{param.asnumpy()}")
```

```text
w:
[[ 4.15680408e-02 -1.20311625e-01  5.02573885e-02]
 [ 1.22175144e-04 -1.34980649e-01  1.17642188e+00]
 [ 7.57667869e-02 -1.74758151e-01 -5.19092619e-01]
 [-1.67846107e+00  3.27240258e-01 -2.06452996e-01]
 [ 5.72323874e-02 -8.27963874e-02  5.94243526e-01]]
b:
[-1.2192779  -0.36789745  0.0946381 ]
```

### Modifying the Parameter

#### Modifying Parameter Values Directly

Parameter is a special kind of Tensor, so its value can be modified by using the Tensor index modification.

```python
net.b[0] = 1.
print(net.b.asnumpy())
```

```text
[ 1.         -0.36789745  0.0946381 ]
```

#### Overriding the Modified Parameter Values

The `Parameter.set_data` method can be called to override the Parameter by using a Tensor with the same Shape. This method is commonly used for [Cell traversal initialization](https://www.mindspore.cn/docs/en/master/model_train/custom_program/initializer.html) by using Initializer.

```python
net.b.set_data(Tensor([3, 4, 5]))
print(net.b.asnumpy())
```

```text
[3. 4. 5.]
```

#### Modifying Parameter Values During Runtime

The main role of parameters is to update their values during model training, which involves parameter modification during runtime after backward propagation to obtain gradients, or when untrainable parameters need to be updated. Due to the compiled design of MindSpore's [Accelerating with Static Graphs](https://www.mindspore.cn/tutorials/en/master/beginner/accelerate_with_static_graph.html), it is necessary at this point to use the `mindspore.ops.assign` interface to assign parameters. This method is commonly used in [Custom Optimizer](https://www.mindspore.cn/docs/en/master/model_train/custom_program/optimizer.html) scenarios. The following is a simple sample modification of parameter values during runtime:

```python
import mindspore as ms

@ms.jit
def modify_parameter():
    b_hat = ms.Tensor([7, 8, 9])
    ops.assign(net.b, b_hat)
    return True

modify_parameter()
print(net.b.asnumpy())
```

```text
[7. 8. 9.]
```

### Parameter Tuple

ParameterTuple, variable tuple, used to store multiple Parameter, is inherited from tuple tuples, and provides cloning function.

The following example provides the ParameterTuple creation method:

```python
from mindspore.common.initializer import initializer
from mindspore import ParameterTuple
# Creation
x = Parameter(default_input=ms.Tensor(np.arange(2 * 3).reshape((2, 3))), name="x")
y = Parameter(default_input=initializer('ones', [1, 2, 3], ms.float32), name='y')
z = Parameter(default_input=2.0, name='z')
params = ParameterTuple((x, y, z))

# Clone from params and change the name to "params_copy"
params_copy = params.clone("params_copy")

print(params)
print(params_copy)
```

```text
(Parameter (name=x, shape=(2, 3), dtype=Int64, requires_grad=True), Parameter (name=y, shape=(1, 2, 3), dtype=Float32, requires_grad=True), Parameter (name=z, shape=(), dtype=Float32, requires_grad=True))
(Parameter (name=params_copy.x, shape=(2, 3), dtype=Int64, requires_grad=True), Parameter (name=params_copy.y, shape=(1, 2, 3), dtype=Float32, requires_grad=True), Parameter (name=params_copy.z, shape=(), dtype=Float32, requires_grad=True))
```

## Cell Training State Change

Some Tensor operations in neural networks do not behave the same during training and inference, e.g., `nn.Dropout` performs random dropout during training but not during inference, and `nn.BatchNorm` requires updating the `mean` and `var` variables during training and fixing their values unchanged during inference. So we can set the state of the neural network through the `Cell.set_train` interface.

When `set_train` is set to True, the neural network state is `train`, and the default value of `set_train` interface is `True`:

```python
net.set_train()
print(net.phase)
```

```text
train
```

When `set_train` is set to False, the neural network state is `predict`:

```python
net.set_train(False)
print(net.phase)
```

```text
predict
```
