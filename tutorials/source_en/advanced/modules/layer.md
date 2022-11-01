<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/advanced/modules/layer.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

# Cell and Parameter

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

#### Modify Parameter Values Directly

Parameter is a special kind of Tensor, so its value can be modified by using the Tensor index modification.

```python
net.b[0] = 1.
print(net.b.asnumpy())
```

```text
[ 1.         -0.36789745  0.0946381 ]
```

#### Overriding the Modified Parameter Values

The `Parameter.set_data` method can be called to override the Parameter by using a Tensor with the same Shape. This method is commonly used for [Cell traversal initialization](https://www.mindspore.cn/tutorials/en/master/advanced/modules/initializer.html) by using Initializer.

```python
net.b.set_data(Tensor([3, 4, 5]))
print(net.b.asnumpy())
```

```text
[3. 4. 5.]
```

#### Modifying Parameter Values During Runtime

The main role of parameters is to update their values during model training, which involves parameter modification during runtime after backward propagation to obtain gradients, or when untrainable parameters need to be updated. Due to the compiled design of MindSpore's [computational graph](https://www.mindspore.cn/tutorials/en/master/advanced/compute_graph.html), it is necessary at this point to use the `mindspore.ops.assign` interface to assign parameters. This method is commonly used in [Custom Optimizer](https://www.mindspore.cn/tutorials/en/master/advanced/modules/optimizer.html) scenarios. The following is a simple sample modification of parameter values during runtime:

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

## Custom Neural Network Layers

Normally, the neural network layer interface and function interface provided by MindSpore can meet the model construction requirements, but since the AI field is constantly updating, it is possible to encounter new network structures without built-in modules. At this point, we can customize the neural network layer through the function interface provided by MindSpore, Primitive operator, and can use the `Cell.bprop` method to customize the reverse. The following are the details of each of the three customization methods.

### Constructing Neural Network Layers by Using the Function Interface

MindSpore provides a large number of basic function interfaces, which can be used to construct complex Tensor operations, encapsulated as neural network layers. The following is an example of `Threshold` with the following equation:

$$
y =\begin{cases}
   x, &\text{ if } x > \text{threshold} \\
   \text{value}, &\text{ otherwise }
   \end{cases}
$$

It can be seen that `Threshold` determines whether the value of the Tensor is greater than the `threshold` value, keeps the value whose judgment result is `True`, and replaces the value whose judgment result is `False`. Therefore, the corresponding implementation is as follows:

```python
class Threshold(nn.Cell):
    def __init__(self, threshold, value):
        super().__init__()
        self.threshold = threshold
        self.value = value

    def construct(self, inputs):
        cond = ops.gt(inputs, self.threshold)
        value = ops.fill(inputs.dtype, inputs.shape, self.value)
        return ops.select(cond, inputs, value)
```

Here `ops.gt`, `ops.fill`, and `ops.select` are used to implement judgment and replacement respectively. The following custom `Threshold` layer is implemented:

```python
m = Threshold(0.1, 20)
inputs = mindspore.Tensor([0.1, 0.2, 0.3], mindspore.float32)
m(inputs)
```

```text
Tensor(shape=[3], dtype=Float32, value= [ 2.00000000e+01,  2.00000003e-01,  3.00000012e-01])
```

It can be seen that `inputs[0] = threshold`, so it is replaced with `20`.

### Constructing Neural Network Layers by Using the Primitive Interface

When the function interface cannot satisfy the complex function implementation, but MindSpore has implemented Primitive operators, we need to be able to use Primitive operators to encapsulate the neural network layer by ourselves as needed. Taking the `Upsample` layer as an example, its function is to upsample 1D, 2D and 3D data. The current `mindspore.nn` does not yet provide the upsample interface, but it has provided `ResizeNearestNeighbor`, `ResizeLinear1D`, `ResizeBilinearV2` and other Primitive operators that can achieve its function. The corresponding package implementation is as follows:

```python
from mindspore.ops.operations.image_ops import ResizeBilinearV2, ResizeLinear1D
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops import constexpr

@constexpr
def _check_scale_factor(shape, scale_factor):
    if isinstance(scale_factor, tuple) and len(scale_factor) != len(shape[2:]):
        raise ValueError(f"the number of 'scale_fator' must match to inputs.shape[2:], "
                         f"but get scale_factor={scale_factor}, inputs.shape[2:]={shape[2:]}")

def _interpolate_output_shape(shape, scales, sizes, mode):
    """calculate output shape"""
    if sizes is not None:
        if mode == "nearest":
            return sizes
        return Tensor(sizes)

    ret = ()
    for i in range(len(shape[2:])):
        if isinstance(scales, float):
            out_i = int(scales * shape[i+2])
        else:
            out_i = int(scales[i] * shape[i+2])
        ret = ret + (out_i,)
    if mode == "nearest":
        return ret
    return Tensor(ret)

class Upsample(nn.Cell):
    def __init__(self, size=None, scale_factor=None,
                 mode='nearest', align_corners=False):
        super().__init__()
        if mode not in ['nearest', 'linear', 'bilinear']:
            raise ValueError(f'do not support mode :{mode}.')
        if size and scale_factor:
            raise ValueError(f"can not set 'size' and 'scale_fator' at the same time.")
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def construct(self, inputs):
        inputs_shape = inputs.shape
        _check_scale_factor(inputs_shape, self.scale_factor)
        sizes = _interpolate_output_shape(inputs_shape, self.scale_factor, self.size, self.mode)
        if self.mode == 'nearest':
            interpolate = _get_cache_prim(ops.ResizeNearestNeighbor)(sizes, self.align_corners)
            return interpolate(inputs)
        if self.mode == 'linear':
            interpolate = _get_cache_prim(ResizeLinear1D)('align_corners' if self.align_corners else 'half_pixel')
            return interpolate(inputs, sizes)
        if self.mode == 'bilinear':
            interpolate = _get_cache_prim(ResizeBilinearV2)(self.align_corners, not self.align_corners)
            return interpolate(inputs, sizes)
        return inputs
```

`constexpr` is a [constant expression modifier](https://www.mindspore.cn/tutorials/experts/en/master/network/constexpr.html) used to support the execution of static constant statements after the computational graph is compiled. `_get_cache_prim` is a Primitive caching mechanism, which can effectively improve the instantiation execution speed of Primitive operators.

The following test encapsulates the `Upsample` layer and you can see that the execution results are correct.

```python
inputs = ops.arange(1, 5).astype(mindspore.float32).view(1, 1, 2, 2)
print(inputs)

m = Upsample(scale_factor=2, mode='nearest')
outputs = m(inputs)
print(outputs)
```

```text
[[[[1. 2.]
   [3. 4.]]]]
[[[[1. 1. 2. 2.]
   [1. 1. 2. 2.]
   [3. 3. 4. 4.]
   [3. 3. 4. 4.]]]]
```

### Custom Cell Reverse

In special scenarios, we not only need to customize the forward logic of the neural network layer, but also want to manually control the computation of its reverse, which we can define through the `Cell.bprop` interface. The function will be used in scenarios such as new neural network structure design and backward propagation speed optimization. In the following, we take `Dropout2d` as an example to introduce custom Cell reverse.

```python
class Dropout2d(nn.Cell):
    def __init__(self, keep_prob):
        super().__init__()
        self.keep_prob = keep_prob
        self.dropout2d = ops.Dropout2D(keep_prob)

    def construct(self, x):
        return self.dropout2d(x)

    def bprop(self, x, out, dout):
        _, mask = out
        dy, _ = dout
        if self.keep_prob != 0:
            dy = dy * (1 / self.keep_prob)
        dy = mask.astype(mindspore.float32) * dy
        return (dy.astype(x.dtype), )

dropout_2d = Dropout2d(0.8)
dropout_2d.bprop_debug = True
```

The `bprop` method has three separate input parameters:

- *x*: Forward input. When there are multiple forward inputs, the same number of inputs are required.
- *out*: Forward input.
- *dout*: When backward propagation is performed, the current Cell executes the previous reverse result.

Generally we need to calculate the reverse result according to the reverse derivative formula based on the forward output and the reverse result of the front layer, and return it. The reverse calculation of `Dropout2d` requires masking the reverse result of the front layer based on the `mask` matrix of the forward output, and then scaling according to `keep_prob`. The final implementation can get the correct calculation result.
