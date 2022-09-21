# Network Arguments

<a href="https://gitee.com/mindspore/docs/blob/r1.9/tutorials/source_en/advanced/modules/parameter.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png"></a>

MindSpore provides initialization modules for parameters and network arguments. You can initialize network arguments by encapsulating operators to call character strings, Initializer subclasses, or customized tensors.

In the following figure, a blue box indicates a specific execution operator, and a green box indicates a tensor. As the data in the neural network model, the tensor continuously flows in the network, including the data input of the network model and the input and output data of the operator. A red box indicates a parameter which is used as a attribute of the network model or operators in the model or as an intermediate parameter and temporary parameter generated in the backward graph.

![parameter.png](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/tutorials/source_zh_cn/advanced/modules/images/parameter.png)

The following describes the data type (`dtype`), parameter (`Parameter`), parameter tuple (`ParameterTuple`), network initialization method, and network argument update.

## dtype

MindSpore tensors support different data types, including int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, and Boolean. These data types correspond to those of NumPy. For details about supported data types, visit [mindspore.dtype](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore.html#mindspore.dtype).

In the computation process of MindSpore, the int data type in Python is converted into the defined int64 type, and the float data type is converted into the defined float32 type.

In the following code, the data type of MindSpore is int32.

```python
import mindspore as ms

data_type = ms.int32
print(data_type)
```

```text
    Int32
```

### Data Type Conversion API

MindSpore provides the following APIs for conversion between NumPy data types and Python built-in data types:

- `dtype_to_nptype`: converts the data type of MindSpore to the corresponding data type of NumPy.
- `dtype_to_pytype`: converts the data type of MindSpore to the corresponding built-in data type of Python.
- `pytype_to_dtype`: converts the built-in data type of Python to the corresponding data type of MindSpore.

The following code implements the conversion between different data types and prints the converted type.

```python
import mindspore as ms

np_type = ms.dtype_to_nptype(ms.int32)
ms_type = ms.pytype_to_dtype(int)
py_type = ms.dtype_to_pytype(ms.float64)

print(np_type)
print(ms_type)
print(py_type)
```

```text
    <class 'numpy.int32'>
    Int64
    <class 'float'>
```

## Parameter

A [Parameter](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore/mindspore.Parameter.html#mindspore.Parameter) of MindSpore indicates an argument that needs to be updated during network training. For example, the most common parameters of the `nn.conv` operator during forward computation include `weight` and `bias`. During backward graph build and backward propagation computation, many intermediate parameters are generated to temporarily store first-step information and intermediate output values.

### Parameter Initialization

There are many methods for initializing `Parameter`, which can receive different data types such as `Tensor` and `Initializer`.

- `default_input`: input data. Four data types are supported: `Tensor`, `Initializer`, `int`, and `float`.
- `name`: name of a parameter, which is used to distinguish the parameter from other parameters on the network.
- `requires_grad`: indicates whether to compute the argument gradient during network training. If the argument gradient does not need to be computed, set `requires_grad` to `False`.

In the following sample code, the `int` or `float` data type is used to directly create a parameter:

```python
import mindspore as ms

x = ms.Parameter(default_input=2.0, name='x')
y = ms.Parameter(default_input=5.0, name='y')
z = ms.Parameter(default_input=5, name='z', requires_grad=False)

print(type(x))
print(x, "value:", x.asnumpy())
print(y, "value:", y.asnumpy())
print(z, "value:", z.asnumpy())
```

```text
    <class 'mindspore.common.parameter.ParameterTensor'>
    Parameter (name=x, shape=(), dtype=Float32, requires_grad=True) value: 2.0
    Parameter (name=y, shape=(), dtype=Float32, requires_grad=True) value: 5.0
    Parameter (name=z, shape=(), dtype=Int32, requires_grad=False) value: 5
```

In the following code, a MindSpore `Tensor` is used to create a parameter:

```python
import numpy as np
import mindspore as ms

my_tensor = ms.Tensor(np.arange(2 * 3).reshape((2, 3)))
x = ms.Parameter(default_input=my_tensor, name="tensor")

print(x)
```

```text
    Parameter (name=tensor, shape=(2, 3), dtype=Int64, requires_grad=True)
```

In the following code example, `Initializer` is used to create a parameter:

```python
from mindspore.common.initializer import initializer as init
import mindspore as ms

x = ms.Parameter(default_input=init('ones', [1, 2, 3], ms.float32), name='x')
print(x)
```

```text
    Parameter (name=x, shape=(1, 2, 3), dtype=Float32, requires_grad=True)
```

### Attribute

The default attributes of a `Parameter` include `name`, `shape`, `dtype`, and `requires_grad`.

The following example describes how to initialize a `Parameter` by using a `Tensor` and obtain the attributes of the `Parameter`. The sample code is as follows:

```python
my_tensor = ms.Tensor(np.arange(2 * 3).reshape((2, 3)))
x = ms.Parameter(default_input=my_tensor, name="x")

print("x: ", x)
print("x.data: ", x.data)
```

```text
    x:  Parameter (name=x, shape=(2, 3), dtype=Int64, requires_grad=True)
    x.data:  Parameter (name=x, shape=(2, 3), dtype=Int64, requires_grad=True)
```

### Parameter Operations

1. `clone`: clones a tensor `Parameter`. After the cloning is complete, you can specify a new name for the new `Parameter`.

    ```python
    x = ms.Parameter(default_input=init('ones', [1, 2, 3], ms.float32))
    x_clone = x.clone()
    x_clone.name = "x_clone"

    print(x)
    print(x_clone)
    ```

    ```text
        Parameter (name=Parameter, shape=(1, 2, 3), dtype=Float32, requires_grad=True)
        Parameter (name=x_clone, shape=(1, 2, 3), dtype=Float32, requires_grad=True)
    ```

2. `set_data`: modifies the data or `shape` of the `Parameter`.

    The `set_data` method has two input parameters: `data` and `slice_shape`. The `data` indicates the newly input data of the `Parameter`. The `slice_shape` indicates whether to change the `shape` of the `Parameter`. The default value is False.

    ```python
    x = ms.Parameter(ms.Tensor(np.ones((1, 2)), ms.float32), name="x", requires_grad=True)
    print(x, x.asnumpy())

    y = x.set_data(ms.Tensor(np.zeros((1, 2)), ms.float32))
    print(y, y.asnumpy())

    z = x.set_data(ms.Tensor(np.ones((1, 4)), ms.float32), slice_shape=True)
    print(z, z.asnumpy())
    ```

    ```text
        Parameter (name=x, shape=(1, 2), dtype=Float32, requires_grad=True) [[1. 1.]]
        Parameter (name=x, shape=(1, 2), dtype=Float32, requires_grad=True) [[0. 0.]]
        Parameter (name=x, shape=(1, 4), dtype=Float32, requires_grad=True) [[1. 1. 1. 1.]]
    ```

3. `init_data`: In parallel scenarios, the shape of a argument changes. You can call the `init_data` method of `Parameter` to obtain the original data.

    ```python
    x = ms.Parameter(ms.Tensor(np.ones((1, 2)), ms.float32), name="x", requires_grad=True)

    print(x.init_data(), x.init_data().asnumpy())
    ```

    ```text
        Parameter (name=x, shape=(1, 2), dtype=Float32, requires_grad=True) [[1. 1.]]
    ```

### Updating Parameters

MindSpore provides the network argument update function. You can use `nn.ParameterUpdate` to update network arguments. The input argument type must be tensor, and the tensor `shape` must be the same as the original network argument `shape`.

The following is an example of updating the weight arguments of a network:

```python
import numpy as np
import mindspore as ms
from mindspore import nn

# Build a network.
network = nn.Dense(3, 4)

# Obtain the weight argument of a network.
param = network.parameters_dict()['weight']
print("Parameter:\n", param.asnumpy())

# Update the weight argument.
update = nn.ParameterUpdate(param)
weight = ms.Tensor(np.arange(12).reshape((4, 3)), ms.float32)
output = update(weight)
print("Parameter update:\n", output)
```

```text
    Parameter:
     [[-0.0164615  -0.01204428 -0.00813806]
     [-0.00270927 -0.0113328  -0.01384139]
     [ 0.00849093  0.00351116  0.00989969]
     [ 0.00233028  0.00649209 -0.0021333 ]]
    Parameter update:
     [[ 0.  1.  2.]
     [ 3.  4.  5.]
     [ 6.  7.  8.]
     [ 9. 10. 11.]]
```

## Parameter Tuple

The [ParameterTuple](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore/mindspore.ParameterTuple.html#mindspore.ParameterTuple) is used to store multiple `Parameter`s. It is inherited from the `tuple` and provides the clone function.

The following example describes how to create a `ParameterTuple`:

```python
import numpy as np
import mindspore as ms
from mindspore.common.initializer import initializer

# Create.
x = ms.Parameter(default_input=ms.Tensor(np.arange(2 * 3).reshape((2, 3))), name="x")
y = ms.Parameter(default_input=initializer('ones', [1, 2, 3], ms.float32), name='y')
z = ms.Parameter(default_input=2.0, name='z')
params = ms.ParameterTuple((x, y, z))

# Clone from params and change the name to "params_copy".
params_copy = params.clone("params_copy")

print(params)
print(params_copy)
```

```text
    (Parameter (name=x, shape=(2, 3), dtype=Int64, requires_grad=True), Parameter (name=y, shape=(1, 2, 3), dtype=Float32, requires_grad=True), Parameter (name=z, shape=(), dtype=Float32, requires_grad=True))
    (Parameter (name=params_copy.x, shape=(2, 3), dtype=Int64, requires_grad=True), Parameter (name=params_copy.y, shape=(1, 2, 3), dtype=Float32, requires_grad=True), Parameter (name=params_copy.z, shape=(), dtype=Float32, requires_grad=True))
```

## Initializing Network Arguments

MindSpore provides multiple network argument initialization modes and encapsulates the argument initialization function in some operators. The following uses the `Conv2d` operator as an example to describe how to use the `Initializer` subclass, character string, and customized `Tensor` to initialize network arguments.

### Initializer

Use `Initializer` to initialize network arguments. The sample code is as follows:

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms
from mindspore.common import initializer as init

ms.set_seed(1)

input_data = ms.Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))
# Convolutional layer. The number of input channels is 3, the number of output channels is 64, the size of the convolution kernel is 3 x 3, and the weight argument is a random number generated in normal distribution.
net = nn.Conv2d(3, 64, 3, weight_init=init.Normal(0.2))
# Network output
output = net(input_data)
```

### Character String Initialization

Use a character string to initialize network arguments. The content of the character string must be the same as the `Initializer` name (case insensitive). If the character string is used for initialization, the default arguments in the `Initializer` class are used. For example, using the character string `Normal` is equivalent to using `Normal()` of `Initializer`. The following is an example:

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms

ms.set_seed(1)

input_data = ms.Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))
net = nn.Conv2d(3, 64, 3, weight_init='Normal')
output = net(input_data)
```

### Tensor Initialization

You can also customize a `Tensor` to initialize the arguments of operators in the network model. The sample code is as follows:

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms

init_data = ms.Tensor(np.ones([64, 3, 3, 3]), dtype=ms.float32)
input_data = ms.Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))

net = nn.Conv2d(3, 64, 3, weight_init=init_data)
output = net(input_data)
```
