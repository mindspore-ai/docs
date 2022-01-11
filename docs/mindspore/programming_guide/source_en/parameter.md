# Network Parameters

`Ascend` `GPU` `CPU` `Model Development`

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/programming_guide/source_en/parameter.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

## Overview

`Parameter` is a variable tensor, indicating the parameters that need to be updated during network training. The following describes the `Parameter` initialization, attributes, methods, and `ParameterTuple`. The following describes the Parameter initialization, attributes, methods,  ParameterTuple and dependency control.

## Parameters

Parameter is a variable tensor, indicating the parameters that need to be updated during network training.

### Declaration

```python
mindspore.Parameter(default_input, name=None, requires_grad=True, layerwise_parallel=False)
```

- `default_input`: Initialize a `Parameter` object. The input data supports the `Tensor`, `Initializer`, `int`, and `float` types. The `initializer` API can be called to generate the `Initializer` object. When `init` is used to initialize `Tensor`, the `Tensor` only stores the shape and type of the tensor, not the actual data. Therefore, `Tensor` does not occupy any memory, you can call the `init_data` API to convert `Tensor` saved in `Parameter` to the actual data.

- `name`: You can specify a name for each `Parameter` to facilitate subsequent operations and updates. It is recommended to use the default value of `name` when initialize a parameter as one attribute of a cell, otherwise, the parameter name may be different than expected.

- `requires_grad`: update a parameter, set `requires_grad` to `True`.

- `layerwise_parallel`: When `layerwise_parallel` is set to True, this parameter will be filtered out during parameter broadcast and parameter gradient aggregation.

For details about the configuration of distributed parallelism, see <https://www.mindspore.cn/docs/programming_guide/en/r1.6/auto_parallel.html>.

In the following example, `Parameter` objects are built using three different data types. All the three `Parameter` objects need to be updated, and layerwise parallelism is not used.

The code sample is as follows:

```python
import numpy as np
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype
from mindspore.common.initializer import initializer

x = Parameter(default_input=Tensor(np.arange(2*3).reshape((2, 3))), name='x')
y = Parameter(default_input=initializer('ones', [1, 2, 3], mstype.float32), name='y')
z = Parameter(default_input=2.0, name='z')

print(x, "\n\n", y, "\n\n", z)
```

The output is as follows:

```text
 Parameter (name=x, shape=(2, 3), dtype=Int32, requires_grad=True)

 Parameter (name=y, shape=(1, 2, 3), dtype=Float32, requires_grad=True)

 Parameter (name=z, shape=(), dtype=Float32, requires_grad=True)
```

### Attributes

- `inited_param`: returns `Parameter` that stores the actual data.

- `name`: specifies a name for an instantiated `Parameter`.

- `sliced`: specifies whether the data stored in `Parameter` is sharded data in the automatic parallel scenario.

If yes, do not shard the data. Otherwise, determine whether to shard the data based on the network parallel strategy.

- `is_init`: initialization status of `Parameter`. At the GE backend, an `init graph` is required to synchronize data from the host to the device. This parameter specifies whether the data has been synchronized to the device.
  This parameter takes effect only at the GE backend. This parameter is set to False at other backends.

- `layerwise_parallel`: specifies whether `Parameter` supports layerwise parallelism. If yes, parameters are not broadcasted and gradient aggregation is not performed. Otherwise, parameters need to be broadcasted and gradient aggregation is performed.

- `requires_grad`: specifies whether to compute the parameter gradient. If a parameter needs to be trained, the parameter gradient needs to be computed. Otherwise, the parameter gradient does not need to be computed.

- `data`: `Parameter`.

In the following example, `Parameter` is initialized through `Tensor` to obtain its attributes.  

```python
import numpy as np

from mindspore import Tensor, Parameter

x = Parameter(default_input=Tensor(np.arange(2*3).reshape((2, 3))))

print("name: ", x.name, "\n",
      "sliced: ", x.sliced, "\n",
      "is_init: ", x.is_init, "\n",
      "inited_param: ", x.inited_param, "\n",
      "requires_grad: ", x.requires_grad, "\n",
      "layerwise_parallel: ", x.layerwise_parallel, "\n",
      "data: ", x.data)
```

The output is as follows:

```text
name:  Parameter
sliced:  False
is_init:  False
inited_param:  None
requires_grad:  True
layerwise_parallel:  False

data:  Parameter (name=Parameter, shape=(2, 3), dtype=Int64, requires_grad=True)
```

### Methods

- `init_data`: When the network uses the semi-automatic or automatic parallel strategy, and the data input during `Parameter` initialization is `Initializer`, this API can be called to convert the data saved by `Parameter` to `Tensor`.

- `set_data`: sets the data saved by `Parameter`. `Tensor`, `Initializer`, `int`, and `float` can be input for setting.
  When the input parameter `slice_shape` of the method is set to True, the shape of `Parameter` can be changed. Otherwise, the configured shape must be the same as the original shape of `Parameter`.

- `set_param_ps`: controls whether training parameters are trained by using the [Parameter Server](https://www.mindspore.cn/docs/programming_guide/en/r1.6/apply_parameter_server_training.html).

- `clone`: clones `Parameter`. You can specify the parameter name after cloning.

In the following example, `Initializer` is used to initialize `Tensor`, and methods related to `Parameter` are called.  

```python
import numpy as np

from mindspore import Tensor, Parameter
from mindspore import dtype as mstype
from mindspore.common.initializer import initializer

x = Parameter(default_input=initializer('ones', [1, 2, 3], mstype.float32))

print(x)

x_clone = x.clone()
x_clone.name = "x_clone"
print(x_clone)

print(x.init_data())
print(x.set_data(data=Tensor(np.arange(2*3).reshape((1, 2, 3)))))
```

The output is as follows:

```text
Parameter (name=Parameter, shape=(1, 2, 3), dtype=Float32, requires_grad=True)
Parameter (name=x_clone, shape=(1, 2, 3), dtype=Float32, requires_grad=True)
Parameter (name=Parameter, shape=(1, 2, 3), dtype=Float32, requires_grad=True)
Parameter (name=Parameter, shape=(1, 2, 3), dtype=Float32, requires_grad=True)
```

## ParameterTuple

Inherited from `tuple`, `ParameterTuple` is used to store multiple `Parameter` objects. `__new__(cls, iterable)` is used to transfer an iterator for storing `Parameter` for building, and the `clone` API is provided for cloning.

The following example builds a `ParameterTuple` object and clones it.  

```python
import numpy as np
from mindspore import Tensor, Parameter, ParameterTuple
from mindspore import dtype as mstype
from mindspore.common.initializer import initializer

x = Parameter(default_input=Tensor(np.arange(2*3).reshape((2, 3))), name='x')
y = Parameter(default_input=initializer('ones', [1, 2, 3], mstype.float32), name='y')
z = Parameter(default_input=2.0, name='z')
params = ParameterTuple((x, y, z))
params_copy = params.clone("params_copy")
print(params, "\n")
print(params_copy)
```

The output is as follows:

```text
(Parameter (name=x, shape=(2, 3), dtype=Int32, requires_grad=True), Parameter (name=y, shape=(1, 2, 3), dtype=Float32, requires_grad=True), Parameter (name=z, shape=(), dtype=Float32, requires_grad=True))

(Parameter (name=params_copy.x, shape=(2, 3), dtype=Int32, requires_grad=True), Parameter (name=params_copy.y, shape=(1, 2, 3), dtype=Float32, requires_grad=True), Parameter (name=params_copy.z, shape=(), dtype=Float32, requires_grad=True))
```

## Using Encapsulation Operator to Initialize Parameters

Mindspore provides a variety of methods of initializing parameters, and encapsulates parameter initialization functions in some operators. This section will introduce the method of initialization of parameters by operators with parameter initialization function. Taking `Conv2D` operator as an example, it will introduce the initialization of parameters in the network by strings, `Initializer` subclass and custom `Tensor`, etc. `Normal`, a subclass of `Initializer`, is used in the following code examples and can be replaced with any of the subclasses of Initializer in the code examples.

### Character String

Network parameters are initialized using a string. The contents of the string need to be consistent with the name of the `Initializer` subclass(Letters are not case sensitive). Initialization using a string will use the default parameters in the `Initializer` subclass. For example, using the string `Normal` is equivalent to using the `Initializer` subclass `Normal()`. The code sample is as follows:

```python
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import set_seed

set_seed(1)

input_data = Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))
net = nn.Conv2d(3, 64, 3, weight_init='Normal')
output = net(input_data)
print(output)
```

The output is as follows:

```text
[[[[ 3.10382620e-02  4.38603461e-02  4.38603461e-02 ...  4.38603461e-02
     4.38603461e-02  1.38719045e-02]
   [ 3.26051228e-02  3.54298912e-02  3.54298912e-02 ...  3.54298912e-02
     3.54298912e-02 -5.54019120e-03]
   [ 3.26051228e-02  3.54298912e-02  3.54298912e-02 ...  3.54298912e-02
     3.54298912e-02 -5.54019120e-03]
   ...
   [ 3.26051228e-02  3.54298912e-02  3.54298912e-02 ...  3.54298912e-02
     3.54298912e-02 -5.54019120e-03]
   [ 3.26051228e-02  3.54298912e-02  3.54298912e-02 ...  3.54298912e-02
     3.54298912e-02 -5.54019120e-03]
   [ 9.66199022e-03  1.24104535e-02  1.24104535e-02 ...  1.24104535e-02
     1.24104535e-02 -1.38977719e-02]]

  ...

  [[ 3.98553275e-02 -1.35465711e-03 -1.35465711e-03 ... -1.35465711e-03
    -1.35465711e-03 -1.00310734e-02]
   [ 4.38403059e-03 -3.60766202e-02 -3.60766202e-02 ... -3.60766202e-02
    -3.60766202e-02 -2.95619294e-02]
   [ 4.38403059e-03 -3.60766202e-02 -3.60766202e-02 ... -3.60766202e-02
    -3.60766202e-02 -2.95619294e-02]
   ...
   [ 4.38403059e-03 -3.60766202e-02 -3.60766202e-02 ... -3.60766202e-02
    -3.60766202e-02 -2.95619294e-02]
   [ 4.38403059e-03 -3.60766202e-02 -3.60766202e-02 ... -3.60766202e-02
    -3.60766202e-02 -2.95619294e-02]
   [ 1.33139016e-02  6.74417242e-05  6.74417242e-05 ...  6.74417242e-05
     6.74417242e-05 -2.27325838e-02]]]]
```

### Initializer Subclass

`Initializer` subclass is used to initialize network parameters, which is similar to the effect of using string to initialize parameters.  The difference is that using string to initialize parameters uses the default parameter of the `Initializer` subclass. If you want to use the parameters in the `Initializer` subclass, the `Initializer` subclass must be used to initialize the parameters. Taking `Normal(0.2)` as an example, the code sample is as follows:

```python
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import set_seed
from mindspore.common.initializer import Normal

set_seed(1)

input_data = Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))
net = nn.Conv2d(3, 64, 3, weight_init=Normal(0.2))
output = net(input_data)
print(output)
```

The output is as follows:

```text
[[[[ 6.2076533e-01  8.7720710e-01  8.7720710e-01 ...  8.7720710e-01
     8.7720710e-01  2.7743810e-01]
   [ 6.5210247e-01  7.0859784e-01  7.0859784e-01 ...  7.0859784e-01
     7.0859784e-01 -1.1080378e-01]
   [ 6.5210247e-01  7.0859784e-01  7.0859784e-01 ...  7.0859784e-01
     7.0859784e-01 -1.1080378e-01]
   ...
   [ 6.5210247e-01  7.0859784e-01  7.0859784e-01 ...  7.0859784e-01
     7.0859784e-01 -1.1080378e-01]
   [ 6.5210247e-01  7.0859784e-01  7.0859784e-01 ...  7.0859784e-01
     7.0859784e-01 -1.1080378e-01]
   [ 1.9323981e-01  2.4820906e-01  2.4820906e-01 ...  2.4820906e-01
     2.4820906e-01 -2.7795550e-01]]

  ...

  [[ 7.9710668e-01 -2.7093157e-02 -2.7093157e-02 ... -2.7093157e-02
    -2.7093157e-02 -2.0062150e-01]
   [ 8.7680638e-02 -7.2153252e-01 -7.2153252e-01 ... -7.2153252e-01
    -7.2153252e-01 -5.9123868e-01]
   [ 8.7680638e-02 -7.2153252e-01 -7.2153252e-01 ... -7.2153252e-01
    -7.2153252e-01 -5.9123868e-01]
   ...
   [ 8.7680638e-02 -7.2153252e-01 -7.2153252e-01 ... -7.2153252e-01
    -7.2153252e-01 -5.9123868e-01]
   [ 8.7680638e-02 -7.2153252e-01 -7.2153252e-01 ... -7.2153252e-01
    -7.2153252e-01 -5.9123868e-01]
   [ 2.6627803e-01  1.3488382e-03  1.3488382e-03 ...  1.3488382e-03
     1.3488382e-03 -4.5465171e-01]]]]
```

### The Custom of the Tensor

In addition to the above two initialization methods,  when the network wants to use data types that are not available in MindSpore, users can customize `Tensor` to initialize the parameters. The code sample is as follows:

```python
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype

weight = Tensor(np.ones([64, 3, 3, 3]), dtype=mstype.float32)
input_data = Tensor(np.ones([1, 3, 16, 50], dtype=np.float32))
net = nn.Conv2d(3, 64, 3, weight_init=weight)
output = net(input_data)
print(output)
```

The output is as follows:

```text
[[[[12. 18. 18. ... 18. 18. 12.]
   [18. 27. 27. ... 27. 27. 18.]
   [18. 27. 27. ... 27. 27. 18.]
   ...
   [18. 27. 27. ... 27. 27. 18.]
   [18. 27. 27. ... 27. 27. 18.]
   [12. 18. 18. ... 18. 18. 12.]]

  ...

  [[12. 18. 18. ... 18. 18. 12.]
   [18. 27. 27. ... 27. 27. 18.]
   [18. 27. 27. ... 27. 27. 18.]
   ...
   [18. 27. 27. ... 27. 27. 18.]
   [18. 27. 27. ... 27. 27. 18.]
   [12. 18. 18. ... 18. 18. 12.]]]]
```

## Dependency Control

If the result of a function depends on or affects an external state, we consider that the function has side effects, such as a function changing an external global variable, and the result of a function depends on the value of a global variable. If the operator changes the value of the input parameter or the output of the operator depends on the value of the global parameter, we think this is an operator with side effects.

Side effects are classified as memory side effects and IO side effects based on memory properties and IO status. At present, memory side effects are mainly Assign, optimizer operators and so on, IO side effects are mainly Print operators. You can view the operator definition in detail, the memory side effect operator has side_effect_mem properties in the definition, and the IO side effect operator has side_effect_io properties in the definition.

Depend is used for processing dependency operations.In most cases, if the operators have IO or memory side effects, they will be executed according to the user's semantics, and there is no need to use the Depend operator to guarantee the execution order.In some cases, if the two operators A and B do not have sequential dependencies, and A must execute before B, we recommend that you use Depend to specify the order in which they are executed. Here's how to use it:

```python
a = A(x)                --->        a = A(x)
b = B(y)                --->        y = Depend(y, a)
                        --->        b = B(y)
```

Please note that a special set of operators for floating point overflow state detection have hidden side effects, but are not IO side effects or memory side effects. In addition, there are strict sequencing requirements for use, i.e., before using the NPUClearFloatStatus operator, you need to ensure that the NPU AllocFloatStatus has been executed, and before using the NPUGetFloatStatus operator, you need to ensure that the NPUClearFlotStatus has been executed. Because these operators are used less, the current scenario is to keep them defined as side-effect-free in the form of Depend ensuring execution order. These operators are only supported on Ascend. Examples are as follows:

```python
import numpy as np
from mindspore.common.tensor import Tensor
from mindspore import ops, context

context.set_context(device_target="Ascend")

npu_alloc_status = ops.NPUAllocFloatStatus()
npu_get_status = ops.NPUGetFloatStatus()
npu_clear_status = ops.NPUClearFloatStatus()
x = Tensor(np.ones([3, 3]).astype(np.float32))
y = Tensor(np.ones([3, 3]).astype(np.float32))
init = npu_alloc_status()
sum_ = ops.Add()(x, y)
product = ops.MatMul()(x, y)
init = ops.depend(init, sum_)
init = ops.depend(init, product)
get_status = npu_get_status(init)
sum_ = ops.depend(sum_, get_status)
product = ops.depend(product, get_status)
out = ops.Add()(sum_, product)
init = ops.depend(init, out)
clear = npu_clear_status(init)
out = ops.depend(out, clear)
print(out)
```

The output is as follows:

```text
[[5. 5. 5.]
 [5. 5. 5.]
 [5. 5. 5.]]
```

Specific usage methods can refer to the implementation of [start_overflow_check functions](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/python/mindspore/nn/wrap/loss_scale.py) in the overflow detection logic.
