# Weight update and Dependency Control

<!-- TOC -->

- [Weight update and Dependency Control](#weight-update-and-dependency-control)
    - [Overview](#overview)
    - [Attributes](#attributes)
    - [Methods](#methods)
    - [ParameterTuple](#parametertuple)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/parameter.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

`Parameter` is a variable tensor, indicating the parameters that need to be updated during network training. The following describes the `Parameter` initialization, attributes, methods, and `ParameterTuple`.

## Attributes

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

The following information is displayed:

```text
name:  Parameter
sliced:  False
is_init:  False
inited_param:  None
requires_grad:  True
layerwise_parallel:  False

data:  Parameter (name=Parameter, shape=(2, 3), dtype=Int64, requires_grad=True)
```

## Methods

- `init_data`: When the network uses the semi-automatic or automatic parallel strategy, and the data input during `Parameter` initialization is `Initializer`, this API can be called to convert the data saved by `Parameter` to `Tensor`.

- `set_data`: sets the data saved by `Parameter`. `Tensor`, `Initializer`, `int`, and `float` can be input for setting.
  When the input parameter `slice_shape` of the method is set to True, the shape of `Parameter` can be changed. Otherwise, the configured shape must be the same as the original shape of `Parameter`.

- `set_param_ps`: controls whether training parameters are trained by using the [Parameter Server](https://www.mindspore.cn/docs/programming_guide/en/master/apply_parameter_server_training.html).

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

The following information is displayed:

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

The following information is displayed:

```text
(Parameter (name=x, shape=(2, 3), dtype=Int32, requires_grad=True), Parameter (name=y, shape=(1, 2, 3), dtype=Float32, requires_grad=True), Parameter (name=z, shape=(), dtype=Float32, requires_grad=True))

(Parameter (name=params_copy.x, shape=(2, 3), dtype=Int32, requires_grad=True), Parameter (name=params_copy.y, shape=(1, 2, 3), dtype=Float32, requires_grad=True), Parameter (name=params_copy.z, shape=(), dtype=Float32, requires_grad=True))
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

Please note that a special set of operators for floating point overflow state detection have hidden side effects, but are not IO side effects or memory side effects. In addition, there are strict sequencing requirements for use, i.e., before using the NPUClearFloatStatus operator, you need to ensure that the NPU AllocFloatStatus has been executed, and before using the NPUGetFloatStatus operator, you need to ensure that the NPUClearFlotStatus has been executed. Because these operators are used less, the current scenario is to keep them defined as side-effect-free in the form of Depend ensuring execution order. Examples are as follows:

```python
import mindspore.ops as ops
self.alloc_status = ops.operations.NPUAllocFloatStatus()
self.get_status = ops.operations.NPUGetFloatStatus()
self.clear_status = ops.operations.NPUClearFloatStatus()
...
init = self.alloc_status()
init = ops.functional.Depend(init, input)
clear_status = self.clear_status(init)
input = ops.functional.Depend(input, clear_status)
output = Compute(input)
init = ops.functional.Depend(init, output)
get_status = self.get_status(init)
```

Specific usage methods can refer to the implementation of [start_overflow_check functions](https://gitee.com/mindspore/mindspore/blob/master/mindspore/nn/wrap/loss_scale.py) in the overflow detection logic.
