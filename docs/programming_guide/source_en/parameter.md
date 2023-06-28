# Parameter

<a href="https://gitee.com/mindspore/docs/blob/r1.1/docs/programming_guide/source_en/parameter.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Overview

`Parameter` is a variable tensor, indicating the parameters that need to be updated during network training. The following describes the `Parameter` initialization, attributes, methods, and `ParameterTuple`.

## Initialization

```python
mindspore.Parameter(default_input, name=None, requires_grad=True, layerwise_parallel=False)
```

Initialize a `Parameter` object. The input data supports the `Tensor`, `Initializer`, `int`, and `float` types.

The `initializer` API can be called to generate the `Initializer` object.

When the network uses the semi-automatic or automatic parallel strategy and `Initializer` is used to initialize `Parameter`, `Parameter` does not store `Tensor` but `MetaTensor`.

Different from `Tensor`, `MetaTensor` only stores the shape and type of the tensor, not the actual data. Therefore, `MetaTensor` does not occupy any memory, you can call the `init_data` API to convert `MetaTensor` saved in `Parameter` to `Tensor`.

You can specify a name for each `Parameter` to facilitate subsequent operations and updates. It is recommended to use the default value of `name` when initialize a parameter as one attribute of a cell, otherwise, the parameter name may be different than expected.

To update a parameter, set `requires_grad` to `True`.

When `layerwise_parallel` is set to True, this parameter will be filtered out during parameter broadcast and parameter gradient aggregation.

For details about the configuration of distributed parallelism, see <https://www.mindspore.cn/doc/programming_guide/en/r1.1/auto_parallel.html>.

In the following example, `Parameter` objects are built using three different data types. All the three `Parameter` objects need to be updated, and layerwise parallelism is not used.  

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

The following information is displayed:

```text
Parameter (name=x)

Parameter (name=y)

Parameter (name=z)
```

## Attributes

- `inited_param`: returns `Parameter` that stores the actual data. If `Parameter` originally stores `MetaTensor`, the data will be converted into `Tensor`.

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

data:  Parameter (name=Parameter)
```

## Methods

- `init_data`: When the network uses the semi-automatic or automatic parallel strategy, and the data input during `Parameter` initialization is `Initializer`, this API can be called to convert the data saved by `Parameter` to `Tensor`.

- `set_data`: sets the data saved by `Parameter`. `Tensor`, `Initializer`, `int`, and `float` can be input for setting.
  When the input parameter `slice_shape` of the method is set to True, the shape of `Parameter` can be changed. Otherwise, the configured shape must be the same as the original shape of `Parameter`.

- `set_param_ps`: controls whether training parameters are trained by using the [Parameter Server](https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/apply_parameter_server_training.html).

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
Parameter (name=Parameter)
Parameter (name=x_clone)
Parameter (name=Parameter)
Parameter (name=Parameter)
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
(Parameter (name=x), Parameter (name=y), Parameter (name=z))

(Parameter (name=params_copy.x), Parameter (name=params_copy.y), Parameter (name=params_copy.z))
```
