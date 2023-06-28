# Parameter

<a href="https://gitee.com/mindspore/docs/blob/r1.0/docs/programming_guide/source_zh_cn/parameter.md" target="_blank"><img src="./_static/logo_source.png"></a>

## 概述

`Parameter`是变量张量，代表在训练网络时，需要被更新的参数。本章主要介绍了`Parameter`的初始化以及属性和方法的使用，同时介绍了`ParameterTuple`。

## 初始化

```python
mindspore.Parameter(default_input, name, requires_grad=True, layerwise_parallel=False)
```

初始化一个`Parameter`对象，传入的数据支持`Tensor`、`Initializer`、`int`和`float`四种类型。

`Initializer`是初始化器，保存了shape和dtype信息，提供`to_tensor`方法生成存有数据的`Tensor`，可调用`initializer`接口生成`Initializer`对象。

当网络采用半自动或者全自动并行策略，并且使用`Initializer`初始化`Parameter`时，`Parameter`里保存的不是`Tensor`，而是`MetaTensor`。

`MetaTensor`与`Tensor`不同，`MetaTensor`仅保存张量的形状和类型，而不保存实际数据，所以不会占用任何内存，可调用`init_data`接口将`Parameter`里保存的`MetaTensor`转化为`Tensor`。

可为每个`Parameter`指定一个名称，便于后续操作和更新。

当参数需要被更新时，需要将`requires_grad`设置为`True`。

当`layerwise_parallel`（混合并行）配置为True时，参数广播和参数梯度聚合时会过滤掉该参数。

有关分布式并行的相关配置，可以参考文档：<https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.0/auto_parallel.html>。

下例通过三种不同的数据类型构造了`Parameter`，三个`Parameter`都需要更新，都不采用layerwise并行。如下：

```python
import numpy as np
from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer

x = Parameter(default_input=Tensor(np.arange(2*3).reshape((2, 3))), name="x")
y = Parameter(default_input=initializer('ones', [1, 2, 3], mstype.float32), name='y')
z = Parameter(default_input=2.0, name='z')

print(x, "\n\n", y, "\n\n", z)
```

输出如下：

```text
Parameter (name=x, value=[[0 1 2]
                          [3 4 5]])

Parameter (name=y, value=[[[1. 1. 1.]
                           [1. 1. 1.]]])

Parameter (name=z, value=2.0)
```

## 属性

- `inited_param`：返回保存了实际数据的`Parameter`，如果`Parameter`原本保存的是`MetaTensor`，会将其转换为`Tensor`。

- `name`：实例化`Parameter`时，为其指定的名字。

- `sliced`：用在自动并行场景下，表示`Parameter`里保存的数据是否是分片数据。

  如果是，就不再对其进行切分，如果不是，需要根据网络并行策略确认是否对其进行切分。

- `is_init`：`Parameter`的初始化状态。在GE后端，Parameter需要一个`init graph`来从主机同步数据到设备侧，该标志表示数据是否已同步到设备。
  此标志仅在GE后端起作用，其他后端将被设置为False。

- `layerwise_parallel`：`Parameter`是否支持layerwise并行。如果支持，参数就不会进行广播和梯度聚合，反之则需要。

- `requires_grad`：是否需要计算参数梯度。如果参数需要被训练，则需要计算参数梯度，否则不需要。

- `data`： `Parameter`本身。

下例通过`Tensor`初始化一个`Parameter`，获取了`Parameter`的相关属性。如下：

```python
import numpy as np

from mindspore import Tensor, Parameter

x = Parameter(default_input=Tensor(np.arange(2*3).reshape((2, 3))), name="x")

print("name: ", x.name, "\n",
      "sliced: ", x.sliced, "\n",
      "is_init: ", x.is_init, "\n",
      "inited_param: ", x.inited_param, "\n",
      "requires_grad: ", x.requires_grad, "\n",
      "layerwise_parallel: ", x.layerwise_parallel, "\n",
      "data: ", x.data)
```

输出如下：

```text
name:  x
sliced:  False
is_init:  False
inited_param:  None
requires_grad:  True
layerwise_parallel:  False

data:  Parameter (name=x, value=[[0 1 2]
                                 [3 4 5]])
```

## 方法

- `init_data`：在网络采用半自动或者全自动并行策略的场景下，
  当初始化`Parameter`传入的数据是`Initializer`时，可调用该接口将`Parameter`保存的数据转换为`Tensor`。

- `set_data`：设置`Parameter`保存的数据，支持传入`Tensor`、`Initializer`、`int`和`float`进行设置，
  将方法的入参`slice_shape`设置为True时，可改变`Parameter`的shape，反之，设置的数据shape必须与`Parameter`原来的shape保持一致。

- `set_param_ps`：控制训练参数是否通过[Parameter Server](https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/advanced_use/apply_parameter_server_training.html)进行训练。

- `clone`：克隆`Parameter`，需要指定克隆之后的参数名称。

下例通过`Initializer`来初始化`Tensor`，调用了`Parameter`的相关方法。如下：

```python
import numpy as np

from mindspore import Tensor, Parameter
from mindspore import dtype as mstype
from mindspore.common.initializer import initializer

x = Parameter(default_input=initializer('ones', [1, 2, 3], mstype.float32), name='x')

print(x)
print(x.clone(prefix="x_c"))
print(x.init_data())
print(x.set_data(data=Tensor(np.arange(2*3).reshape((1, 2, 3)))))
```

输出如下：

```text
Parameter (name=x, value=[[[1. 1. 1.]
                           [1. 1. 1.]]])
Parameter (name=x_c.x, value=[[[1. 1. 1.]
                               [1. 1. 1.]]])
Parameter (name=x, value=[[[1. 1. 1.]
                           [1. 1. 1.]]])
Parameter (name=x, value=[[[0. 1. 2.]
                           [3. 4. 5.]]])
```

## ParameterTuple

继承于`tuple`，用于保存多个`Parameter`，通过`__new__(cls, iterable)`传入一个存放`Parameter`的迭代器进行构造，提供`clone`接口进行克隆。

下例构造了一个`ParameterTuple`对象，并进行了克隆。如下：

```python
import numpy as np
from mindspore import Tensor, Parameter, ParameterTuple
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer

x = Parameter(default_input=Tensor(np.arange(2*3).reshape((2, 3))), name="x")
y = Parameter(default_input=initializer('ones', [1, 2, 3], mstype.float32), name='y')
z = Parameter(default_input=2.0, name='z')
params = ParameterTuple((x, y, z))
params_copy = params.clone("params_copy")
print(params, "\n")
print(params_copy)
```

输出如下：

```text
(Parameter (name=x, value=Tensor(shape=[2, 3], dtype=Int64,
[[ 0,  1,  2],
 [ 3,  4,  5]])), Parameter (name=y, value=Tensor(shape=[1, 2, 3], dtype=Float32,
[[[ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00],
  [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00]]])), Parameter (name=z, value=Tensor(shape=[], dtype=Float32, 2)))

(Parameter (name=params_copy.x, value=Tensor(shape=[2, 3], dtype=Int64,
[[ 0,  1,  2],
 [ 3,  4,  5]])), Parameter (name=params_copy.y, value=Tensor(shape=[1, 2, 3], dtype=Float32,
[[[ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00],
  [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00]]])), Parameter (name=params_copy.z, value=Tensor(shape=[], dtype=Float32, 2)))
```
