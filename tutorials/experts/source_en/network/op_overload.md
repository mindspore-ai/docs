# Compiler optimization for optimizer

<a href="https://gitee.com/mindspore/docs/blob/r1.8/tutorials/experts/source_en/network/op_overload.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_source_en.png"></a>

## Overview

`mindspore.ops.composite` provides some operator combinations related to graph transformation such as `MultitypeFuncGraph` and `HyperMap`.

## MultitypeFuncGraph

`MultitypeFuncGraph` is used to generate overloaded functions to support different types of inputs. Users can use `MultitypeFuncGraph` to define a group of overloaded functions. The implementation varies according to the function type. First initialize a `MultitypeFuncGraph` object, and use `register` with input type as the decorator of the function to be registered, so that the object can be called with different types of inputs. For more instructions, see [MultitypeFuncGraph](https://www.mindspore.cn/docs/en/r1.8/api_python/ops/mindspore.ops.MultitypeFuncGraph.html).

A code example is as follows:

```python
import numpy as np
from mindspore.ops import MultitypeFuncGraph
import mindspore as ms
import mindspore.ops as ops

add = MultitypeFuncGraph('add')
@add.register("Number", "Number")
def add_scalar(x, y):
    return ops.scalar_add(x, y)

@add.register("Tensor", "Tensor")
def add_tensor(x, y):
    return ops.add(x, y)

tensor1 = ms.Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
tensor2 = ms.Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
print('tensor', add(tensor1, tensor2))
print('scalar', add(1, 2))
```

The following information is displayed:

```text
tensor [[2.4 4.2]
 [4.4 6.4]]
scalar 3
```

## HyperMap

`HyperMap` can apply a specified operation to one or more input sequences, which can be used with `MultitypeFuncGraph`. For example, after defining a group of overloaded `add` functions, we can apply `add` operation to multiple input groups of different types. Unlike `Map`, `HyperMap` can be used in nested structures to perform specified operations on the input in a sequence or nested sequence. For more instructions, see [HyperMap](https://www.mindspore.cn/docs/en/r1.8/api_python/ops/mindspore.ops.HyperMap.html).

A code example is as follows:

```python
import mindspore as ms
from mindspore.ops import MultitypeFuncGraph, HyperMap
import mindspore.ops as ops

add = MultitypeFuncGraph('add')
@add.register("Number", "Number")
def add_scalar(x, y):
    return ops.scalar_add(x, y)

@add.register("Tensor", "Tensor")
def add_tensor(x, y):
    return ops.tensor_add(x, y)

add_map = HyperMap(add)
output = add_map((ms.Tensor(1, ms.float32), ms.Tensor(2, ms.float32), 1), (ms.Tensor(3, ms.float32), ms.Tensor(4, ms.float32), 2))
print("output =", output)
```

The following information is displayed:

```text
output = (Tensor(shape=[], dtype=Float32, value= 4), Tensor(shape=[], dtype=Float32, value= 6), 3)
```

In this example, the input of `add_map` contains two sequences. `HyperMap` will get the corresponding elements from the two sequences as `x` and `y` for the inputs of `add` in the form of `operation(args[0][i], args[1][i])`. For example, `add(Tensor(1, mstype.float32), Tensor(3, mstype.float32))`.
