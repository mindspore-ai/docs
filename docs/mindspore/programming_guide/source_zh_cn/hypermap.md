# 运算重载

<!-- TOC -->

- [运算重载](#运算重载)
    - [概述](#概述)
    - [MultitypeFuncGraph](#multitypefuncgraph)
    - [HyperMap](#hypermap-1)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/programming_guide/source_zh_cn/hypermap.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## 概述

`mindspore.ops.composite`中提供了一些涉及图变换的组合类算子，例如`MultitypeFuncGraph`、`HyperMap`等。

## MultitypeFuncGraph

`MultitypeFuncGraph`用于生成重载函数，支持不同类型的输入。用户可以使用`MultitypeFuncGraph`定义一组重载的函数，根据不同类型，采用不同实现。首先初始化一个`MultitypeFuncGraph` 对象，使用带有输入类型的 `register` 作为待注册函数的装饰器，使得该对象支持多种类型的输入。更多使用方法见：[MultitypeFuncGraph](https://www.mindspore.cn/docs/api/zh-CN/r1.5/api_python/ops/mindspore.ops.MultitypeFuncGraph.html)。

代码样例如下：

```python
import numpy as np
from mindspore.ops import MultitypeFuncGraph
from mindspore import Tensor
import mindspore.ops as ops

add = MultitypeFuncGraph('add')
@add.register("Number", "Number")
def add_scalar(x, y):
    return ops.scalar_add(x, y)

@add.register("Tensor", "Tensor")
def add_tensor(x, y):
    return ops.tensor_add(x, y)

tensor1 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
tensor2 = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype('float32'))
print('tensor', add(tensor1, tensor2))
print('scalar', add(1, 2))
```

运行结果如下：

```text
tensor [[2.4 4.2]
 [4.4 6.4]]
scalar 3
```

## HyperMap

`HyperMap`可以对一组或多组输入做指定的运算，可以配合`MultitypeFuncGraph`一起使用。例如定义一组重载的`add`函数后，对多组不同类型的输入进行`add`运算。不同于`Map`，`HyperMap` 能够用于嵌套结构，对序列或嵌套序列中的输入做指定运算。更多使用方法见：[HyperMap](https://www.mindspore.cn/docs/api/zh-CN/r1.5/api_python/ops/mindspore.ops.HyperMap.html)。

代码样例如下：

```python
from mindspore import dtype as mstype
from mindspore import Tensor
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
output = add_map((Tensor(1, mstype.float32), Tensor(2, mstype.float32), 1), (Tensor(3, mstype.float32), Tensor(4, mstype.float32), 2))
print("output =", output)
```

运行结果如下：

```text
output = (Tensor(shape=[], dtype=Float32, value= 4), Tensor(shape=[], dtype=Float32, value= 6), 3)
```

此例子中传入`add_map`的输入包含了两个序列，`HyperMap`会以`operation(args[0][i], args[1][i])`的形式分别从两个序列中取相应的元素作为`add`函数的输入`x`和`y`，例如`add(Tensor(1, mstype.float32), Tensor(3, mstype.float32))`。
