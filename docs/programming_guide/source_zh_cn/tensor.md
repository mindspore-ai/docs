# Tensor

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/docs/programming_guide/source_zh_cn/tensor.md)

## 概述

张量（Tensor）是MindSpore网络运算中的基本数据结构。张量中的数据类型可参考[dtype](https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.0/dtype.html)。

不同维度的张量分别表示不同的数据，0维张量表示标量，1维张量表示向量，2维张量表示矩阵，3维张量可以表示彩色图像的RGB三通道等等。

> 本文中的所有示例，支持在PyNative模式下运行，暂不支持CPU。

## 张量构造

构造张量时，支持传入`Tensor`、`float`、`int`、`bool`、`tuple`、`list`和`NumPy.array`类型。

`Tensor`作为初始值时，可指定dtype，如果没有指定dtype，`int`、`float`、`bool`分别对应`int32`、`float32`、`bool_`，`tuple`和`list`生成的1维`Tensor`数据类型与`tuple`和`list`里存放数据的类型相对应。

代码样例如下：

```python
import numpy as np
from mindspore import Tensor
from mindspore.common import dtype as mstype

x = Tensor(np.array([[1, 2], [3, 4]]), mstype.int32)
y = Tensor(1.0, mstype.int32)
z = Tensor(2, mstype.int32)
m = Tensor(True, mstype.bool_)
n = Tensor((1, 2, 3), mstype.int16)
p = Tensor([4.0, 5.0, 6.0], mstype.float64)

print(x, "\n\n", y, "\n\n", z, "\n\n", m, "\n\n", n, "\n\n", p)
```

输出如下：

```text
[[1 2]
 [3 4]]

1

2

True

[1 2 3]

[4. 5. 6.]
```

## 张量的属性和方法

### 属性

张量的属性包括形状（shape）和数据类型（dtype）。

- 形状：`Tensor`的shape，是一个tuple。
- 数据类型：`Tensor`的dtype，是MindSpore的一个数据类型。

代码样例如下：

```python
import numpy as np
from mindspore import Tensor
from mindspore.common import dtype as mstype

x = Tensor(np.array([[1, 2], [3, 4]]), mstype.int32)
x_shape = x.shape
x_dtype = x.dtype

print(x_shape, x_dtype)
```

输出如下：

```text
(2, 2) Int32
```

### 方法

张量的方法包括`all`、`any`和`asnumpy`，`all`和`any`方法目前只支持Ascend。

- `all(axis, keep_dims)`：在指定维度上通过`and`操作进行归约，`axis`代表归约维度，`keep_dims`表示是否保留归约后的维度。
- `any(axis, keep_dims)`：在指定维度上通过`or`操作进行归约，参数含义同`all`。
- `asnumpy()`：将`Tensor`转换为NumPy的array。

代码样例如下：

```python
import numpy as np
from mindspore import Tensor
from mindspore.common import dtype as mstype

x = Tensor(np.array([[True, True], [False, False]]), mstype.bool_)
x_all = x.all()
x_any = x.any()
x_array = x.asnumpy()

print(x_all, "\n\n", x_any, "\n\n", x_array)
```

输出如下：

```text
False

True

[[ True  True]
 [False False]]

```
