# 张量

<!-- TOC -->

- [张量](#张量)
    - [概述](#概述)
    - [常量张量](#常量张量)
    - [变量张量](#变量张量)
    - [张量的属性和操作](#张量的属性和操作)
        - [属性](#属性)
        - [操作](#操作)
    - [张量操作](#张量操作)
        - [结构操作](#结构操作)
        - [数学运算](#数学运算)
    - [广播](#广播)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/api/source_zh_cn/programming_guide/tensor.md" target="_blank"><img src="./_static/logo_source.png"></a>

## 概述

张量是MindSpore网络运算中的基本数据结构，即为多维数组，分为常量张量（Tensor）和变量张量（Parameter），常量张量的值在网络中不能被改变，而变量张量的值则可以被更新。

张量里的数据分为不同的类型，支持的类型有int8、int16、int32、int64、uint8、uint16、uint32、uint64、float16、float32、float64、bool_，与NumPy里的数据类型一一对应。

不同维度的张量分别表示不同的数据，0维张量表示标量，1维张量表示向量，2维张量表示矩阵，3维张量可以表示彩色图像的RGB三通道等等。
  
## 常量张量

常量张量的值在网络中不能被改变，构造时支持传入float、int、bool、tuple、list和numpy.array。

Tensor作为初始值可指定dtype，如果没有指定dtype，int、float、bool分别对应int32、float32、bool_，tuple和list生成的1维Tensor数据类型与tuple和list里存放数据的类型相对应。

代码样例如下：

```
import numpy as np
from mindspore import Tensor
from mindspore.common import dtype as mstype

x = Tensor(np.array([1, 2], [3, 4]]), mstype.int32)
y = Tensor(1.0, mstype.int32)
z = Tensor(2, mstype.int32)
m = Tensor(True, mstype.bool_)
n = Tensor((1, 2, 3), mstype.int16)
p = Tensor([4.0, 5.0, 6.0], mstype.float64)

print(x, "\n\n", y, "\n\n", z, "\n\n", m, "\n\n", n, "\n\n", p, "\n\n", q)
```

输出如下：

```
[[1 2]
 [3 4]]

1.0

2

True

[1, 2, 3] 
```

## 变量张量

变量张量的值在网络中可以被更新，用来表示需要被更新的参数，MindSpore使用Tensor的子类Parameter构造变量张量，构造时支持传入Tensor Initializer。

代码样例如下：

```
import numpy as np
from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer

x = Tensor(np.arange(2*3).reshape((2, 3)))
y = Parameter(x, name="x")
z = Parameter(initializer('ones', [1, 2, 3], mstype.float32), name='y')

print(x, "\n\n", y, "\n\n", z)
```

输出如下：

```
[[0 1 2]
 [3 4 5]]

Parameter (name=x, value=[[0 1 2] [3 4 5]])

Parameter (name=y, value=[[1. 1. 1.] [1. 1. 1.]]
```
  
## 张量的属性和操作
### 属性

张量的属性包括形状（shape）和数据类型（dtype）。
- 形状：Tensor的shape，是一个tuple。
- 数据类型：Tensor的的dtype，是MindSpore的一个数据类型。

代码样例如下：

```
import numpy as np
from mindspore import Tensor
from mindspore.common import dtype as mstype

x = Tensor(np.array([1, 2], [3, 4]]), mstype.int32)
x_shape = x.shape
x_dtype = x.dtype

print(x_shape, x_dtype)
```

输出如下：

```
(2, 2) Int32
```
 
### 操作

张量的常见操作包括`all`、`any`和`asnumpy`。
- all(axis, keep_dims)：在指定维度上通过“and”操作进行归约，axis代表归约维度，keep_dims表示是否保留归约后的维度。
- any(axis, keep_dims)：在指定维度上通过“any”操作进行归约，axis代表归约维度，keep_dims表示是否保留归约后的维度。
- asnumpy()：将Tensor转换为NumPy的array。

代码样例如下：

```
import numpy as np
from mindspore import Tensor
from mindspore.common import dtype as mstype

x = Tensor(np.array([1, 2], [3, 4]]), mstype.int32)
x_all = x.all()
x_any = a.any()
x_array = x.asnumpy()

print(x_all, "\n\n", x_any, "\n\n", x_array)

```

输出如下：

```
False

True

[[True True]
 [False True]]
```

## 张量操作

张量的操作主要包括张量的结构操作和数学运算。

### 结构操作

张量的结构操作主要包括张量创建、索引切片、维度变换和合并分割。

- 张量创建

  MindSpore创建张量的算子有`Range`、`Fill`、`ScalarToArray`、`ScalarToTensor`、`TupleToArray`、`Cast`、`ZerosLike`、`OnesLike`等。

  代码样例如下：
  
  ```
  from mindspore.common import dtype as mstype
  from mindspore.ops import operations as P

  x = P.Fill()(mstype.float32, (2, 2), 1)
  y = P.ScalarToArray()(1.0)
  z = P.TupleToArray()((1, 2, 3))

  print(x, "\n\n", y, "\n\n", z)
  ```

  输出如下：
  
  ```
  [[1. 1.]
  [1. 1.]]

  1.0

  [1 2 3]
  ```
 
- 索引切片

  MindSpore的索引操作跟NumPy的索引操作保持一致，包括取值和赋值，支持整数索引、bool索引、None索引、切片索引、Tensor索引和混合索引。
  
  支持索引操作的算子主要有`Slice`、`StridedSlice`、`Gather`、`GatherNd`、`ScatterUpdate`、`ScatterNdUpdate`等。

  代码样例如下：

  ```
  import numpy as np
  from mindspore import Tensor
  from mindspore.common import dtype as mstype

  x = Tensor(np.arange(3*4*5).reshape((3, 4, 5)))
  indices = Tensor(np.array([[0, 1], [1, 2]]), mstype.int32)
  y = [:3, indices, 3]

  print(x, "\n\n", y)
  ```
  
  输出如下：
  
  ```
  [[[3 8]
    [8 13]]
  [[23 28]
    [28 33]]
  [[43 48]
    [48 53]]]
  ```
 
- 维度变化

  MindSpore的维度变化，主要涉及shape改变、维度扩展、维度消除、转置，支持的算子有`Reshape`、`ExpandDims`、`Squeeze`、`Transpose`、Reduce类算子，具体含义如下：
  - Reshape：改变张量的shape，改变前后张量中元素个数保持一致。
  - ExpanDims：在张量里插入一维，长度为1。
  - Squeeze：将张量里长度为1的维度消除。
  - Transpose：将张量转置，交换维度。
  - Reduce类算子：对张量在指定维度上按照一定计算规则进行归约，Tensor的`all`和`any`接口就是归约操作中的两种。

  代码样例如下：

  ```
  import numpy as np
  from mindspore import Tensor
  from mindspore.ops import operations as P

  x = Tensor(np.arange(2*3).reshape((2, 3)))
  y = P.Reshape()(x, (4, 3, 5))
  z = P.ExpandDims()(x, 1)
  m = P.Squeeze(axis=3)(x)
  n = P.Transpose()(x, (0, 2, 3, 1))
  ```

- 合并分割

  MindSpore可以将多个张量合并为一个，也可以将一个张量拆分成多个，支持的算子有`Pack`、`Concat`和`Split`，具体含义如下：
  - Pack：将多个Tensor打包成一个，会增加一个维度，增加维度的长度等于参与打包算子的个数。
  - Concat：将多个Tensor在某一个维度上进行拼接，不会增加维度。
  - Split：将一个Tensor进行拆分。

  代码样例如下：

  ```
  import numpy as np
  from mindspore import Tensor
  from mindspore.ops import operations as P

  x = Tensor(np.arange(2*3).reshape((2, 3)))
  x = Tensor(np.arange(2*3).reshape((2, 3)))
  z = P.Pack(axis=0)((x, y))
  m = P.Concat(axis=0)((x, y))
  n = P.Split(0, 2)(x)

  print(z, "\n\n", m, "\n\n", n[0], "\n", n[1])
  ```

  输出如下：
  
  ```
  [[[0 1 2]
    [3 4 5]]
  [[0 1 2]
    [3 4 5]]] 

  [[0 1 2]
  [3 4 5]
  [0 1 2]
  [3 4 5]] 

  [[0 1 2]] 
  [[3 4 5]]
  ```

### 数学运算

数学运算主要是对张量的一些数学运算上的操作，包括加减乘除、求模求余求幂、比大小等等。支持的算子包括`TensorAdd`、`Sub`、`Mul`、`RealDiv`、`FloorMod`、`FloorDiv`、`Pow`、`Maximum`、`Minimum`等。

## 广播

MindSpore支持对张量进行广播，包括显式广播和隐式广播。显式广播为算子`Tile`，隐式广播是当两个shape不一样的张量进行运算，且它们的shape满足广播的要求时，系统会自动将这两个张量广播成shape相同的张量进行运算。

代码样例如下：

```
import numpy as np
from mindspore import Tensor
from mindspore.ops import operations as P

x = Tensor(np.arange(2*3).reshape((2, 3)))
y = P.Tile()(x, (2, 3))

print(x, "\n\n", y)

```

输出如下：

```
[[0 1 2]
 [3 4 5]] 

[[0 1 2 0 1 2 0 1 2]
 [3 4 5 3 4 5 3 4 5]
 [0 1 2 0 1 2 0 1 2]
 [3 4 5 3 4 5 3 4 5]]
```
