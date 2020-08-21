# 张量数据结构

<!-- TOC -->

- [张量数据结构](#张量数据结构)
    - [概述](#概述)  
    - [常量张量](#常量张量)
    - [变量张量](#变量张量)
    - [张量属性和接口](#张量属性和接口)
    - [张量操作](#张量操作)
    - [广播](#广播)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/source_zh_cn/constraints_on_network_construction.md" target="_blank"><img src="./_static/logo_source.png"></a>

## 概述
张量是MindSpore网络运算中的基本数据结构，即为多维数组，与numpy中的array类似。
从行为特性来看的话，分为常量张量和变量张量，常量张量的值在网络中不能被改变，而变量张量的值则可以被更新。
张量里的数据分为不同的类型，当前的类型有"int8", "int16", "int32", "int64", "uint8", "uint16",
"uint32","uint64", "float16"，"float32", "float64", "bool_"， 与numpy里的数据类型一一对应。
而根据张量的维度(rank)不同，0维张量表示标量，1维张量表示向量，2维张量表示矩阵，3维张量可以表示彩色图像的RGB三通道等等。
可以简单的理解为有几层中括号，就有几个维度。
  
 ## 常量张量
常量张量的值在网络里不能被改变, 构造时支持传入float, int, bool, tuple, list, numpy.array，Tensor作为初始值
可指定dtype，如果没有指定dtype，int、float、bool分别对应int32、float32、bool_，tuple和list生成的1维Tensor数据类型
与tuple和list里存放数据的类型相对应。如下：
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
变量张量的值在网络里可以被更新，用来表示需要被更新的参数，mindspore构造变量张量使用Tensor的子类Parameter，构造时支持传入Tensor
Initializer，如下：
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
  
## 张量的属性和接口
### 属性
* shape: Tensor的shape，是一个tuple。
* dtype: Tensor的的dtype，是mindspore的一个数据类型。
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
 
### 接口：
* all(axis, keep_dims): 在指定维度上通过‘and‘操作进行归约，axis代表归约维度， keep_dims表示是否保留归约后的维度。
* any(axis, keep_dims): 在指定维度上通过‘any‘操作进行归约，axis代表归约维度， keep_dims表示是否保留归约后的维度。
* asnumpy(): 将Tensor转换为numpy的array。
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
张量的结构操作主要包括张量创建，索引切片，维度变换和合并分割。
* 张量创建

mindspore创建张量的算子有Range，Fill，ScalarToArray ScalarToTensor，TupleToArray，Cast，ZerosLike，OnesLike。
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
 
* 索引切片

mindspore的索引操作跟numpy的索引操作保持一致，包括取值和赋值，支持整数索引，bool索引，None索引，切片索引，Tensor索引，混合索引。
支持索引操作的算子主要有Slice，StridedSlice，Gaher，GatherNd，ScatterUpdata，ScatterNdUpdate等。

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
 
* 维度变化

mindspore的维度变化，主要涉及shape改变，维度扩展，维度消除，转置，支持的算子有Reshape，ExpandDims，Squeeze，Transpose，Reduce类算子。
Reshape改变张量的shape，改变前后张量中元素个数一致；ExpanDims在张量里插入一维，长度为1；Squeeze将张量里长度为1的维度消除；
Transpose将张量转置，交换维度；Reduce类算子主要是对张量在指定维度上按照一定计算规则进行归约，Tensor的all和any接口就是归约操作中的两种。

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

* 合并分割

mindspore可以将多个张量合并为一个，也可以将一个张量拆分成多个，支持的算子有Pack，Concat，Split。
Pack是将多个Tensor打包成一个，会增加一个维度，增加维度的长度等于参与打包算子的个数；
Concat是将多个Tensor在某一个维度上进行拼接，不会增加维度；Split是将一个Tensor进行拆分。

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
数学运算主要是对张量的一些数学运算上的操作，包括加减乘除，求模求余求幂，比大小等等。支持的算子包括TensorAdd，Sub，Mul，RealDiv，
FloorMod，FloorDiv，Pow，Maximum，Minimum等等。

## 广播
mindspore支持对张量进行广播，可显示广播，也可隐式广播。显示广播算子Tile，隐式广播是当两个shape不一样的张量进行运算时，
当他们的shape满足广播的要求时，系统会自动将这两个张量广播成shape一样的张量进行运算。
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
