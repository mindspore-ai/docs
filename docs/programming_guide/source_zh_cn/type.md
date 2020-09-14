# 数据类型

<!-- TOC -->

- [数据类型](#数据类型)
    - [概述](#概述)
    - [操作接口](#操作接口)
    

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/api/source_zh_cn/programming_guide/type.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

MindSpore张量支持不同的数据类型，有`int8`、`int16`、`int32`、`int64`、`uint8`、`uint16`、`uint32`、`uint64`、
`float16`、`float32`、`float64`、`bool_`， 与NumPy的数据类型一一对应，Python里的`int`数会被转换为定义的int64进行运算，
Python里的`float`数会被转换为定义的`float32`进行运算。

## 操作接口
- `dtype_to_nptype`

  可通过该接口将MindSpore的数据类型转换为NumPy对应的数据类型。

- `dtype_to_pytype`

  可通过该接口将MindSpore的数据类型转换为Python对应的内置数据类型。


- `pytype_to_dtype`

  可通过该接口将Python内置的数据类型转换为MindSpore对应的数据类型。

示例如下：

```
from mindspore import dtype as mstype

np_type = mstype.dtype_to_nptype(mstype.int32)
ms_type = mstype.pytype_to_dtype(int)
py_type = mstype.dtype_to_pytype(mstype.float64)

print(np_type)
print(ms_type)
print(py_type)
```

输出如下：

```
<class 'numpy.int32'>
Int64
<class 'float'>
```
