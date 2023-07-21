# dtype

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/docs/programming_guide/source_zh_cn/dtype.md)

## 概述

MindSpore张量支持不同的数据类型，包含`int8`、`int16`、`int32`、`int64`、`uint8`、`uint16`、`uint32`、`uint64`、`float16`、`float32`、`float64`、`bool_`，与NumPy的数据类型一一对应。

在MindSpore的运算处理流程中，Python中的`int`数会被转换为定义的int64类型，`float`数会被转换为定义的`float32`类型。

详细的类型支持情况请参考<https://www.mindspore.cn/doc/api_python/zh-CN/r1.0/mindspore/mindspore.html#mindspore.dtype>。

以下代码，打印MindSpore的数据类型int32。

```python
from mindspore import dtype as mstype

data_type = mstype.int32
print(data_type)
```

输出如下：

```text
Int32
```

## 数据类型转换接口

MindSpore提供了以下几个接口，实现与NumPy数据类型和Python内置的数据类型间的转换。

- `dtype_to_nptype`：将MindSpore的数据类型转换为NumPy对应的数据类型。
- `dtype_to_pytype`：将MindSpore的数据类型转换为Python对应的内置数据类型。
- `pytype_to_dtype`：将Python内置的数据类型转换为MindSpore对应的数据类型。

以下代码实现了不同数据类型间的转换，并打印转换后的类型。

```python
from mindspore import dtype as mstype

np_type = mstype.dtype_to_nptype(mstype.int32)
ms_type = mstype.pytype_to_dtype(int)
py_type = mstype.dtype_to_pytype(mstype.float64)

print(np_type)
print(ms_type)
print(py_type)
```

输出如下：

```text
<class 'numpy.int32'>
Int64
<class 'float'>
```
