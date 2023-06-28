# dtype

<a href="https://gitee.com/mindspore/docs/blob/r1.1/docs/programming_guide/source_zh_cn/dtype.md" target="_blank"><img src="./_static/logo_source.png"></a>
&nbsp;&nbsp;
<a href="https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/r1.1/programming_guide/mindspore_dtype.ipynb"><img src="./_static/logo_notebook.png"></a>
&nbsp;&nbsp;
<a href="https://console.huaweicloud.com/modelarts/?region=cn-north-4#/notebook/loading?share-url-b64=aHR0cHM6Ly9vYnMuZHVhbHN0YWNrLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vbWluZHNwb3JlLXdlYnNpdGUvbm90ZWJvb2svbW9kZWxhcnRzL3Byb2dyYW1taW5nX2d1aWRlL21pbmRzcG9yZV9kdHlwZS5pcHluYg==&image_id=65f636a0-56cf-49df-b941-7d2a07ba8c8c" target="_blank"><img src="./_static/logo_modelarts.png"></a>

## 概述

MindSpore张量支持不同的数据类型，包含`int8`、`int16`、`int32`、`int64`、`uint8`、`uint16`、`uint32`、`uint64`、`float16`、`float32`、`float64`、`bool_`，与NumPy的数据类型一一对应。

在MindSpore的运算处理流程中，Python中的`int`数会被转换为定义的int64类型，`float`数会被转换为定义的`float32`类型。

详细的类型支持情况请参考<https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/mindspore.html#mindspore.dtype>。

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
