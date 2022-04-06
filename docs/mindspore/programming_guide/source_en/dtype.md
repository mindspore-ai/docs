# DataType

`Ascend` `GPU` `CPU` `Beginner`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/dtype.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

MindSpore tensors support different data types, including `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `float16`, `float32`, `float64` and `bool_`, which correspond to the data types of NumPy.

In the computation process of MindSpore, the `int` data type in Python is converted into the defined `int64` type, and the `float` data type is converted into the defined `float32` type.

For details about the supported types, see <https://www.mindspore.cn/docs/api/en/master/api_python/mindspore.html#mindspore.dtype>.

In the following code, the data type of MindSpore is int32.

```python
from mindspore import dtype as mstype

data_type = mstype.int32
print(data_type)
```

The following information is displayed:

```text
Int32
```

## Data Type Conversion API

MindSpore provides the following APIs for conversion between NumPy data types and Python built-in data types:

- `dtype_to_nptype`: converts the data type of MindSpore to the corresponding data type of NumPy.
- `dtype_to_pytype`: converts the data type of MindSpore to the corresponding built-in data type of Python.
- `pytype_to_dtype`: converts the built-in data type of Python to the corresponding data type of MindSpore.

The following code implements the conversion between different data types and prints the converted type.

```python
from mindspore import dtype as mstype

np_type = mstype.dtype_to_nptype(mstype.int32)
ms_type = mstype.pytype_to_dtype(int)
py_type = mstype.dtype_to_pytype(mstype.float64)

print(np_type)
print(ms_type)
print(py_type)
```

The following information is displayed:

```text
<class 'numpy.int32'>
Int64
<class 'float'>
```
