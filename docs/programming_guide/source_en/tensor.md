# Tensor

<a href="https://gitee.com/mindspore/docs/blob/r1.1/docs/programming_guide/source_en/tensor.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Overview

Tensor is a basic data structure in the MindSpore network computing. For details about data types in tensors, see [dtype](https://www.mindspore.cn/doc/programming_guide/en/r1.1/dtype.html).

Tensors of different dimensions represent different data. For example, a 0-dimensional tensor represents a scalar, a 1-dimensional tensor represents a vector, a 2-dimensional tensor represents a matrix, and a 3-dimensional tensor may represent the three channels of RGB images.

> All examples in this document can be run in the PyNative mode.

## Tensor Structure

During tensor creation, the `Tensor`, `float`, `int`, `bool`, `tuple`, `list`, and `NumPy.array` types can be transferred, while `tuple` and `list` can only store `float`, `int`, and `bool` data.

`dtype` can be specified when `Tensor` is initialized. When the `dtype` is not specified, if the initial value is `int`, `float` or `bool`, then a 0-dimensional `Tensor` with data types `mindspore.int32`, `mindspore.float32` or `mindspore.bool_` will be generated respectively. If the initial values are `tuple` and `list`, the generated 1-dimensional `Tensor` data type corresponds to the type stored in `tuple` and `list`. If it contains multiple different types of data, follow the below priority: `bool` < `int` < `float`, to select the mindspore data type corresponding to the highest relative priority type. If the initial value is `Tensor`,  the consistent data type `Tensor` is generated. If the initial value is `NumPy.array`, the corresponding data type `Tensor` is generated.

A code example is as follows:

```python
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype

x = Tensor(np.array([[1, 2], [3, 4]]), mstype.int32)
y = Tensor(1.0, mstype.int32)
z = Tensor(2, mstype.int32)
m = Tensor(True, mstype.bool_)
n = Tensor((1, 2, 3), mstype.int16)
p = Tensor([4.0, 5.0, 6.0], mstype.float64)
q = Tensor(p, mstype.float64)

print(x, "\n\n", y, "\n\n", z, "\n\n", m, "\n\n", n, "\n\n", p, "\n\n", q)
```

The following information is displayed:

```text
[[1 2]
 [3 4]]

1

2

True

[1 2 3]

[4. 5. 6.]

[4. 5. 6.]
```

## Tensor Attributes and Methods

### Attributes

Tensor attributes include shape and data type (dtype).

- shape: a tuple
- dtype: a data type of MindSpore

A code example is as follows:

```python
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype

x = Tensor(np.array([[1, 2], [3, 4]]), mstype.int32)
x_shape = x.shape
x_dtype = x.dtype

print(x_shape, x_dtype)
```

The following information is displayed:

```text
(2, 2) Int32
```

### Methods

Tensor methods include `all`, `any`, and `asnumpy`. Currently, the `all` and `any` methods support only Ascend, and the data type of `Tensor` is required to be `mindspore.bool_`.
.

- `all(axis, keep_dims)`: performs the `and` operation on a specified dimension to reduce the dimension. `axis` indicates the reduced dimension, and `keep_dims` indicates whether to retain the reduced dimension.
- `any(axis, keep_dims)`: performs the `or` operation on a specified dimension to reduce the dimension. The parameter meaning is the same as that of `all`.
- `asnumpy()`: converts `Tensor` to an array of NumPy.

A code example is as follows:

```python
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype

x = Tensor(np.array([[True, True], [False, False]]), mstype.bool_)
x_all = x.all()
x_any = x.any()
x_array = x.asnumpy()

print(x_all, "\n\n", x_any, "\n\n", x_array)
```

The following information is displayed:

```text
False

True

[[ True  True]
 [False False]]

```
