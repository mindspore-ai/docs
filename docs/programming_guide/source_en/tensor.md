# Tensor

<!-- TOC -->

- [Tensor](#tensor)
    - [Overview](#overview)
    - [Tensor Structure](#tensor-structure)
    - [Tensor Attributes and Methods](#tensor-attributes-and-methods)
        - [Attributes](#attributes)
        - [Methods](#methods)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/programming_guide/source_en/tensor.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## Overview

Tensor is a basic data structure in the MindSpore network computing. For details about data types in tensors, see [dtype](https://www.mindspore.cn/doc/programming_guide/en/master/dtype.html).

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

Tensor attributes include `shape`，`dtype`, `T`, `itemsize`, `nbytes`, `ndim`, `size`, `strides`.

- shape: a tuple
- dtype: a data type of MindSpore
- T: transposed view of original tensor
- itemsize: an integer, representing the number of bytes consumed by a single element in the `Tensor`
- nbytes: an integer, representing the total number of bytes consumed by `Tensor`
- ndim: an integer, representing the rank of the `Tensor`
- size: an integer, representing the total number of elements in `Tensor`
- strides: the tuple of bytes to traverse in each dimension in `Tensor`

A code example is as follows:

```python
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype

x = Tensor(np.array([[1, 2], [3, 4]]), mstype.int32)
x_shape = x.shape
x_dtype = x.dtype
x_transposed = x.T
x_itemsize = x.itemsize
x_nbytes = x.nbytes
x_ndim = x.ndim
x_size = x.size
x_strides = x.strides
print("x_shape:", x_shape)
print("x_dtype:", x_dtype)
print("x_transposed:", x_transposed)
print("x_itemsize:", x_itemsize)
print("x_nbytes:", x_nbytes)
print("x_ndim:", x_ndim)
print("x_size:", x_size)
print("x_strides:", x_strides)
```

The following information is displayed:

```text
x_shape: (2, 2)
x_dtype: Int32
x_transposed: [[1 3]
 [2 4]]
x_itemsize: 4
x_nbytes: 16
x_ndim: 2
x_size: 4
x_strides: (8, 4)
```

### Methods

Tensor methods include `all`, `any`, `asnumpy` and many other functions. Numpy-like ndarray methods are also provided. For a full description of all tensor methods, please see [API: mindspore.Tensor](https://www.mindspore.cn/doc/api_python/en/master/mindspore/mindspore.html#mindspore.Tensor). The following is a brief introduction to some of the tensor methods.

- `all(axis, keep_dims)`: performs the `and` operation on a specified dimension to reduce the dimension. `axis` indicates the reduced dimension, and `keep_dims` indicates whether to retain the reduced dimension.
- `any(axis, keep_dims)`: performs the `or` operation on a specified dimension to reduce the dimension. The parameter meaning is the same as that of `all`.
- `asnumpy()`: converts `Tensor` to an array of NumPy.
- `sum(axis, dtype, keepdims, initial)`: sums the tensor over the given `axis`, `axis` indicates the reduced dimension, `dtype` specifies the output data type, `keepdims` indicates whether to retain the reduced dimension, and `initial` indicates the starting value for the sum.

A code example is as follows:

```python
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype

x = Tensor(np.array([[True, True], [False, False]]), mstype.bool_)
x_all = x.all()
x_any = x.any()
x_array = x.asnumpy()
print("x_all:", x_all)
print("x_any:", x_any)
print("x_array:", x_array)

import mindspore.numpy as mnp
y = Tensor(np.array([[1., 2.], [3., 4.]]), mstype.float32)
# y.sum() and mindspore.numpy.sum(y) are equivalent methods
y_sum_tensor = y.sum()
y_sum_mnp = mnp.sum(y)
print("y_sum_tensor:", y_sum_tensor)
print("y_sum_mnp:", y_sum_mnp)
```

The following information is displayed:

```text
x_all: False
x_any: True
x_array: [[ True  True]
 [False False]]
y_sum_tensor: 10.0
y_sum_mnp: 10.0
```
