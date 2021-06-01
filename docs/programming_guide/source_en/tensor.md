# Tensor

<!-- TOC -->

- [Tensor](#tensor)
    - [Overview](#overview)
    - [Tensor Structure](#tensor-structure)
    - [Tensor Attributes and Methods](#tensor-attributes-and-methods)
        - [Attributes](#attributes)
        - [Methods](#methods)
    - [Sparse tensor](#Sparse-tensor)
        - [RowTensor](#RowTensor)
        - [SparseTensor](#SparseTensor)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/programming_guide/source_en/tensor.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## Overview

Tensor is a basic data structure in the MindSpore network computing. For details about data types in tensors, see [dtype](https://www.mindspore.cn/doc/programming_guide/en/master/dtype.html).

Tensors of different dimensions represent different data. For example, a 0-dimensional tensor represents a scalar, a 1-dimensional tensor represents a vector, a 2-dimensional tensor represents a matrix, and a 3-dimensional tensor may represent the three channels of RGB images.

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

## Sparse Tensor

Sparse tensor is a special kind of tensor which most of the elements are zero. In some scenario, like in the
recommendation system, the data is sparse. If we use common dense tensors to represent the data, we may introduce many
unnecessary calculations, storage and communication costs. In this situation, it is better to use sparse tensor to
represent the data.

The common structure of sparse tensor is `<indices:Tensor,values:Tensor,dense_shape:Tensor>`. `indices` means index of
non-zero elements, `values` means the values of these non-zero elements and `dense_shape` means the dense shape of
the sparse tensor. Using this structure, we define data structure `RowTensor` and `SparseTensor`.

> Now, PyNative mode does not support sparse tensor.

### RowTensor

`RowTensor` is typically used to represent a subset of a larger tensor dense of shape `[L0, D1, ..., DN]`
where `L0` >> `D0`, and `D0` is the number of non-zero elements.

- `indices`: A 1-D integer tensor of shape `[D0]`. Represents the position of non-zero elements.
- `values`: A tensor of any data type of shape `[D0, D1, ..., DN]`. Represents the value of non-zero elements.
- `dense_shape`: An integer tuple which contains the shape of the corresponding dense tensor.

A code example is as follows:

```python
import mindspore as ms
import mindspore.nn as nn
from mindspore import RowTensor
class Net(nn.Cell):
    def __init__(self, dense_shape):
        super(Net, self).__init__()
        self.dense_shape = dense_shape
    def construct(self, indices, values):
        x = RowTensor(indices, values, self.dense_shape)
        return x.values, x.indices, x.dense_shape

indices = Tensor([0])
values = Tensor([[1, 2]], dtype=ms.float32)
out = Net((3, 2))(indices, values)
print(out[0])
print(out[1])
print(out[2])
```

The following information is displayed:

```text
[[1. 2.]]

[0]

(3, 2)

```

### SparseTensor

`SparseTensor` represents a set of nonzero elememts from a tensor at given indices. If the number of non-zero elements
is `N` and the dense shape of the sparse tensor is `ndims`：

- `indices`: A 2-D integer Tensor of shape `[N, ndims]`. Each line represents the index of non-zero elements.
- `values`: A 1-D tensor of any type and shape `[N]`. Represents the value of non-zero elements.
- `dense_shape`: A integer tuple of size `ndims`, which specifies the dense shape of the sparse tensor.

A code example is as follows:

```python
import mindspore as ms
import mindspore.nn as nn
from mindspore import SparseTensor
class Net(nn.Cell):
    def __init__(self, dense_shape):
       super(Net, self).__init__()
       self.dense_shape = dense_shape
    def construct(self, indices, values):
       x = SparseTensor(indices, values, self.dense_shape)
       return x.values, x.indices, x.dense_shape

indices = Tensor([[0, 1], [1, 2]])
values = Tensor([1, 2], dtype=ms.float32)
out = Net((3, 4))(indices, values)
print(out[0])
print(out[1])
print(out[2])
```

The following information is displayed:

```text
[1. 2.]

[[0 1]
 [1 2]]

(3, 4)

```
