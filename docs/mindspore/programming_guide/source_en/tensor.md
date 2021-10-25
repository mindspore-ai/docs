# Tensor

`Ascend` `GPU` `CPU` `Beginner`

<!-- TOC -->

- [Tensor](#tensor)
    - [Overview](#overview)
    - [Tensor Structure](#tensor-structure)
    - [Tensor Operations, Attributes and Methods](#tensor-operations-attributes-and-methods)
        - [Operations](#operations)
        - [Attributes](#attributes)
        - [Methods](#methods)
    - [Sparse tensor](#Sparse-tensor)
        - [RowTensor](#RowTensor)
        - [SparseTensor](#SparseTensor)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/tensor.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

Tensor is a basic data structure in the MindSpore network computing. For details about data types in tensors, see [dtype](https://www.mindspore.cn/docs/programming_guide/en/master/dtype.html).

Tensors of different dimensions represent different data. For example, a 0-dimensional tensor represents a scalar, a 1-dimensional tensor represents a vector, a 2-dimensional tensor represents a matrix, and a 3-dimensional tensor may represent the three channels of RGB images.

## Tensor Structure

During tensor creation, the `Tensor`, `float`, `int`, `bool`, `tuple`, `list`, `complex`, and `NumPy.array` types can be transferred, while `tuple` and `list` can only store `float`, `int`, `bool` and `complex` data, where `complex` represets the complex data types.

`dtype` can be specified when `Tensor` is initialized. When the `dtype` is not specified, if the initial value is `int`, `float`, `bool` or `complex`, then a 0-dimensional `Tensor` with data types `mindspore.int32`, `mindspore.float64` , `mindspore.bool_` or `mindspore.complex128` will be generated respectively. If the initial values are `tuple` and `list`, the generated 1-dimensional `Tensor` data type corresponds to the type stored in `tuple` and `list`. If it contains multiple different types of data, follow the below priority: `bool` < `int` < `float` < `complex`, to select the mindspore data type corresponding to the highest relative priority type. If the initial value is `Tensor`,  the consistent data type `Tensor` is generated. If the initial value is `NumPy.array`, the corresponding data type `Tensor` is generated.

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

## Tensor Operations, Attributes and Methods

### Operations

Tensor supports a variety of operations, including arithmetic operations and logical operations. When two arrays of different shapes are subjected to numerical operations, the `broadcast` mechanism similar to `Numpy` will be triggered. Some commonly used operators are as follows:

- arithmetic operations: add (`+`), subtract (`-`), multiply (`*`), divide (`/`), modulus (`%`), power (`**`), divide (`//`)

- logical operations：equal to (`==`), not equal to (`!=`), greater than (`>`), greater than or equal to (`>=`), less than (`<`), less than or equal to (`<=`)

A code example is as follows:

```python
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype

x = Tensor(np.array([1, 2, 3]), mstype.float32)
y = Tensor(np.array([4, 5, 6]), mstype.float32)
output_add = x + y
output_sub = x - y
output_mul = x * y
output_div = y / x
output_mod = x % y
output_pow = x ** 2
output_floordiv = y // x
print("add:", output_add)
print("sub:", output_sub)
print("mul:", output_mul)
print("div:", output_div)
print("mod:", output_mod)
print("pow:", output_pow)
print("floordiv:", output_floordiv)

a = Tensor(np.array([2, 2, 2]), mstype.int32)
b = Tensor(np.array([1, 2, 3]), mstype.int32)
output_eq = a == b
output_ne = a != b
output_gt = a > b
output_gq = a >= b
output_lt = a < b
output_lq = a <= b
print("equal:", output_eq)
print("not equal:", output_ne)
print("greater than:", output_gt)
print("greater or equal:", output_gq)
print("less than:", output_lt)
print("less or equal:", output_lq)
```

The following information is displayed:

```text
add: [5. 7. 9.]
sub: [-3. -3. -3.]
mul: [ 4. 10. 18.]
div: [4. 2.5 2. ]
mod: [1. 2. 3.]
pow: [1. 4. 9.]
floordiv: [4. 2. 2.]
equal: [False True False]
not equal: [ True False True]
greater than: [ True False False]
greater or equal: [ True True False]
less than: [False False True]
less or equal: [False True True]
```

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

Tensor methods include `len`, `str`, `repr`, `hash`, `all`, `any`, `asnumpy` and many other functions. Numpy-like ndarray methods are also provided. For a full description of all tensor methods, please see [API: mindspore.Tensor](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore/mindspore.Tensor.html). The following is a brief introduction to some of the tensor methods.

- `len()`: returns the length of the tensor.
- `str()`: returns the string representation of the tensor.
- `repr()`: returns the string representation of the tensor for the interpreter to read.
- `hash()`: get the hash value of the tensor.
- `all(axis, keep_dims)`: performs the `and` operation on a specified dimension to reduce the dimension. `axis` indicates the reduced dimension, and `keep_dims` indicates whether to retain the reduced dimension.
- `any(axis, keep_dims)`: performs the `or` operation on a specified dimension to reduce the dimension. The parameter meaning is the same as that of `all`.
- `asnumpy()`: converts `Tensor` to an array of NumPy.
- `sum(axis, dtype, keepdims, initial)`: sums the tensor over the given `axis`, `axis` indicates the reduced dimension, `dtype` specifies the output data type, `keepdims` indicates whether to retain the reduced dimension, and `initial` indicates the starting value for the sum.

A code example is as follows:

```python
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype

t = Tensor(np.array([1, 2, 3]), mstype.int32)
t_len = len(t)
t_str = str(t)
t_repr = repr(t)
t_hash = hash(t)
print("t_len:", t_len)
print("t_str:", t_str)
print("t_repr:", t_repr)
print("t_hash:", t_hash)

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
t_len: 3
t_str: [1 2 3]
t_repr: Tensor(shape=[3], dtype=Int32, value= [1, 2, 3])
t_hash: 281470264268272
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

`RowTensor` can only be used in the `Cell`’s construct method. For details, see [mindspore.RowTensor](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore/mindspore.RowTensor.html). A code example is as follows:

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

`SparseTensor` can only be used in the `Cell`’s construct method. For details, see [mindspore.SparseTensor](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore/mindspore.SparseTensor.html). A code example is as follows:

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
