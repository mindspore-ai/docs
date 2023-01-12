<a href="https://gitee.com/mindspore/docs/blob/r1.9/tutorials/source_en/beginner/tensor.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png"></a>

[Introduction](https://www.mindspore.cn/tutorials/en/r1.9/beginner/introduction.html) || [Quick Start](https://www.mindspore.cn/tutorials/en/r1.9/beginner/quick_start.html) || **Tensor** || [Dataset](https://www.mindspore.cn/tutorials/en/r1.9/beginner/dataset.html) || [Transforms](https://www.mindspore.cn/tutorials/en/r1.9/beginner/transforms.html) || [Model](https://www.mindspore.cn/tutorials/en/r1.9/beginner/model.html) || [Autograd](https://www.mindspore.cn/tutorials/en/r1.9/beginner/autograd.html) || [Train](https://www.mindspore.cn/tutorials/en/r1.9/beginner/train.html) || [Save and Load](https://www.mindspore.cn/tutorials/en/r1.9/beginner/save_load.html)

# Tensor

Tensor is a multilinear function that can be used to represent linear relationships between vectors, scalars, and other tensors. The basic examples of these linear relations are the inner product, the outer product, the linear map, and the Cartesian product. In the $n$ dimensional space, its coordinates have $n^{r}$ components. Each component is a function of coordinates, and these components are also linearly transformed according to certain rules when the coordinates are transformed. $r$ is called the rank or order of this tensor (not related to the rank or order of the matrix).

A tensor is a special data structure that is similar to arrays and matrices. [Tensor](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore/mindspore.Tensor.html) is the basic data structure in MindSpore network operations. This tutorial describes the attributes and usage of tensors and sparse tensors.

```python
import numpy as np
import mindspore
from mindspore import ops
from mindspore import Tensor, CSRTensor, COOTensor, RowTensor
```

## Creating a Tensor

There are multiple methods for creating tensors. When building a tensor, you can pass the `Tensor`, `float`, `int`, `bool`, `tuple`, `list`, and `numpy.ndarray` types.

- **Generating a tensor based on data**

You can create a tensor based on data. The data type can be set or automatically inferred by the framework.

```python
data = [1, 0, 1, 0]
x_data = Tensor(data)
```

- **Generating a tensor from the NumPy array**

You can create a tensor from the NumPy array.

```python
np_array = np.array(data)
x_np = Tensor(np_array)
```

- **Generating a tensor by using init**

When `init` is used to initialize a tensor, the `init`, `shape`, and `dtype` parameters can be transferred.

- `init`: supports the subclass of [initializer](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore.common.initializer.html).

- `shape`: supports `list`, `tuple`, and `int`.

- `dtype`: supports [mindspore.dtype](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore.html#mindspore.dtype).

```python
from mindspore.common.initializer import One, Normal

# Initialize a tensor with ones
tensor1 = mindspore.Tensor(shape=(2, 2), dtype=mindspore.float32, init=One())
# Initialize a tensor from normal distribution
tensor2 = mindspore.Tensor(shape=(2, 2), dtype=mindspore.float32, init=Normal())

print("tensor1:\n", tensor1)
print("tensor2:\n", tensor2)
```

```text
tensor1:
 [[1. 1.]
 [1. 1.]]
tensor2:
 [[-0.00063482 -0.00916224]
 [ 0.01324238 -0.0171206 ]]
```

The `init` is used for delayed initialization in parallel mode. Usually, it is not recommended to use `init` interface to initialize parameters.

- **Inheriting attributes of another tensor to form a new tensor**

```python
from mindspore import ops

x_ones = ops.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_zeros = ops.zeros_like(x_data)
print(f"Zeros Tensor: \n {x_zeros} \n")
```

```text
Ones Tensor:
 [1 1 1 1]

Zeros Tensor:
 [0 0 0 0]
```

## Tensor Attributes

Tensor attributes include shape, data type, transposed tensor, item size, number of bytes occupied, dimension, size of elements, and stride per dimension.

- shape: the shape of `Tensor`, a tuple.

- dtype: the dtype of `Tensor`, a data type of MindSpore.

- itemsize: the number of bytes occupied by each element in `Tensor`, which is an integer.

- nbytes: the total number of bytes occupied by `Tensor`, which is an integer.

- ndim: rank of `Tensor`, that is, len(tensor.shape), which is an integer.

- size: the number of all elements in `Tensor`, which is an integer.

- strides: the number of bytes to traverse in each dimension of `Tensor`, which is a tuple.

```python
x = Tensor(np.array([[1, 2], [3, 4]]), mindspore.int32)

print("x_shape:", x.shape)
print("x_dtype:", x.dtype)
print("x_itemsize:", x.itemsize)
print("x_nbytes:", x.nbytes)
print("x_ndim:", x.ndim)
print("x_size:", x.size)
print("x_strides:", x.strides)
```

```text
x_shape: (2, 2)
x_dtype: Int32
x_itemsize: 4
x_nbytes: 16
x_ndim: 2
x_size: 4
x_strides: (8, 4)
```

## Tensor Indexing

Tensor indexing is similar to NumPy indexing. Indexing starts from 0, negative indexing means indexing in reverse order, and colons `:` and `...` are used for slicing.

```python
tensor = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))

print("First row: {}".format(tensor[0]))
print("value of bottom right corner: {}".format(tensor[1, 1]))
print("Last column: {}".format(tensor[:, -1]))
print("First column: {}".format(tensor[..., 0]))
```

```text
First row: [0. 1.]
value of bottom right corner: 3.0
Last column: [1. 3.]
First column: [0. 2.]
```

## Tensor Operation

There are many operations between tensors, including arithmetic, linear algebra, matrix processing (transposing, indexing, and slicing), and sampling. The usage of tensor operation is similar to that of NumPy. The following describes several operations.

> Common arithmetic operations include: addition (+), subtraction (-), multiplication (\*), division (/), modulo (%), and exact division (//).

```python
x = Tensor(np.array([1, 2, 3]), mindspore.float32)
y = Tensor(np.array([4, 5, 6]), mindspore.float32)

output_add = x + y
output_sub = x - y
output_mul = x * y
output_div = y / x
output_mod = y % x
output_floordiv = y // x

print("add:", output_add)
print("sub:", output_sub)
print("mul:", output_mul)
print("div:", output_div)
print("mod:", output_mod)
print("floordiv:", output_floordiv)
```

```text
add: [5. 7. 9.]
sub: [-3. -3. -3.]
mul: [ 4. 10. 18.]
div: [4.  2.5 2. ]
mod: [0. 1. 0.]
floordiv: [4. 2. 2.]
```

`Concat` connects a series of tensors in a given dimension.

```python
data1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
data2 = Tensor(np.array([[4, 5], [6, 7]]).astype(np.float32))
output = ops.concat((data1, data2), axis=0)

print(output)
print("shape:\n", output.shape)
```

```text
[[0. 1.]
 [2. 3.]
 [4. 5.]
 [6. 7.]]
shape:
 (4, 2)
```

`Stack` combines two tensors from another dimension.

```python
data1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
data2 = Tensor(np.array([[4, 5], [6, 7]]).astype(np.float32))
output = ops.stack([data1, data2])

print(output)
print("shape:\n", output.shape)
```

```text
[[[0. 1.]
  [2. 3.]]

 [[4. 5.]
  [6. 7.]]]
shape:
 (2, 2, 2)
```

## Conversion Between Tensor and NumPy

Tensor and NumPy can be converted to each other.

### Tensor to NumPy

Use `asnumpy()` to convert Tensor to NumPy, which is same as tensor building.

```python
t = ops.ones(5, mindspore.float32)
print(f"t: {t}")
n = t.asnumpy()
print(f"n: {n}")
```

```text
t: [1. 1. 1. 1. 1.]
n: [1. 1. 1. 1. 1.]
```

### NumPy to Tensor

Use `Tensor()` to convert NumPy to Tensor.

```python
n = np.ones(5)
t = Tensor.from_numpy(n)
```

```python
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```

```text
t: [2. 2. 2. 2. 2.]
n: [2. 2. 2. 2. 2.]
```

## Sparse Tensor

A sparse tensor is a special tensor in which the value of the most significant element is zero.

In some scenarios (such as recommendation systems, molecular dynamics, graph neural networks), the data is sparse. If you use common dense tensors to represent the data, you may introduce many unnecessary calculations, storage, and communication costs. In this case, it is better to use sparse tensor to represent the data.

MindSpore now supports the two most commonly used `CSR` and `COO` sparse data formats.

The common structure of the sparse tensor is `<indices:Tensor, values:Tensor, shape:Tensor>`. `indices` means the indexes of non-zero elements, `values` means the values of non-zero elements, and `shape` means the dense shape of the sparse tensor. In this structure, we define data structure `CSRTensor`, `COOTensor`, and `RowTensor`.

### CSRTensor

The compressed sparse row (`CSR`) is efficient in both storage and computation. All the non-zero values are stored in `values`, and their positions are stored in `indptr` (row) and `indices` (column). The meaning of each parameter is as follows:

- `indptr`: 1-D integer tensor, indicating the start and end points of the non-zero elements in each row of the sparse data in `values`. The index data type can only be int32.

- `indices`: 1-D integer tensor, indicating the position of the sparse tensor non-zero elements in the column and has the same length as `values`. The index data type can only be int32.

- `values`: 1-D tensor, indicating that the value of the non-zero element corresponding to the `CSRTensor` and has the same length as `indices`.

- `shape`: indicates the shape of a compressed sparse tensor. The data type is `Tuple`. Currently, only 2-D `CSRTensor` is supported.

> For details about `CSRTensor`, see [mindspore.CSRTensor](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore/mindspore.CSRTensor.html).

The following are some examples of using the CSRTensor:

```python
indptr = Tensor([0, 1, 2])
indices = Tensor([0, 1])
values = Tensor([1, 2], dtype=mindspore.float32)
shape = (2, 4)

# Make a CSRTensor
csr_tensor = CSRTensor(indptr, indices, values, shape)

print(csr_tensor.astype(mindspore.float64).dtype)
```

```text
Float64
```

The above code generates a `CSRTensor` as shown in the following equation:

$$
 \left[
 \begin{matrix}
   1 & 0 & 0 & 0 \\
   0 & 2 & 0 & 0
  \end{matrix}
  \right]
$$

### COOTensor

The `COO` (Coordinate Format) sparse tensor format is used to represent a collection of nonzero elements of a tensor on a given index. If the number of non-zero elements is `N` and the dimension of the compressed tensor is `ndims`. The meaning of each parameter is as follows:

- `indices`: 2-D integer tensor. Each row indicates a non-zero element subscript. Shape: `[N, ndims]`. The index data type can only be int32.

- `values`: 1-D tensor of any type, indicating the value of the non-zero element. Shape: `[N]`.

- `shape`: indicates the shape of a compressed sparse tensor. Currently, only 2-D `COOTensor` is supported.

> For details about `COOTensor`, see [mindspore.COOTensor](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore/mindspore.COOTensor.html).

The following are some examples of using COOTensor:

```python
indices = Tensor([[0, 1], [1, 2]], dtype=mindspore.int32)
values = Tensor([1, 2], dtype=mindspore.float32)
shape = (3, 4)

# Make a COOTensor
coo_tensor = COOTensor(indices, values, shape)

print(coo_tensor.values)
print(coo_tensor.indices)
print(coo_tensor.shape)
print(coo_tensor.astype(mindspore.float64).dtype)  # COOTensor to float64
```

```text
[1. 2.]
[[0 1]
 [1 2]]
(3, 4)
Float64
```

The preceding code generates `COOTensor` as follows:

$$
 \left[
 \begin{matrix}
   0 & 1 & 0 & 0 \\
   0 & 0 & 2 & 0 \\
   0 & 0 & 0 & 0
  \end{matrix}
  \right]
$$

### RowTensor

`RowTensor` is used to compress tensors that are sparse in the dimension 0. If the dimension of `RowTensor` is `[L0, D1, D2, ..., DN ]` and the number of non-zero elements in the dimension 0 is `D0`, then `L0 >> D0`. The meaning of each parameter is as follows:

- `indices`: 1-D integer tensor, indicating the position of non-zero elements in the dimension 0 of the sparse tensor. The shape is `[D0]`.

- `values`: indicating the value of the corresponding non-zero element. The shape is `[D0, D1, D2, ..., DN]`.

- `dense_shape`: indicates the shape of a compressed sparse tensor.

> For the detailed documentation of `RowTensor`, see the code example in [mindspore.RowTensor](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore/mindspore.RowTensor.html).

A code example is as follows:

```python
indices = Tensor([0])
values = Tensor([[1, 2]], dtype=mindspore.float32)
dense_shape = (3, 2)

x = RowTensor(indices, values, dense_shape)

print("non-zero values:", x.values)
print("non-zero indices:", x.indices)
print("shape:", x.dense_shape)
```

```text
non-zero values: [[1. 2.]]
non-zero indices: [0]
shape: (3, 2)
```

The preceding code generates `RowTensor` as shown in the following equation:

$$
 \left[
 \begin{matrix}
   1 & 2 \\
   0 & 0 \\
   0 & 0
  \end{matrix}
  \right]
$$
