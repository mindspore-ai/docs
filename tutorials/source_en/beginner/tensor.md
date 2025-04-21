[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_en/beginner/tensor.md)

[Introduction](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/introduction.html) || [Quick Start](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/quick_start.html) || **Tensor** || [Data Loading and Processing](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/dataset.html) || [Model](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/model.html) || [Autograd](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/autograd.html) || [Train](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/train.html) || [Save and Load](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/save_load.html) || [Accelerating with Static Graphs](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/accelerate_with_static_graph.html) || [Mixed Precision](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/mixed_precision.html)

# Tensor

Tensor is a multilinear function that can be used to represent linear relationships between vectors, scalars, and other tensors. The basic examples of these linear relations are the inner product, the outer product, the linear map, and the Cartesian product. In the $n$ dimensional space, its coordinates have $n^{r}$ components. Each component is a function of coordinates, and these components are also linearly transformed according to certain rules when the coordinates are transformed. $r$ is called the rank or order of this tensor (not related to the rank or order of the matrix).

A tensor is a special data structure that is similar to arrays and matrices. [Tensor](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/mindspore/mindspore.Tensor.html) is the basic data structure in MindSpore network operations. This tutorial describes the attributes and usage of tensors.

```python
import numpy as np
import mindspore
from mindspore import ops
from mindspore import Tensor
```

## Creating a Tensor

There are multiple methods for creating tensors. When building a tensor, you can pass the `Tensor`, `float`, `int`, `bool`, `tuple`, `list`, and `numpy.ndarray` types.

- **Generating a tensor based on data**

    You can create a tensor based on data. The data type can be set or automatically inferred by the framework.

    ```python
    data = [1, 0, 1, 0]
    x_data = Tensor(data)
    print(x_data, x_data.shape, x_data.dtype)
    ```

    ```text
    [1 0 1 0] (4,) Int4
    ```

- **Generating a tensor from the NumPy array**

    You can create a tensor from the NumPy array.

    ```python
    np_array = np.array(data)
    x_np = Tensor(np_array)
    print(x_np, x_np.shape, x_np.dtype)
    ```

    ```text
    [1 0 1 0] (4,) Int4
    ```

- **Generating a tensor by using init**

    When `init` is used to initialize a tensor, the `init`, `shape`, and `dtype` parameters can be transferred.

    - `init`: supports the subclass of [initializer](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/mindspore.common.initializer.html). For example, [One()](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/mindspore.common.initializer.html#mindspore.common.initializer.One) and [Normal()](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/mindspore.common.initializer.html#mindspore.common.initializer.Normal) below.
    - `shape`: supports `list`, `tuple`, and `int`.
    - `dtype`: supports [mindspore.dtype](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/mindspore/mindspore.dtype.html#mindspore.dtype).

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

[concat](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/ops/mindspore.ops.concat.html) connects a series of tensors in a given dimension.

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

[stack](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/ops/mindspore.ops.stack.html) combines two tensors from another dimension.

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

Use [Tensor.asnumpy()](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/mindspore/Tensor/mindspore.Tensor.asnumpy.html) to convert Tensor to NumPy, which is same as tensor building.

```python
t = Tensor([1., 1., 1., 1., 1.])
print(f"t: {t}", type(t))
n = t.asnumpy()
print(f"n: {n}", type(n))
```

```text
t: [1. 1. 1. 1. 1.] <class 'mindspore.common.tensor.Tensor'>
n: [1. 1. 1. 1. 1.] <class 'numpy.ndarray'>
```

### NumPy to Tensor

Use `Tensor()` to convert NumPy to Tensor.

```python
n = np.ones(5)
t = Tensor.from_numpy(n)
```

```python
np.add(n, 1, out=n)
print(f"n: {n}", type(n))
print(f"t: {t}", type(t))
```

```text
n: [2. 2. 2. 2. 2.] <class 'numpy.ndarray'>
t: [2. 2. 2. 2. 2.] <class 'mindspore.common.tensor.Tensor'>
```