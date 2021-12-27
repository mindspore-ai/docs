# Tensor

`Ascend` `GPU` `CPU` `Beginner`

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/tensor.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

Tensor is a basic data structure in the MindSpore network computing.

Import the required modules and APIs:

```python
import numpy as np
from mindspore import Tensor, context
from mindspore import dtype as mstype
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
```

## Initializing a Tensor

There are multiple methods for initializing tensors. When building a tensor, you can pass the [Tensor](https://www.mindspore.cn/docs/api/en/master/api_python/mindspore/mindspore.Tensor.html), `float`, `int`, `bool`, `tuple`, `list`, and `NumPy.array` types.

- **Generate a tensor based on data.**

You can create a tensor based on data. The data type can be set or automatically inferred.

```python
x = Tensor(0.1)
```

- **Generate a tensor from the NumPy array.**

You can create a tensor from the NumPy array.

```python
arr = np.array([1, 0, 1, 0])
x_np = Tensor(arr)
```

If the initial value is `NumPy.array`, the generated `Tensor` data type corresponds to `NumPy.array`.

- **Generate a tensor from the init**

You can create a tensor with the `init`, `shape` and `dtype`.

- `init`: Supported subclasses of incoming Subclass of [initializer](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.common.initializer.html).
- `shape`: Supported subclasses of incoming `list`, `tuple`, `int`.
- `dtype`: Supported subclasses of incoming [mindspore.dtype](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.html#mindspore.dtype).

```python
from mindspore import Tensor
from mindspore import set_seed
from mindspore import dtype as mstype
from mindspore.common.initializer import One, Normal

set_seed(1)

tensor1 = Tensor(shape=(2, 2), dtype=mstype.float32, init=One())
tensor2 = Tensor(shape=(2, 2), dtype=mstype.float32, init=Normal())
print(tensor1)
print(tensor2)
```

```text
    [[1. 1.]
     [1. 1.]]
    [[-0.00128023 -0.01392901]
     [ 0.0130886  -0.00107818]]
```

The `init` is used for delayed initialization in parallel mode. Usually, it is not recommended to use `init` interface to initialize parameters in other conditions.

- **Inherit attributes of another tensor to form a new tensor.**

```python
from mindspore import ops
oneslike = ops.OnesLike()
x = Tensor(np.array([[0, 1], [2, 1]]).astype(np.int32))
output = oneslike(x)
print(output)
```

```text
    [[1 1]
     [1 1]]
```

- **Output a constant tensor of a specified size.**

`shape` is the size tuple of a tensor, which determines the dimension of the output tensor.

```python
import mindspore.ops as ops

shape = (2, 2)
ones = ops.Ones()
output = ones(shape, mstype.float32)
print(output)

zeros = ops.Zeros()
output = zeros(shape, mstype.float32)
print(output)
```

```text
    [[1. 1.]
     [1. 1.]]
    [[0. 0.]
     [0. 0.]]
```

During `Tensor` initialization, dtype can be specified to, for example, `mstype.int32`, `mstype.float32` or `mstype.bool_`.

## Tensor Attributes

Tensor attributes include shape and data type (dtype).

- shape: a tuple
- dtype: a data type of MindSpore

```python
t1 = Tensor(np.zeros([1,2,3]), mstype.float32)
print("Datatype of tensor: {}".format(t1.dtype))
print("Shape of tensor: {}".format(t1.shape))
```

```text
    Datatype of tensor: Float32
    Shape of tensor: (1, 2, 3)
```

## Tensor Operation

There are many operations between tensors, including arithmetic, linear algebra, matrix processing (transposing, indexing, and slicing), and sampling. The following describes several operations. The usage of tensor computation is similar to that of NumPy.

Indexing and slicing operations similar to NumPy:

```python
tensor = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
print("First row: {}".format(tensor[0]))
print("First column: {}".format(tensor[:, 0]))
print("Last column: {}".format(tensor[..., -1]))
```

```text
    First row: [0. 1.]
    First column: [0. 2.]
    Last column: [1. 3.]
```

`Concat` connects a series of tensors in a given dimension.

```python
data1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
data2 = Tensor(np.array([[4, 5], [6, 7]]).astype(np.float32))
op = ops.Concat()
output = op((data1, data2))
print(output)
```

```text
    [[0. 1.]
     [2. 3.]
     [4. 5.]
     [6. 7.]]
```

`Stack` combines two tensors from another dimension.

```python
data1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
data2 = Tensor(np.array([[4, 5], [6, 7]]).astype(np.float32))
op = ops.Stack()
output = op([data1, data2])
print(output)
```

```text
    [[[0. 1.]
      [2. 3.]]

     [[4. 5.]
      [6. 7.]]]
```

Common computation:

```python
input_x = Tensor(np.array([1.0, 2.0, 3.0]), mstype.float32)
input_y = Tensor(np.array([4.0, 5.0, 6.0]), mstype.float32)
mul = ops.Mul()
output = mul(input_x, input_y)
print(output)
```

```text
    [ 4. 10. 18.]
```

## Conversion Between Tensor and NumPy

Tensor and NumPy can be converted to each other.

### Tensor to NumPy

```python
zeros = ops.Zeros()
output = zeros((2,2), mstype.float32)
print("output: {}".format(type(output)))
n_output = output.asnumpy()
print("n_output: {}".format(type(n_output)))
```

```text
    output: <class 'mindspore.common.tensor.Tensor'>
    n_output: <class 'numpy.ndarray'>
```

### NumPy to Tensor

```python
output = np.array([1, 0, 1, 0])
print("output: {}".format(type(output)))
t_output = Tensor(output)
print("t_output: {}".format(type(t_output)))
```

```text
    output: <class 'numpy.ndarray'>
    t_output: <class 'mindspore.common.tensor.Tensor'>
```
