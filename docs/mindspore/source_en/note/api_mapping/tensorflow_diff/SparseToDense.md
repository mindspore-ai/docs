# Function Differences with tf.raw_ops.SparseToDense

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/SparseToDense.md)

## tf.raw_ops.SparseToDense

```text
tf.raw_ops.SparseToDense(
    sparse_indices,
    output_shape,
    sparse_values,
    default_value,
    validate_indices=True,
    name=None
) -> Tensor
```

For more information, see [tf.raw_ops.SparseToDense](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/raw_ops/SparseToDense).

## mindspore.ops.SparseToDense

```text
class mindspore.ops.SparseToDense(
    indices,
    values,
    sparse_shape
) -> Tensor
```

For more information, see [mindspore.ops.SparseToDense](https://www.mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.SparseToDense.html).

## Differences

TensorFlow: SparseToDense converts a sparse representation of a Tensor to a dense Tensor.

MindSpore: MindSpore API basically implements the same functions as TensorFlow. TensorFlow default_value parameter can specify a default padding value, while MindSpore does not have this parameter, but it can be implemented by calling SparseToDense twice. The principle is to first create a temporary dense Tensor using SparseToDense, then add a default_value consistent with TensorFlow to each element, and finally subtract the default_value from the position set by the indices to get the target result.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1  | sparse_indices            | indices       | Same function, different parameter names         |
|      | Parameter 2  | output_shape              | sparse_shape   | Same function, different parameter names       |
|      | Parameter 3  | sparse_values             | values         | Same function, different parameter names      |
|      | Parameter 4  | default_value             | -             | MindSpore does not have this parameter, but you can call SparseToDense twice to achieve the same function     |
|      | Parameter 5  | validate_indices          | -             | Not involved        |
|      | Parameter 6  | name                      | -        | Not involved |

### Code Example 1

> The two APIs achieve the same function when 0 is padded by default.

```python
# TensorFlow
import tensorflow as tf

indices = tf.constant([[0, 1], [1, 2], [2, 3]], dtype=tf.int64)
values = tf.constant([1, 2, 3], dtype=tf.float32)
shape = tf.constant([3, 4], dtype=tf.int64)
sparse_tensor = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)

default_value = tf.constant(0, dtype=tf.float32)
out = tf.raw_ops.SparseToDense(sparse_indices=sparse_tensor.indices,
                                       output_shape=sparse_tensor.dense_shape,
                                       sparse_values=sparse_tensor.values,
                                       default_value=default_value)
print(out)
# tf.Tensor(
# [[0. 1. 0. 0.]
#  [0. 0. 2. 0.]
#  [0. 0. 0. 3.]], shape=(3, 4), dtype=float32)

# MindSpore
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

indices = Tensor([[0, 1], [1, 2], [2, 3]], dtype=mindspore.int64)
values = Tensor([1, 2, 3], dtype=mindspore.float32)
sparse_shape = (3, 4)
out = ops.SparseToDense()(indices, values, sparse_shape)
print(out)
# [[0. 1. 0. 0.]
#  [0. 0. 2. 0.]
#  [0. 0. 0. 3.]]
```

### Code Example 2

> TensorFlow default_value parameter can specify a default padding value, while MindSpore implements functions with two calls to SparseToDense. First create a temporary dense Tensor using SparseToDense, then add a default_value consistent with TensorFlow to each element, subtract the default_value from the element at the location specified by indices.

```python
# TensorFlow
import tensorflow as tf

indices = tf.constant([[0, 1], [1, 2], [2, 3]], dtype=tf.int64)
values = tf.constant([1, 2, 3], dtype=tf.float32)
shape = tf.constant([3, 4], dtype=tf.int64)
sparse_tensor = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)

default_value = tf.constant(5, dtype=tf.float32)
dense_tensor = tf.raw_ops.SparseToDense(sparse_indices=sparse_tensor.indices,
                                       output_shape=sparse_tensor.dense_shape,
                                       sparse_values=sparse_tensor.values,
                                       default_value=default_value)
print(dense_tensor)
# tf.Tensor(
# [[5. 1. 5. 5.]
#  [5. 5. 2. 5.]
#  [5. 5. 5. 3.]], shape=(3, 4), dtype=float32)

# MindSpore
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

indices = Tensor([[0, 1], [1, 2], [2, 3]], dtype=mindspore.int64)
values = Tensor([1, 2, 3], dtype=mindspore.float32)
sparse_shape = (3, 4)
default_value = 5

out_plus_default = ops.SparseToDense()(indices, values, sparse_shape) + default_value

values = Tensor([5, 5, 5], dtype=mindspore.float32)
temp = ops.SparseToDense()(indices, values, sparse_shape)
out = out_plus_default - temp
print(out)
# [[5. 1. 5. 5.]
#  [5. 5. 2. 5.]
#  [5. 5. 5. 3.]]
```
