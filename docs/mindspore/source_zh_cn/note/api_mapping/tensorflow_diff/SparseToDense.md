# 比较与tf.raw_ops.SparseToDense的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/SparseToDense.md)

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

更多内容详见[tf.raw_ops.SparseToDense](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/raw_ops/SparseToDense)。

## mindspore.ops.SparseToDense

```text
class mindspore.ops.SparseToDense(
    indices,
    values,
    sparse_shape
) -> Tensor
```

更多内容详见[mindspore.ops.SparseToDense](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.SparseToDense.html)。

## 差异对比

TensorFlow：SparseToDense将Tensor的稀疏表示转换为密集Tensor。

MindSpore：MindSpore此API实现功能与TensorFlow基本一致。不过TensorFlow的default_value参数可以指定默认填充值，MindSpore没有这个参数，但是可以两次调用SparseToDense来实现，原理是首先使用SparseToDense创建一个临时dense Tensor，然后将其中每个元素加上与TensorFlow一致的default_value，最后再将indices制定的位置减去default_value即得到目标结果。

| 分类 | 子类   | TensorFlow                | MindSpore     | 差异                      |
| ---- | ------ | --------------------- -- | ------------- | ------------------------ |
| 参数 | 参数1  | sparse_indices            | indices       | 功能一致，参数名不同         |
|      | 参数2  | output_shape              | sparse_shape   | 功能一致，参数名不同       |
|      | 参数3  | sparse_values             | values         | 功能一致，参数名不同      |
|      | 参数4  | default_value             | -             | MindSpore无此参数，但是可以两次调用SparseToDense实现同样的功能     |
|      | 参数5  | validate_indices          | -             | 不涉及        |
|      | 参数6  | name                      | -        | 不涉及 |

### 代码示例1

> 默认充值0时，两API实现功能一致。

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

### 代码示例2

> TensorFlow的default_value参数可以指定默认填充值。MindSpore以两次调用SparseToDense来实现这一功能。首先使用SparseToDense创建一个临时dense Tensor，然后将其中每个元素加上与TensorFlow一致的default_value。然后将indices指定位置的元素减去default_value即可。

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
