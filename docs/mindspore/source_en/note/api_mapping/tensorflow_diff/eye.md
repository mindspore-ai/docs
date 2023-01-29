# Function Differences with tf.eye

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/eye.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.eye

```text
tf.eye(
    num_rows,
    num_columns=None,
    batch_shape=None,
    dtype=tf.dtypes.float32,
    name=None
) -> Tensor
```

For more information, see [tf.eye](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/eye).

## mindspore.ops.eye

```text
mindspore.ops.eye(n, m, t) -> Tensor
```

For more information, see [mindspore.ops.eye](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.eye.html).

## Differences

TensorFlow: It is possible to accept `batch_shape` in the parameters in TensorFlow to make the output have such a shape.

MindSpore: The number of columns and data types cannot be defaulted, and there is no difference in function.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1 | num_rows    | n         | Same function, different parameter names                                        |
|      | Parameter 2 | num_columns | m         | Specifies the number of columns of the tensor. Optional in TensorFlow; without this parameter, a tensor with the same number of columns and rows is returned; required in MindSpore |
|      | Parameter 3 | batch_shape | -       | Makes the output have the specified shape. MindSpore does not have this parameter. For example, `batch_shape=[3]` |
|      | Parameter 4 | dtype       | t         | The name is different, optional in TensorFlow. If not, the default is `tf.dtypes.float32`; required in MindSpore |
|      | Parameter 5 | name       | -        | Not involved |

## Differences Analysis and Examples

### Code Example 1

> TensorFlow can default `num_columns`, and MindSpore cannot default.

```python
# TensorFlow
import tensorflow as tf

e1 = tf.eye(3)
print(e1.numpy())
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# MindSpore
import mindspore
import mindspore.ops as ops
e1 = ops.eye(3, 3, mindspore.float32)
print(e1.numpy())
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

### Code Example 2

> TensorFlow can default `dtype`, and MindSpore cannot default.

```python
# TensorFlow
import tensorflow as tf
e2 = tf.eye(3, 2)
print(e2.numpy())
# [[1. 0.]
#  [0. 1.]
#  [0. 0.]]

# MindSpore
import mindspore
import mindspore.ops as ops
e2 = ops.eye(3, 2, mindspore.float32)
print(e2)
# [[1. 0.]
#  [0. 1.]
#  [0. 0.]]
```



