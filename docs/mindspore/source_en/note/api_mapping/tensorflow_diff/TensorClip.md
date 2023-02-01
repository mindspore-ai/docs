# Function Differences with tf.clip_by_value

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/TensorClip.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.clip_by_value

```python
tf.clip_by_value(
    t, clip_value_min, clip_value_max, name=None
)
```

For more information, see [tf.clip_by_value](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/clip_by_value).

## mindspore.Tensor.clip

```python
mindspore.Tensor.clip(xmin, xmax, dtype=None)
```

For more information, see [mindspore.Tensor.clip](https://www.mindspore.cn/docs/en/master/api_python/mindspore/Tensor/mindspore.Tensor.clip.html#mindspore.Tensor.clip).

## Usage

The main functions are the same. `tf.clip_by_value` throws a type error when `t` is `int32` and `clip_value_min` or `clip_value_max` is of type `float32`, and `mindspore.Tensor.clip` does not have this restriction.

## Code Example

```python
import mindspore as ms

x = ms.Tensor([1, 2, 3, -4, 0, 3, 2, 0]).astype(ms.int32)
print(x.clip(0, 2))
# [1 2 2 0 0 2 2 0]
print(x.clip(0., 2.))
# [1 2 2 0 0 2 2 0]
print(x.clip(Tensor([1, 1, 1, 1, 1, 1, 1, 1]), 2))
# [1 2 2 1 1 2 2 1]

import tensorflow as tf
tf.enable_eager_execution()

A = tf.constant([1, 2, 3, -4, 0, 3, 2, 0])
B = tf.clip_by_value(A, clip_value_min=0, clip_value_max=2)
print(B.numpy())
# [1 2 2 0 0 2 2 0]
C = tf.clip_by_value(A, clip_value_min=0., clip_value_max=2.)
# throws `TypeError`
D = tf.clip_by_value(A, [1, 1, 1, 1, 1, 1, 1, 1], 2)
print(D.numpy())
# [1 2 2 1 1 2 2 1]
```

