# Function Differences with tf.arg_min

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/TensorArgmin.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.arg_min

```python
tf.arg_min(input, dimension, output_type=tf.dtypes.int64, name=None)
```

For more information, see [tf.arg_min](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/arg_min).

## mindspore.Tensor.argmin

```python
mindspore.Tensor.argmin(axis=None)
```

For more information, see [mindspore.Tensor.argmin](https://mindspore.cn/docs/en/master/api_python/mindspore/Tensor/mindspore.Tensor.argmin.html#mindspore.Tensor.argmin).

## Usage

Same function. Two interfaces of MindSpore and TensorFlow decide on which dimension to return the index of the minimum value through the parameters `axis` and `dimension`, respectively.

The difference is that in the default state, `axis=None` of MindSpore returns the global index of the minimum value; TensorFlow `dimension` returns the minimum index of `dimension=0` by default when no value is passed in.

## Code Example

```python
import mindspore as ms

a = ms.Tensor([[1, 10, 166.32, 62.3], [1, -5, 2, 200]], ms.float32)
print(a.argmin())
print(a.argmin(axis=0))
print(a.argmin(axis=1))
# output:
# 5
# [0 1 1 0]
# [0 1]

import tensorflow as tf
tf.enable_eager_execution()

b = tf.constant([[1, 10, 166.32, 62.3], [1, -5, 2, 200]])
print(tf.argmin(b).numpy())
print(tf.argmin(b, dimension=0).numpy())
print(tf.argmin(b, dimension=1).numpy())
# output:
# [0 1 1 0]
# [0 1 1 0]
# [0 1]
```
