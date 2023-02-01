# Function Differences with tf.arg_max

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/TensorArgmax.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.arg_max

```python
tf.arg_max(input, dimension, output_type=tf.dtypes.int64, name=None)
```

For more information, see [tf.arg_max](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/arg_max).

## mindspore.Tensor.argmax

```python
mindspore.Tensor.argmax(axis=None)
```

For more information, see [mindspore.Tensor.argmax](https://mindspore.cn/docs/en/master/api_python/mindspore/Tensor/mindspore.Tensor.argmax.html#mindspore.Tensor.argmax).

## Usage

Same function. Two interfaces of MindSpore and TensorFlow decide on which dimension to return the index of the maximum value through the parameters `axis` and `dimension`, respectively.

The difference is that in the default state, `axis=None` of MindSpore returns the global index of the maximum value; TensorFlow's `dimension` returns the maximum index of `dimension=0` by default when no value is passed in.

## Code Example

```python
import mindspore as ms

a = ms.Tensor([[1, 10, 166.32, 62.3], [1, -5, 2, 200]], ms.float32)
print(a.argmax())
print(a.argmax(axis=0))
print(a.argmax(axis=1))
# output:
# 7
# [0 0 0 1]
# [2 3]

import tensorflow as tf
tf.enable_eager_execution()

b = tf.constant([[1, 10, 166.32, 62.3], [1, -5, 2, 200]])
print(tf.argmax(b).numpy())
print(tf.argmax(b, dimension=0).numpy())
print(tf.argmax(b, dimension=1).numpy())
# output:
# [0 0 0 1]
# [0 0 0 1]
# [2 3]
```

