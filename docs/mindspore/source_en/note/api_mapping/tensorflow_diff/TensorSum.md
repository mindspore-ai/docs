# Function Differences with tf.math.reduce_sum

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/TensorSum.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.math.reduce_sum

```python
tf.math.reduce_sum(
    input_tensor, axis=None, keepdims=None, name=None, reduction_indices=None,
    keep_dims=None
)
```

For more information, see [tf.math.reduce_sum](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/math/reduce_sum).

## mindspore.Tensor.sum

```python
mindspore.Tensor.sum(self, axis=None, dtype=None, keepdims=False, initial=None)
```

For more information, see [mindspore.Tensor.sum](https://mindspore.cn/docs/en/master/api_python/mindspore/Tensor/mindspore.Tensor.sum.html#mindspore.Tensor.sum).

## Usage

Both interfaces have the same basic function of computing the sum of Tensor in some dimension. The difference is that `mindspore.Tensor.sum` has one more parameter `initial` to set the starting value.

## Code Example

```python
import mindspore as ms

a = ms.Tensor([10, -5], ms.float32)
print(a.sum()) # 5.0
print(a.sum(initial=2)) # 7.0

import tensorflow as tf
tf.enable_eager_execution()

b = tf.constant([10, -5])
print(tf.math.reduce_sum(b).numpy()) # 5
```
