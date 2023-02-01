# Function Differences with tf.math.reduce_std

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/TensorStd.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.math.reduce_std

```python
tf.math.reduce_std(input_tensor, axis=None, keepdims=False, name=None)
```

For more information, see [tf.math.reduce_std](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/math/reduce_std).

## mindspore.Tensor.std

```python
mindspore.Tensor.std(self, axis=None, ddof=0, keepdims=False)
```

For more information, see [mindspore.Tensor.std](https://mindspore.cn/docs/en/master/api_python/mindspore/Tensor/mindspore.Tensor.std.html#mindspore.Tensor.std).

## Usage

The basic function of the two interfaces is the same. Both calculate the standard deviation of the Tensor in some dimension, calculated as: std = sqrt(mean(x)), where x = abs(a - a.mean())**2.

The difference is that `mindspore.Tensor.std` has one more input parameter `ddof`. In general, the mean value is x.sum() / N, where N=len(x), and if `ddof` is configured, the denominator will change from N to N-ddof.

## Code Example

```python
import mindspore as ms
import numpy as np

a = ms.Tensor(np.array([[1, 2], [3, 4]]), ms.float32)
print(a.std()) # 1.118034
print(a.std(axis=0)) # [1. 1.]
print(a.std(axis=1)) # [0.5 0.5]
print(a.std(ddof=1)) # 1.2909944

import tensorflow as tf
tf.enable_eager_execution()

x = tf.constant([[1., 2.], [3., 4.]])
print(tf.math.reduce_std(x).numpy())  # 1.118034
print(tf.math.reduce_std(x, 0).numpy())  # [1., 1.]
print(tf.math.reduce_std(x, 1).numpy())  # [0.5,  0.5]
```
