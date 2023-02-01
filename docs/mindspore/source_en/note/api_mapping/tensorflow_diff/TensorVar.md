# Function Differences with tf.math.reduce_variance

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/TensorVar.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.math.reduce_variance

```python
tf.math.reduce_variance(input_tensor, axis=None, keepdims=False, name=None)
```

For more information, see [tf.math.reduce_variance](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/math/reduce_variance).

## mindspore.Tensor.var

```python
mindspore.Tensor.var(axis=None, ddof=0, keepdims=False)
```

For more information, see [mindspore.Tensor.var](https://mindspore.cn/docs/en/master/api_python/mindspore/Tensor/mindspore.Tensor.var.html#mindspore.Tensor.var).

## Usage

The basic function of the two interfaces is the same. Both calculate the variance of the Tensor in some dimension, calculated as: var = mean(x), where x = abs(a - a.mean())**2.

The difference is that `mindspore.Tensor.var` has one more input parameter `ddof`. In general, the mean value is x.sum() / N, where N=len(x), and if `ddof` is configured, the denominator will change from N to N-ddof.

## Code Example

```python
import mindspore as ms
import numpy as np

a = ms.Tensor(np.array([[1, 2], [3, 4]]), ms.float32)
print(a.var()) # 1.25
print(a.var(axis=0)) # [1. 1.]
print(a.var(axis=1)) # [0.25 0.25]
print(a.var(ddof=1)) # 1.6666666

import tensorflow as tf
tf.enable_eager_execution()

x = tf.constant([[1., 2.], [3., 4.]])
print(tf.math.reduce_variance(x).numpy())  # 1.25
print(tf.math.reduce_variance(x, 0).numpy())  # [1., 1.]
print(tf.math.reduce_variance(x, 1).numpy())  # [0.25,  0.25]
```
