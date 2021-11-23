# 比较与tf.math.cumsum的功能差异

## tf.math.cumsum

```python
tf.math.cumsum(x, axis=0, exclusive=False, reverse=False, name=None)
```

更多内容详见[tf.math.cumsum](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/cumsum)。

## mindspore.Tensor.cumsum

```python
mindspore.Tensor.cumsum(self, axis=None, dtype=None)
```

更多内容详见[mindspore.Tensor.cumsum](https://mindspore.cn/docs/api/zh-CN/master/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor.cumsum)。

## 使用方式

> 在TensorFlow1.15版本中，tf.math.cumsum==tf.cumsum。

两接口基本功能相同，都是计算某个维度上Tensor的累加和(cumulative sum)。不同点在于，`tf.math.cumsum`多两个入参：`exclusive`用于指定是否包含当前值的累加，`reverse`用于指定返回值是否做逆转。

## 代码示例

```python
from mindspore import Tensor
import numpy as np

a = Tensor(np.ones((2,3)).astype("float32"))
print(a.cumsum(axis=0))
# [[1. 1. 1.]
#  [2. 2. 2.]]

import tensorflow as tf
tf.enable_eager_execution()

b = tf.constant(np.ones((2, 3)))
print(tf.cumsum(b, axis=0))
# tf.Tensor(
# [[1. 1. 1.]
#  [2. 2. 2.]], shape=(2, 3), dtype=float64)

print(tf.cumsum(b, exclusive=True))
# tf.Tensor(
# [[0. 0. 0.]
#  [1. 1. 1.]], shape=(2, 3), dtype=float64)

print(tf.cumsum(b, reverse=True))
# tf.Tensor(
# [[2. 2. 2.]
#  [1. 1. 1.]], shape=(2, 3), dtype=float64)
```
