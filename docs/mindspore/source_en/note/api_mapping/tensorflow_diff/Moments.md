# Function Differences with tf.nn.moments

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/Moments.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.nn.moments

```text
tf.nn.moments(x, axes, shift=None, keepdims=False, name=None) -> Tensor
```

For more information, see [tf.nn.moments](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/moments).

## mindspore.nn.Moments

```text
class mindspore.nn.Moments(axis=None, keep_dims=None)(x) -> Tensor
```

For more information, see [mindspore.nn.Moments](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Moments.html).

## Differences

TensorFlow: Calculates the mean and variance of the input x along the specified axis axes.

MindSpore: MindSpore API implements the same function as TensorFlow, and only the parameter names are different.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | parameter 1 | x          | x         | -                                             |
| | parameter 2 | axes        | axis      | Same function, different parameter names                                   |
| | parameter 3 |   shift            | -         | This parameter is not used in the current implementation of TensorFlow and is useless; MindSpore does not have this parameter |
| | parameter 4 | keepdims      | keep_dims | Same function, different parameter names                                   |
| | parameter 5 |   name            | -         | Not involved |

### Code Example 1

> The two APIs are used to calculate the mean and variance of the specified axes of Tensor, with the same usage.

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), dtype='float32')
mean, variance = tf.nn.moments(x, axes=0, keepdims=True)
print(mean.numpy())
# [[[3. 4.]
#   [5. 6.]]]
print(variance.numpy())
# [[[4. 4.]
#   [4. 4.]]]

# MindSpore
import mindspore
from mindspore import Tensor
from mindspore import nn
import numpy as np

x = Tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), mindspore.float32)
net = nn.Moments(axis=0, keep_dims=True)
mean, variance = net(x)
print(mean)
# [[[3. 4.]
#   [5. 6.]]]
print(variance)
# [[[4. 4.]
#   [4. 4.]]]
```

### Code Example 2

> Two APIs for calculating the mean and variance of all values of Tensor. TensorFlow API requires manual specification of all axes, and MindSpore API specifies all axes by default.

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), dtype='float32')
mean, variance = tf.nn.moments(x, axes=[0, 1, 2])
print(mean.numpy())
# 4.5
print(variance.numpy())
# 5.25

# MindSpore
import mindspore
from mindspore import Tensor
from mindspore import nn
import numpy as np

x = Tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), mindspore.float32)
net = nn.Moments()
mean, variance = net(x)
print(mean)
# 4.5
print(variance)
# 5.25
```
