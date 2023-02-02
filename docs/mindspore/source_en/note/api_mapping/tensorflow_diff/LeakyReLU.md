# Function Differences with tf.nn.leaky_relu

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/LeakyReLU.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.nn.leaky_relu

```text
tf.nn.leaky_relu(features, alpha=0.2, name=None) -> Tensor
```

For more information, see [tf.nn.leaky_relu](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/leaky_relu).

## mindspore.nn.LeakyReLU

```text
class mindspore.nn.LeakyReLU(alpha=0.2)(x) -> Tensor
```

For more information, see [mindspore.nn.LeakyReLU](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.LeakyReLU.html).

## Differences

TensorFlow: Apply the Leaky ReLU activation function, where the parameter `alpha` is used to control the slope of the activation function.

MindSpore: MindSpore API basically implements the same function as TensorFlow.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | features | x | Same function, different parameter names |
| | Parameter 2 | alpha | alpha | - |
| | Parameter 3 | name | - | Not involved |

### Code Example

> The two APIs achieve the same function and have the same usage.

```python
# TensorFlow
import tensorflow as tf

features = tf.constant([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]], dtype=tf.float32)
output = tf.nn.leaky_relu(features).numpy()
print(output)
# [[-0.2  4.  -1.6]
#  [ 2.  -1.   9. ]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn

x = Tensor([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]).astype('float32')
m = nn.LeakyReLU()
output = m(x)
print(output)
# [[-0.2  4.  -1.6]
#  [ 2.  -1.   9. ]]
```
