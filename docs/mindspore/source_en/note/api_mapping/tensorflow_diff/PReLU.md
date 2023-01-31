# Function Differences with tf.keras.layers.PReLU

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/PReLU.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.keras.layers.PReLU

```text
tf.keras.layers.PReLU(
  alpha_initializer='zeros',
  alpha_regularizer=None,
  alpha_constraint=None,
  shared_axes=None
)(x) -> Tensor
```

For more information, see [tf.keras.layers.PReLU](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/keras/layers/PReLU).

## mindspore.nn.PReLU

```text
class mindspore.nn.PReLU(channel=1, w=0.25)(x) -> Tensor
```

For more information, see [mindspore.nn.PReLU](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.PReLU.html).

## Differences

TensorFlow: PReLU activation function.

MindSpore: MindSpore API basically implements the same function as TensorFlow, but the parameter setting is different.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | parameter 1 | alpha_initializer | w | Initialization function of weights, same function of parameters, different default values, different parameter names |
| | parameter 2 | alpha_regularizer | - | Regularizer of weights. MindSpore does not have this parameter. |
| | parameter 3 | alpha_constraint | - | Constraints of Weights. MindSpore does not have this parameter. |
| | parameter 4 | shared_axes | - | Shared axes of learnable parameters of the activation function. MindSpore does not have this parameter. |
| | parameter 5  | -                 | channel   | TensorFlow does not have this parameter.      |
| Input | Single input | x | x | - |

### Code Example 1

> TensorFlow alpha_initializer parameter is functionally identical to MindSpore parameter, with different default values and different parameter names. Default alpha of TensorFlow is 0.0, so using MindSpore, you only need to set w to 0.0 to achieve the same function.

```python
# TensorFlow
import tensorflow as tf
from keras.layers import PReLU
import numpy as np

x = tf.constant([[-1.0, 2.2], [3.3, -4.0]], dtype=tf.float32)
m = PReLU()
out = m(x)
print(out.numpy())
# [[0.  2.2]
#  [3.3 0. ]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

x = Tensor(np.array([[-1.0, 2.2], [3.3, -4.0]]), mindspore.float32)
prelu = nn.PReLU(w=0.0)
output = prelu(x)
print(output)
# [[0.  2.2]
#  [3.3 0. ]]
```

### Code Example 2

> TensorFlow alpha_initializer parameter can change the alpha value through the initialization function, and MindSpore simply sets w to the corresponding value to achieve the same function.

```python
# TensorFlow
import tensorflow as tf
from keras.layers import PReLU
import numpy as np
x = tf.constant([[-1.0, 2.2], [3.3, -4.0]], dtype=tf.float32)
m = PReLU(alpha_initializer=tf.constant_initializer(0.5))
out = m(x)
print(out.numpy())
# [[-0.5  2.2]
#  [ 3.3 -2. ]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np
x = Tensor(np.array([[-1.0, 2.2], [3.3, -4.0]]), mindspore.float32)
prelu = nn.PReLU(w=0.5)
output = prelu(x)
print(output)
# [[-0.5  2.2]
#  [ 3.3 -2. ]]
```
