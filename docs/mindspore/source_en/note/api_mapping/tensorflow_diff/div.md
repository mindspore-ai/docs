# Function Differences with tf.math.divide

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/div.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.math.divide

```text
tf.math.divide(x, y, name=None) -> Tensor
```

For more information, see [tf.math.divide](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/math/divide).

## mindspore.ops.div

```text
mindspore.ops.div(input, other, rounding_mode=None) -> Tensor
```

For more information, see [mindspore.ops.div](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.div.html).

## Differences

TensorFlow: The quotient is obtained by dividing two Tensors element-wise.

MindSpore: When the parameter rounding_mode of MindSpore API is None by default, MindSpore implements the same function as TensorFlow.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | x          | input  | Same function, different parameter names               |
| | Parameter 2 | y          | other    | Same function, different parameter names          |
| | Parameter 3 | -          |  rounding_mode | This parameter is not available in TensorFlow. MindSpore is an optional parameter that determines the rounding type of the result, and the default value is None |
| | Parameter 4 | name         |  - | Not involved |

### Code Example

> When the parameter rounding_mode of MindSpore is not specified, the two APIs achieve the same function and have the same usage.

```python
# TensorFlow
import tensorflow as tf
import numpy

x = tf.constant([[2, 4, 6, 8], [1, 2, 3, 4]])
y = tf.constant([5, 8, 8, 16])
out = tf.math.divide(x, y).numpy()
print(out)
# [[0.4   0.5   0.75  0.5  ]
#  [0.2   0.25  0.375 0.25 ]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

x_ = np.array([[2, 4, 6, 8], [1, 2, 3, 4]])
y_ = np.array([5, 8, 8, 16])
x = Tensor(x_, mindspore.float64)
y = Tensor(y_, mindspore.float64)
output = ops.div(x, y)
print(output)
# [[0.4   0.5   0.75  0.5  ]
#  [0.2   0.25  0.375 0.25 ]]
```