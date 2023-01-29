# Function Differences with tf.nn.elu

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/elu.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.nn.elu

```text
tf.nn.elu(features, name=None) -> Tensor
```

For more information, see [tf.nn.elu](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/elu).

## mindspore.ops.elu

```text
mindspore.ops.elu(input_x, alpha=1.0) -> Tensor
```

For more information, see [mindspore.ops.elu](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.elu.html).

## Differences

TensorFlow: Compute the exponential linear value of the input features and return the result as
$\left\{\begin{array}{ll}
e^{\text {feature }}-1, & \text { feature }<0 \\
\text { feature } & , \text { feature } \geq 0
\end{array}\right.$

MindSpore: MindSpore API basically implements the same function as TensorFlow, but the supported data types are different.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | features | input_x |Same function, different parameter names |
| | Parameter 2 | name |  | Not involved |
| | Parameter 3 | - | alpha | MindSpore currently only supports alpha equal to 1.0, consistent with the TensorFlow interface |

### Code Example 1

> Both APIs implement the same function, and the output tensor has the same shape and data type as the input.

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x_ = np.array([[np.arange(-6,0).reshape(2, 3),np.arange(0,6).reshape(2, 3)]])
x = tf.convert_to_tensor(x_, dtype=tf.float32)
output = tf.nn.elu(x).numpy()
print(output)
# [[[[-0.9975212  -0.99326205 -0.9816844 ]
#    [-0.95021296 -0.86466473 -0.6321205 ]]
#
#   [[ 0.          1.          2.        ]
#   [ 3.          4.          5.        ]]]]

# MindSpore
import mindspore as ms
from mindspore import ops, nn
import numpy as np

x_ = np.array([[np.arange(-6,0).reshape(2, 3),np.arange(0,6).reshape(2, 3)]])
x = ms.Tensor(x_, ms.float32)
output = ops.elu(x)
print(output)
# [[[[-0.9975212  -0.99326205 -0.9816844 ]
#   [-0.95021296 -0.86466473 -0.6321205 ]]
#
#  [[ 0.          1.          2.        ]
#   [ 3.          4.          5.        ]]]]
```
