# Function Differences with tf.nn.relu

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/ReLU.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.nn.relu

```text
tf.nn.relu(features, name=None) -> Tensor
```

For more information, see [tf.nn.relu](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/relu).

## mindspore.nn.ReLU

```text
class mindspore.nn.ReLU()(x) -> Tensor
```

For more information, see [mindspore.nn.ReLU](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.ReLU.html).

## Differences

TensorFlow: PReLU activation function.

MindSpore: MindSpore API implements the same function as TensorFlow, but the parameter setting is different, and the operator needs to be instantiated first.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | parameter 1 | features | x | Same function, different parameter names |
| | parameter 2 | name | - | Not involved |

### Code Example

> The two APIs implement the same function, but the TensorFlow operator is functional and can accept input directly. The operator in MindSpore needs to be instantiated first.

```python
# TensorFlow
import tensorflow as tf

x = tf.constant([[-1.0, 2.2], [3.3, -4.0]], dtype=tf.float16)
out = tf.nn.relu(x).numpy()
print(out)
# [[0.  2.2]
#  [3.3 0. ]]

# MindSpore
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np

x = Tensor(np.array([[-1.0, 2.2], [3.3, -4.0]]), mindspore.float16)
relu = nn.ReLU()
output = relu(x)
print(output)
# [[0.  2.2]
#  [3.3 0. ]]
```
