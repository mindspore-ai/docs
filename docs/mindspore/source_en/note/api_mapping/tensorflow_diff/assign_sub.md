# Function Differences with tf.compat.v1.assign_sub

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/assign_sub.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png"></a>

## tf.compat.v1.assign_sub

```text
tf.compat.v1.assign_sub(ref, value, use_locking=None, name=None) -> Tensor
```

For more information, see [tf.compat.v1.assign_sub](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/compat/v1/assign_sub).

## mindspore.ops.assign_sub

```text
mindspore.ops.assign_sub(variable, value)-> Tensor
```

For more information, see [mindspore.ops.assign_sub](https://www.mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.assign_sub.html).

## Differences

TensorFlow: Update the network parameters by subtracting a specific value from the network parameters, and return a Tensor with the same type as ref.

MindSpore: MindSpore API implements the same functions as TensorFlow, with some different parameter names.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | ref | variable        | Same function, different parameter names           |
|  | Parameter 2 | value       | value          | - |
|  | Parameter 3 | use_locking       | -         | In TensorFlow, whether to use locks in update operations. Default value: False. |
|  | Parameter 4 | name | -           | Not involved |

### Code Example 1

The outputs of MindSpore and TensorFlow are consistent.

```python
# TensorFlow
import tensorflow as tf
import numpy as np

variable = tf.Variable(np.array([[2.4, 1], [0.1, 6]]), dtype=tf.float32)
value = tf.constant(np.array([[-2, 3], [3.6, 1]]), dtype=tf.float32)
out = tf.compat.v1.assign_sub(variable, value)
print(out.numpy())
# [[ 4.4 -2. ]
#  [-3.5  5. ]]

# MindSpore
import mindspore
import numpy as np
from mindspore.ops import function as ops
from mindspore import Tensor

variable = Tensor(np.array([[2.4, 1], [0.1, 6]]), mindspore.float32)
value = Tensor(np.array([[-2, 3], [3.6, 1]]), mindspore.float32)
out = ops.assign_sub(variable, value)
print(out)
# [[ 4.4 -2. ]
#  [-3.5  5. ]]
```
