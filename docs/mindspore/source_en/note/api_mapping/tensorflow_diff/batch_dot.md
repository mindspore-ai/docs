# Function Differences with tf.keras.backend.batch_dot

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/batch_dot.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.keras.backend.batch_dot

```text
tf.keras.backend.batch_dot(x, y, axes=None)
```

For more information, see [tf.keras.backend.batch_dot](https://keras.io/zh/backend/#batch_dot).

## mindspore.ops.batch_dot

```text
mindspore.ops.batch_dot(x1, x2, axes=None)
```

For more information, see [mindspore.ops.batch_dot](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.batch_dot.html).

## Differences

TensorFlow: When the input x and y are batch data, batch_dot returns the dot product of x and y.

MindSpore: MindSpore API implements the same function as Keras, and only the parameter names are different.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1 | x          | x1        | Same function, different parameter names |
|      | Parameter 2 | y          | x2        | Same function, different parameter names |
|      | Parameter 3 | axes       | axes      | -             |

### Code Example 1

The two APIs without axes parameter achieve the same function and the same usage.

```python
# TensorFlow
import keras.backend as K
import tensorflow as tf
import numpy as np

x = K.variable(np.random.randint(10,size=(10,12,4,5)), dtype=tf.float32)
y = K.variable(np.random.randint(10,size=(10,12,5,8)), dtype=tf.float32)
output = K.batch_dot(x, y)
print(output.shape)
# (10, 12, 4, 12, 8)

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x1 = Tensor(np.random.randint(10,size=(10,12,4,5)), mindspore.float32)
x2 = Tensor(np.random.randint(10,size=(10,12,5,8)), mindspore.float32)
output = ops.batch_dot(x1, x2)
print(output.shape)
# (10, 12, 4, 12, 8)
```

### Code Example 2

The two APIs with axes parameter achieve the same function and the same usage.

```python
# TensorFlow
import keras.backend as K
import tensorflow as tf
import numpy as np

x = K.variable(np.ones(shape=[2, 2]), dtype=tf.float32)
y = K.variable(np.ones(shape=[2, 3, 2]), dtype=tf.float32)
axes = (1, 2)
output = K.batch_dot(x, y, axes)
print(output.shape)
# (2, 3)

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x1 = Tensor(np.ones(shape=[2, 2]), mindspore.float32)
x2 = Tensor(np.ones(shape=[2, 3, 2]), mindspore.float32)
axes = (1, 2)
output = ops.batch_dot(x1, x2, axes)
print(output.shape)
# (2, 3)
```
