# Function Differences with tf.expand_dims

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/expand_dims.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.expand_dims

```text
tf.expand_dims(x, axis, name=None) -> Tensor
```

For more information, see [tf.expand_dims](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/expand_dims).

## mindspore.ops.expand_dims

```text
mindspore.ops.expand_dims(input_x, axis) -> Tensor
```

For more information, see [mindspore.ops.expand_dims](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.expand_dims.html).

## Differences

TensorFlow: Add an extra dimension to the input x on the given axis.

MindSpore: MindSpore API implements the same function as TensorFlow, and only the parameter names are different.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1 | x          | input_x   | Same function, different parameter names |
|      | Parameter 2 | axis       | axis      | - |
|      | Parameter 3 | name       | -      | Not involved |

### Code Example 1

> The two APIs achieve the same function and have the same usage.

```python
# TensorFlow
import numpy as np
import tensorflow as tf

x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)
axis = 1
out = tf.expand_dims (x, axis).numpy()
print(out)
# [[[ 1.  2.  3.  4.]]
#  [[ 5.  6.  7.  8.]]
#  [[ 9. 10. 11. 12.]]]

# MindSpore
import mindspore
import numpy as np
import mindspore.ops as ops
from mindspore import Tensor

input_params = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]), mindspore.float32)
axis = 1
output = ops.expand_dims(input_params,  axis)
print(output)
# [[[ 1.  2.  3.  4.]]
#  [[ 5.  6.  7.  8.]]
#  [[ 9. 10. 11. 12.]]]

```

### Code Example 2

> The two APIs achieve the same function and have the same usage.

```python
# TensorFlow
import numpy as np
import tensorflow as tf

x = np.array([[1,1,1]], dtype=np.float32)
axis = 2
out = tf.expand_dims (x, axis).numpy()
print(out)
# [[[1.]
#   [1.]
#   [1.]]]


# MindSpore
import mindspore
import numpy as np
import mindspore.ops as ops
from mindspore import Tensor

input_params = Tensor(np.array([[1,1,1]]), mindspore.float32)
axis = 2
output = ops.expand_dims(input_params,  axis)
print(output)
# [[[1.]
#   [1.]
#   [1.]]]
```
