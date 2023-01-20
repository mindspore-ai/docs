# Function Differences with tf.nn.bias_add

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/bias_add.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.nn.bias_add

```text
class tf.nn.bias_add(value, bias, data_format=None, name=None)
```

For more information, see [tf.nn.bias_add](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/bias_add).

## mindspore.ops.bias_add

```text
mindspore.ops.bias_add(input_x, bias)
```

For more information, see [mindspore.ops.bias_add](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.bias_add.html).

## Differences

TensorFlow: Return the sum of the tensor of input value and bias, where bias is restricted to a 1D tensor and value supports various numbers of dimensions, and bias is broadcasted to be consistent with the shape of input value before the two are summed.

MindSpore: MindSpore API basically implements the same function as TensorFlow. However, MindSpore input input_x only supports 2-5 dimensional shapes.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1 | value      | input_x   | Same function, different parameter names                  |
|      | Parameter 2 | bias       | bias      | Same function                              |
|      | Parameter 3 | data_format | -         | The data format of the input data. MindSpore does not have this parameter |
|      | Parameter 4 | name       | -         | Not involved   |

### Code Example 1

The two APIs achieve the same function and have the same usage.

```python
# TensorFlow
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
value = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
bias = tf.constant([-2, -1], dtype=tf.float32)
result = tf.nn.bias_add(value, bias)
ss = tf.compat.v1.Session()
output = ss.run(result)
print(output)
# [[-1.  1.]
#  [ 1.  3.]
#  [ 3.  5.]]

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

input_x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]), mindspore.float32)
bias = Tensor(np.array([-2 , -1]), mindspore.float32)
output = ops.bias_add(input_x, bias)
print(output)
# [[-1.  1.]
#  [ 1.  3.]
#  [ 3.  5.]]
```
