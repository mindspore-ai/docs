# Function Differences with tf.math.argmax

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/argmax.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.math.argmax

```text
tf.math.argmax(
    input,
    axis=None,
    output_type=tf.dtypes.int64,
    name=None,
) -> Tensor
```

For more information, see [tf.math.argmax](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/math/argmax).

## mindspore.ops.argmax

```text
mindspore.ops.argmax(x, axis=None, keepdims=False) -> Tensor
```

For more information, see [mindspore.ops.argmax](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.argmax.html).

## Differences

TensorFlow: Return the index of the maximum value of the Tensor along the given dimension. The return value type defaults to tf.int64, and the default is to return the index of the maximum value when axis is 0.

MindSpore: MindSpore API basically implements the same function as TensorFlow. The return value type is ms.int32 by default, and the default is to return the index of the maximum value when axis is -1.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Input | Single input | input | x |Both are input Tensor. Both do not support zero-dimensional tensor. TensorFlow supports Tensor type and Numpy.ndarray type input, while MindSpore only supports Tensor type input |
|Parameters | Parameter 1 | axis | axis |Same function, same parameter name, different default value |
| | Parameter 2 | output_type | - | Specify the output type, MindSpore does not have this parameter |
| | Parameter 3 | name | - | Not involved |
| | Parameter 4 | - | keepdims | PyTorch does not have this parameter. MindSpore parameter keepdims is set to True to keep the dimension for aggregation and set to 1. |

### Code Example 1

> Whne TensorFlow argmax operator does not explicitly give the axis parameter, the computation result is the index of the maximum value when axis is 0 by default, while MindSpore returns the index of the maximum value when axis is -1 by default. Therefore, in order to get the same result, the mindspore.ops.argmax operator axis is assigned to 0 before the calculation, and to ensure that the output types are the same, use [mindspore.ops.Cast](https://mindspore.cn/docs/en/Cast.html) operator to convert the result of MindSpore to mindspore.int64.

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x = np.arange(2*3*4).reshape(2, 3, 4).astype(np.float32)
tf_argmax = tf.math.argmax
tf_output = tf.math.argmax(tf.constant(x))
tf_out_np = tf_output.numpy()
print(tf_out_np)
# [[1 1 1 1]
#  [1 1 1 1]
#  [1 1 1 1]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor

x = np.arange(2*3*4).reshape(2,3,4).astype(np.float32)
axis = 0
ms_argmax = mindspore.ops.argmax
ms_output = ms_argmax(Tensor(x), axis)
ms_cast = mindspore.ops.Cast()
ms_output = ms_cast(ms_output, mindspore.int64)
ms_out_np = ms_output.asnumpy()
print(ms_out_np)
# [[1 1 1 1]
#  [1 1 1 1]
#  [1 1 1 1]]
```

### Code Example 2

> The way that TensorFlow and MindSpore parameters are passed in does not affect the function.

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x = np.arange(2*3*4).reshape(2, 3, 4).astype(np.float32)
tf_argmax = tf.math.argmax
axis = 2
tf_output = tf.math.argmax(tf.constant(x), axis)
tf_out_np = tf_output.numpy()
print(tf_out_np)
# [[3 3 3]
#  [3 3 3]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor

x = np.arange(2*3*4).reshape(2,3,4).astype(np.float32)
axis = 2
ms_argmax = mindspore.ops.argmax
ms_output = ms_argmax(Tensor(x), axis)
ms_cast = mindspore.ops.Cast()
ms_output = ms_cast(ms_output, mindspore.int64)
ms_out_np = ms_output.asnumpy()
print(ms_out_np)
# [[3 3 3]
#  [3 3 3]]
```

### Code Example 3

> The TensorFlow parameter output_type is used to specify the output data type, and the default is tf.int64. The default value of the MindSpore parameter output_type is mindspore.int32. To ensure that the two output types are the same, use the [mindspore.ops.Cast](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Cast.html) operator to convert the MindSpore into mindspore.int64. The TensorFlow parameter name is used to define the name of the executed operation and does not affect the result, while MindSpore does not have this parameter.

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x = np.arange(2*3*4).reshape(2, 3, 4).astype(np.float32)
tf_argmax = tf.math.argmax
axis = 1
tf_output = tf.math.argmax(tf.constant(x), axis, name="tf_output")
tf_out_np = tf_output.numpy()
print(tf_out_np)
# [[2 2 2 2]
#  [2 2 2 2]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor

x = np.arange(2*3*4).reshape(2,3,4).astype(np.float32)
axis = 1
ms_argmax = mindspore.ops.argmax
ms_output = ms_argmax(Tensor(x), axis)
ms_cast = mindspore.ops.Cast()
ms_output = ms_cast(ms_output, mindspore.int64)
ms_out_np = ms_output.asnumpy()
print(ms_out_np)
# [[2 2 2 2]
#  [2 2 2 2]]
```
