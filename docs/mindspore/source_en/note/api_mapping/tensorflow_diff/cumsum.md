# Function Differences with tf.math.cumsum

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/cumsum.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.math.cumsum

```text
tf.math.cumsum(x, axis=0, exclusive=False, reverse=False, name=None) -> Tensor
```

For more information, see [tf.math.cumsum](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/math/cumsum).

## mindspore.ops.cumsum

```text
mindspore.ops.cumsum(x, axis, dtype=None) -> Tensor
```

For more information, see [mindspore.ops.cumsum](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.cumsum.html).

## Differences

TensorFlow: Calculates the cumulative sum of the input Tensor on the specified axis.

MindSpore: MindSpore API basically implements the same function as TensorFlow, and there are differences in parameter settings.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | x | x |- |
| | Parameter 2 | axis | axis | MindSpore has no default value and can specify dimensions |
| | Parameter 3 | exclusive | - | MindSpore does not have this parameter |
| | Parameter 4 | reverse | - | MindSpore does not have this parameter |
| | Parameter 5 | name | - | Not involved |
| | Parameter 6 | - | dtype | Setting the output data type in MindSpore |

### Code Example 1

> The same input tensor, with axis -1, accumulates the innermost layer of the input tensor from left to right, and the two APIs achieve the same function.

```python
# TensorFlow
import tensorflow as tf
a = tf.constant([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]])
y = tf.cumsum(a, -1)
print(y.numpy())
# [[ 3  7 13 23]
#  [ 1  7 14 23]
#  [ 4  7 15 22]
#  [ 1  4 11 20]]

# MindSpore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np
x = Tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]]))
y = ops.cumsum(x, -1)
print(y)
# [[ 3  7 13 23]
#  [ 1  7 14 23]
#  [ 4  7 15 22]
#  [ 1  4 11 20]]
```
