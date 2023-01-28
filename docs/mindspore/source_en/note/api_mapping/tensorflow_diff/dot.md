# Function Differences with tf.keras.backend.dot

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/dot.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.keras.backend.dot

```text
tf.keras.backend.dot(x, y) -> Tensor
```

For more information, see [tf.keras.backend.dot](https://keras.io/zh/backend/#dot).

## mindspore.ops.dot

```text
mindspore.ops.dot(x1, x2) -> Tensor
```

For more information, see [mindspore.ops.dot](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.dot.html).

## Differences

TensorFlow: Compute the dot product between two Tensor or Variable.

MindSpore: When both input parameters are tensor, MindSpore API implements the same function as TensorFlow, and only the parameter names are different. Supported only by TensorFlow when either of the two input parameters is a variable.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| :-: | :-: | :-: | :-: |:-:|
|Parameters | Parameter 1 | x | x1 |Same function, different parameter names, and MindSpore parameters can only be Tensor type |
| | Parameter 2 | y | x2 |Same function, different parameter names, and MindSpore parameters can only be Tensor type |

### Code Example

> When both input parameters are of Tensor type, the function is the same and the usage is the same.

```python
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.ones([1, 3, 2])
xy = tf.keras.backend.dot(x, y)
print(xy.numpy())
# [[[3. 3.]]
#  [[3. 3.]]]

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

x1 = Tensor(np.ones(shape=[2, 3]), mindspore.float32)
x2 = Tensor(np.ones(shape=[1, 3, 2]), mindspore.float32)
out = mindspore.ops.dot(x1, x2)
print(out)
# [[[3. 3.]]
#  [[3. 3.]]]
```
