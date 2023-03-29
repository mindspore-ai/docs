# Function Differences with tf.compat.v1.scatter_mul

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/scatter_mul.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png"></a>

## tf.compat.v1.scatter_mul

```python
tf.compat.v1.scatter_mul(
    ref,
    indices,
    updates,
    use_locking=False,
    name=None
) -> Tensor
```

For more information, see [tf.compat.v1.scatter_mul](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/scatter_mul).

## mindspore.ops.scatter_mul

```python
mindspore.ops.scatter_mul(
    input_x,
    indices,
    updates
) -> Tensor
```

For more information, see [mindspore.ops.scatter_mul](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.scatter_mul.html).

## Usage

TensorFlow: In-place scatter update for Tensor.

MindSpore: Implement the same function as TensorFlow. TensorFlow can use the use_locking parameter to control whether locking is used when updating the tensor. Locking ensures that the Tensor can be updated correctly in a multi-threaded environment, and the default is False. MindSpore implements unlocked function by default.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | Parameter1 | ref | input_x | Same function, different parameter names |
| | Parameter2 | indices | indices | - |
| | Parameter3 | updates | updates | - |
| | Parameter4 | use_locking | - | MindSpore does not have this parameter and implements unlocked functionality by default. |
| | Parameter5 | name | - | Not involved |

### Code Example

> When use_locking is False in TensorFlow, the two APIs implement the same function.

```python
# TensorFlow
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

ref = tf.Variable(np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), dtype=tf.float32)
indices = tf.constant(np.array([0, 1]),  dtype=tf.int32)
updates = tf.constant(np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]), dtype=tf.float32)
op = tf.compat.v1.scatter_mul(ref, indices, updates, use_locking=False)

init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init)
    out = sess.run(op)
print(out)
# [[ 1.  6. 15.]
#  [ 2.  8. 18.]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor, Parameter
import mindspore.ops as ops

input_x = Parameter(Tensor(np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), mindspore.float32), name="x")
indices = Tensor(np.array([0, 1]), mindspore.int32)
updates = Tensor(np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]), mindspore.float32)
output = ops.scatter_mul(input_x, indices, updates)
print(output)
# [[ 1.  6. 15.]
#  [ 2.  8. 18.]]
```
