# 比较与tf.compat.v1.scatter_mul的差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/scatter_mul.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

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

更多内容详见[tf.compat.v1.scatter_mul](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/scatter_mul)。

## mindspore.ops.scatter_mul

```python
mindspore.ops.scatter_mul(
    input_x,
    indices,
    updates
) -> Tensor
```

更多内容详见[mindspore.ops.scatter_mul](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.scatter_mul.html)。

## 使用方式

TensorFlow：对Tensor进行原地散点式更新。

MindSpore：实现与TensorFlow一致的功能。TensorFlow可以使用use_locking参数控制在更新张量时是否使用锁定，加锁可以保证Tensor在多线程环境下可以被正确更新，默认为False。MindSpore默认实现不加锁的功能。

| 分类 | 子类 | TensorFlow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | ref | input_x | 功能一致，参数名不同 |
| | 参数2 | indices | indices | - |
| | 参数3 | updates | updates | - |
| | 参数4 | use_locking | - | MindSpore无此参数，默认实现不加锁的功能。 |
| | 参数5 | name | - | 不涉及 |

### 代码示例

> TensorFlow中use_locking为False时，两API实现功能一致。

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
