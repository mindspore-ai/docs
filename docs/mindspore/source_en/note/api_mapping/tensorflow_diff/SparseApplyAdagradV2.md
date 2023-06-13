# Function Differences with tf.raw_ops.SparseApplyAdagradV2

<a href="https://gitee.com/mindspore/docs/blob/r1.11/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/SparseApplyAdagradV2.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.11/resource/_static/logo_source_en.png"></a>

## tf.raw_ops.SparseApplyAdagradV2

```text
tf.raw_ops.SparseApplyAdagradV2(
    var,
    accum,
    lr,
    epsilon,
    grad,
    indices,
    use_locking=False,
    update_slots=True,
    name=None
)  -> Tensor
```

For more information, see [tf.raw_ops.SparseApplyAdagradV2](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/raw_ops/SparseApplyAdagradV2).

## mindspore.ops.SparseApplyAdagradV2

```text
mindspore.ops.SparseApplyAdagradV2(lr, epsilon, update_slots=True, use_locking=False)(var, accum, grad, indices) -> (Tensor, Tensor)
```

For more information, see [mindspore.ops.SparseApplyAdagradV2](https://www.mindspore.cn/docs/en/r1.11/api_python/ops/mindspore.ops.SparseApplyAdagradV2.html).

## Differences

TensorFlow: Update the relevant parameters according to the Adagrad algorithm and return a Tensor with the same type as var.

MindSpore: MindSpore API implements the same function as TensorFlow, and the first parameter returned by MindSpore is the same as that of TensorFlow. The second parameter returned by MindSpore shape has the same data type as accum.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | var | var         | -   |
|  | Parameter 2 | accum       | accum          | - |
|  | Parameter 3 | lr       | lr         | - |
|  | Parameter 4 | epsilon       | epsilon          | - |
|  | Parameter 5 | grad       | grad         | - |
|  | Parameter 6 | indices       | indices          | - |
| | Parameter 7 | use_locking | use_locking      | |
|  | Parameter 8 | update_slots       | update_slots         | - |
| | Parameter 9 | name | -           | Not involved |

### Code Example 1

The outputs of MindSpore and TensorFlow are consistent.

```python
# TensorFlow
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

var = tf.Variable(np.array([[1,1,1],[1,1,1],[1,1,1]]).astype(np.float32))
accum = tf.Variable(np.array([[2,2,1],[1,1,2],[1,1,2]]).astype(np.float32))
grad = tf.constant(np.array([[0.01,0.02,0.01],[0.01,0.02,0.02]]).astype(np.float32))
indices = tf.constant(np.array([1,1]).astype(np.int32))
lr = tf.constant(0.1)
epsilon = tf.constant(0.001)
op = tf.raw_ops.SparseApplyAdagradV2(var=var, accum=accum, grad=grad, indices=indices, lr=lr, epsilon=epsilon)
init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init)
    out = sess.run(op)
print(out)
# [[1. 1. 1.]
#  [0.99980021 0.99960052 0.99788034]
#  [1. 1. 1.]]

# MindSpore
import mindspore
import numpy as np
from mindspore.ops import operations as ops
from mindspore import Tensor
from mindspore import Parameter

var = Parameter(Tensor(np.array([[1,1,1],[1,1,1],[1,1,1]]).astype(np.float32)), name="var")
accum = Parameter(Tensor(np.array([[2,2,1],[1,1,2],[1,1,2]]).astype(np.float32)), name="accum")
grad = Tensor(np.array([[0.01,0.02,0.01],[0.01,0.02,0.02]]).astype(np.float32))
indices = Tensor(np.array([1,1]).astype(np.int32))
sparse_apply_adagrad_v2 = ops.SparseApplyAdagradV2(lr=0.01, epsilon=0.001)
out_var,out_accum = sparse_apply_adagrad_v2(var, accum, grad, indices)
print(out_var)
# [[1. 1. 1.]
#  [0.9998002 0.9996005 0.99978805]
#  [1. 1. 1.]]
print(out_accum)
# [[2. 2. 1.]
#  [1.0002 1.0007999 2.0005]
#  [1. 1. 2.]]

```
