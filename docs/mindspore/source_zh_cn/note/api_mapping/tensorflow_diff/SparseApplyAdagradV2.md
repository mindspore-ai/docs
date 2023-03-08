# 比较与tf.raw_ops.SparseApplyAdagradV2的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/SparseApplyAdagradV2.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

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

更多内容详见[tf.raw_ops.SparseApplyAdagradV2](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/raw_ops/SparseApplyAdagradV2)。

## mindspore.ops.SparseApplyAdagradV2

```text
mindspore.ops.SparseApplyAdagradV2(lr, epsilon, update_slots=True, use_locking=False) -> (Tensor, Tensor)
```

更多内容详见[mindspore.ops.SparseApplyAdagradV2](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.SparseApplyAdagradV2.html)。

## 差异对比

TensorFlow：根据Adagrad算法更新相关参数，返回一个和var类型一样的Tensor。

MindSpore：MindSpore此API实现功能、返回的第一个参数和TensorFlow返回一致，第二个返回参数shape和数据类型与accum相同。

| 分类 | 子类 |TensorFlow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | var | var         | -   |
|  | 参数2 | accum       | accum          | - |
|  | 参数3 | lr       | lr         | - |
|  | 参数4 | epsilon       | epsilon          | - |
|  | 参数5 | grad       | grad         | - |
|  | 参数6 | indices       | indices          | - |
| | 参数7 | use_locking | use_locking      | |
|  | 参数8 | update_slots       | update_slots         | - |
| | 参数9 | name | -           | 不涉及 |

### 代码示例1

MindSpore和TensorFlow输出结果一致。

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
