# Function Differences with tf.gradients

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.10/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.10/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/GradOperation.md)

## tf.gradients

```python
tf.gradients(
    ys,
    xs,
    grad_ys=None,
    name='gradients',
    colocate_gradients_with_ops=False,
    gate_gradients=False,
    aggregation_method=None,
    stop_gradients=None,
    unconnected_gradients=tf.UnconnectedGradients.NONE
)
```

For more information, see [tf.gradients](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/gradients).

## mindspore.ops.GradOperation

```python
class mindspore.ops.GradOperation(
  get_all=False,
  get_by_list=False,
  sens_param=False
)
```

For more information, see [mindspore.ops.GradOperation](https://mindspore.cn/docs/en/r1.10/api_python/ops/mindspore.ops.GradOperation.html).

## Differences

TensorFlow: Compute the gradient of `ys` with respect to `xs`, and return a list of the same length as `xs`.

MindSpore: Compute the first derivative. When `get_all` is set to False, the first input derivative is computed. When `get_all` is set to True, all input derivatives are computed. When `get_by_list` is set to False, weight derivatives are not computed. When `get_by_list` is set to True, the weight derivative is computed. `sens_param` scales the output value of the network to change the final gradient.

## Code Example

```python
# In MindSpore:
import numpy as np
import mindspore.nn as nn
import mindspore as ms
from mindspore import ops

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()
        self.z = ms.Parameter(ms.Tensor(np.array([1.0], np.float32)), name='z')
    def construct(self, x, y):
        x = x * self.z
        out = self.matmul(x, y)
        return out

class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation()
    def construct(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y)

x = ms.Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=ms.float32)
y = ms.Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=ms.float32)
output = GradNetWrtX(Net())(x, y)
print(output)
# Out:
# [[1.4100001 1.5999999 6.6      ]
#  [1.4100001 1.5999999 6.6      ]]

# In TensorFlow:
import tensorflow as tf
w1 = tf.get_variable('w1', shape=[3])
w2 = tf.get_variable('w2', shape=[3])
w3 = tf.get_variable('w3', shape=[3])
w4 = tf.get_variable('w4', shape=[3])
z1 = w1 + w2+ w3
z2 = w3 + w4
grads = tf.gradients([z1, z2], [w1, w2, w3, w4], grad_ys=[tf.convert_to_tensor([2.,2.,3.]),
                                                          tf.convert_to_tensor([3.,2.,4.])])

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(grads))
# Out:
# [array([2., 2., 3.], dtype=float32),
#  array([2., 2., 3.], dtype=float32),
#  array([5., 4., 7.], dtype=float32),
#  array([3., 2., 4.], dtype=float32)]
```