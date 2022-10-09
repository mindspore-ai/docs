# Function Differences with tf.gradients

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/grad.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

## mindspore.grad

```python
mindspore.grad(
  fn,
  grad_position=0,
  weights=None,
  hax_aux=False
)
```

For more information, see [mindspore.grad](https://mindspore.cn/docs/en/master/api_python/mindspore.grad.html).

## Differences

TensorFlow: Compute the gradient of `ys` with respect to `xs`, and return a list of the same length as `xs`.

MindSpore: Compute the first derivative. When `grad_position` is set to int or tuple of int, the corresponding input derivatives are computed. if `weights` is set, the network parameters derivatives will be computed. If `has_aux` is True,  only the first output of `fn` participates in the computation, in this case, the `fn` should has at least two outputs.

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
    def construct(self, x, y):
        gradient_function = ms.grad(self.net)
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
