# 比较与tf.gradients的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/GradOperation.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

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

更多内容详见[tf.gradients](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/gradients)。

## mindspore.ops.GradOperation

```python
class mindspore.ops.GradOperation(
  get_all=False,
  get_by_list=False,
  sens_param=False
)
```

更多内容详见[mindspore.ops.GradOperation](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.GradOperation.html)。

## 使用方式

TensorFlow：计算`ys`关于`xs`的梯度，返回一个与`xs`长度相同的列表。

MindSpore：计算梯度，其中`get_all`为False时，只会对第一个输入求导，为True时，会对所有输入求导；`get_by_list`为False时，不会对权重求导，为True时，会对权重求导；`sens_param`对网络的输出值做缩放以改变最终梯度。

## 代码示例

```python
# In MindSpore：
import numpy as np
import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore import ops, Tensor, Parameter

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()
        self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')
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

x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
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
