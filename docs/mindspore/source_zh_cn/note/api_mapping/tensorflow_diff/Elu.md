# 比较与tf.nn.elu的功能差异

## tf.nn.elu

``` text
tf.nn.elu(features) -> Tensor
```

更多内容详见 [tf.nn.elu](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/elu)。

## mindspore.ops.Elu

``` text
class mindspore.ops.Elu(x, alpha=1.0)(x)
```

更多内容详见 [mindspore.ops.Elu](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Elu.html)。

## 差异对比

TensorFlow：计算输入features的指数线性值，返回结果为
$\left\{\begin{array}{ll}
e^{\text {feature }}-1, & \text { feature }<0 \\
\text { feature } & , \text { feature } \geq 0
\end{array}\right.$

MindSpore：MindSpore此API实现功能与TensorFlow基本一致， 不过支持数据类型有所差异。

| 分类 | 子类 |TensorFlow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | features | x |功能一致， 参数名不同 |
| | 参数2 | - | alpha | Mindspore目前只支持alpha等于1.0，与TensorFlow接口一致 |

### 代码示例1

> 两API实现相同功能，MindSpore使用时需要先实例化方法，输出tensor的shape和数据类型与输入相同。

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x_ = np.array([[np.arange(-6,0).reshape(2, 3),np.arange(0,6).reshape(2, 3)]])
x = tf.convert_to_tensor(x_, dtype=tf.float32)
output = tf.nn.elu(x).numpy()
print(output)
# [[[[-0.9975212  -0.99326205 -0.9816844 ]
# [-0.95021296 -0.86466473 -0.6321205 ]]
#
#  [[ 0.          1.          2.        ]
#   [ 3.          4.          5.        ]]]]

# MindSpore
import mindspore as ms
from mindspore import ops, nn
import numpy as np

x_ = np.array([[np.arange(-6,0).reshape(2, 3),np.arange(0,6).reshape(2, 3)]])
x = ms.Tensor(x_, ms.float32)
elu = ops.Elu()
output = elu(x)
print(output)
##[[[[-0.9975212  -0.99326205 -0.9816844 ]
#   [-0.95021296 -0.86466473 -0.6321205 ]]
#
#  [[ 0.          1.          2.        ]
#   [ 3.          4.          5.        ]]]]

```