# 比较与tf.keras.optimizers.SGD的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/SGD.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.keras.optimizers.SGD

```text
tf.keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.0,
    nesterov=False,
    name='SGD',
    **kwargs
) -> Tensor
```

更多内容详见[tf.keras.optimizers.SGD](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/keras/optimizers/SGD)。

## mindspore.nn.SGD

```text
class mindspore.nn.SGD(
    params,
    learning_rate=0.1,
    momentum=0.0,
    dampening=0.0,
    weight_decay=0.0,
    nesterov=False,
    loss_scale=1.0
)(gradients) -> Tensor
```

更多内容详见[mindspore.nn.SGD](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.SGD.html)。

## 差异对比

TensorFlow：实现的是随机梯度下降（带动量）的优化器功能。

MindSpore：MindSpore此API实现功能与TensorFlow基本一致。

| 分类 | 子类 |TensorFlow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | learning_rate | learning_rate |功能一致，默认值不同 |
| | 参数2 | momentum | momentum |- |
| | 参数3 | nesterov | nesterov |- |
| | 参数4 | name | - |不涉及 |
| | 参数5 | **kwargs | - | 不涉及 |
| | 参数6 | - | params |由Parameter类组成的列表或由字典组成的列表，TensorFlow中无此参数 |
| | 参数7 | - | dampening |浮点动量阻尼值，默认值：0.0，TensorFlow中无此参数 |
| | 参数8 | - | weight_decay |权重衰减（L2 penalty），默认值：0.0，TensorFlow中无此参数 |
| | 参数9 | - | loss_scale |梯度缩放系数，默认值：1.0，TensorFlow中无此参数 |
| | 参数10 | - | gradients  |优化器中`params` 的梯度，TensorFlow中无此参数 |

### 代码示例

> 两API实现功能一致。

```python
# TensorFlow
import tensorflow as tf

opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
var = tf.Variable(1.0)
val0 = var.value()
loss = lambda: (var ** 2)/2.0
step_count1 = opt.minimize(loss, [var]).numpy()
val1 = var.value()
print([val1.numpy()])
# [0.9]
step_count2 = opt.minimize(loss, [var]).numpy()
val2 = var.value()
print([val2.numpy()])
# [0.71999997]

# MindSpore
import mindspore.nn as nn
import mindspore as ms
import numpy as np
from mindspore.dataset import NumpySlicesDataset

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.w = ms.Parameter(ms.Tensor(np.array([1.0], np.float32)), name='w')

    def construct(self, x):
        f = self.w * x
        return f

class MyLoss(nn.LossBase):
    def __init__(self, reduction='none'):
        super(MyLoss, self).__init__()

    def construct(self, y, y_pred):
        return (y - y_pred) ** 2 / 2.0

net = Net()
loss = MyLoss()
optim = nn.SGD(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
model = ms.Model(net, loss_fn=loss, optimizer=optim)
data_x = np.array([1.0], dtype=np.float32)
data_y = np.array([0.0], dtype=np.float32)
data = NumpySlicesDataset((data_x, data_y), ["x", "y"])
input_x = ms.Tensor(np.array([1.0], np.float32))
y0 = net(input_x)
model.train(1, data)
y1 = net(input_x)
print(y1)
# [0.9]
model.train(1, data)
y2 = net(input_x)
print(y2)
# [0.71999997]
```
