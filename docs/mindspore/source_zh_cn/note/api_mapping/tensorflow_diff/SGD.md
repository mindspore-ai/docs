# 比较与tf.keras.optimizers.SGD的功能差异

## tf.keras.optimizers.SGD

```text
class tf.keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.0,
    nesterov=False,
    name='SGD',
    **kwargs
) -> Tensor
```

更多内容详见 [tf.keras.optimizers.SGD](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/keras/optimizers/SGD)。

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

更多内容详见 [mindspore.nn.SGD](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.SGD.html)。

## 差异对比

TensorFlow：实现的是梯度下降（带动量）优化器的功能。

MindSpore：MindSpore此API实现功能与TensorFlow基本一致。除了`learning_rate`设置的默认值不同外，MindSpore提供了参数分组`params`、动量的抑制因子`dampening`、权重衰减`weight_decay`、混合精度`loss_scale`等入参配置，TensorFlow无这些参数。而TensorFlow中的kwargs参数可以设置为`clipvalue`，`clipnorm`，梯度截断Clip可以将梯度约束在某一个区间之内，MindSpore无此功能。

| 分类 | 子类 |TensorFlow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | learning_rate | learning_rate |功能一致，参数默认值不同 |
| | 参数2 | momentum | momentum |- |
| | 参数3 | nesterov | nesterov |- |
| | 参数4 | name | - |不涉及 |
| | 参数5 | **kwargs | - | 不涉及 |
| | 参数6 | - | params |MindSpore提供参数分组功能，且支持为不同参数组设置不同配置值，通过入参params传入参数组字典实现，TensorFlow没有此入参配置 |
| | 参数7 | - | dampening |动量的抑制因子，TensorFlow无此参数 |
| | 参数8 | - | weight_decay |实现对需要优化的参数使用权重衰减的策略，以避免模型过拟合问题，TensorFlow无此参数 |
| | 参数9 | - | loss_scale |梯度缩放系数，TensorFlow无此参数 |
| | 参数10 | - | gradients  |反向输入，TensorFlow无此参数 |

### 代码示例

> 两API实现功能一致，用法相同。

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
t = {"x": [1.0], "y": [0.0]}
data = NumpySlicesDataset(t)
y0 = net(1.0)
model.train(1, data)
y1 = net(1.0)
print(y1)
# [0.9]
model.train(1, data)
y2 = net(1.0)
print(y2)
# [0.71999997]
```
