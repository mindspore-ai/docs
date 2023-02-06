# 比较与tf.keras.optimizers.Ftrl的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/FTRL.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## tf.keras.optimizers.Ftrl

```python
tf.keras.optimizers.Ftrl(
    learning_rate=0.001,
    learning_rate_power=-0.5,
    initial_accumulator_value=0.1,
    l1_regularization_strength=0.0,
    l2_regularization_strength=0.0,
    name='Ftrl',
    l2_shrinkage_regularization_strength=0.0,
    beta=0.0,
    **kwargs) -> Tensor
```

更多内容详见[tf.keras.optimizers.Ftrl](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/keras/optimizers/Ftrl)。

## mindspore.nn.FTRL

```python
class mindspore.nn.FTRL(
    params,
    initial_accum=0.1,
    learning_rate=0.001,
    lr_power=-0.5,
    l1=0.0,
    l2=0.0,
    use_locking=False,
    loss_scale=1.0,
    weight_decay=0.0)(grads) -> Tensor
```

更多内容详见[mindspore.nn.FTRL](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.FTRL.html)。

## 差异对比

TensorFlow：一种在线凸优化算法，适合具有大而稀疏特征特征空间的浅层模型。

MindSpore：MindSpore此API实现功能与TensorFlow基本一致。

| 分类 | 子类  | TensorFlow                 | MindSpore | 差异 |
| ---- | ----- | -------------------------- | --------- | ---- |
| 参数 | 参数1 | learning_rate | learning_rate | - |
|      | 参数2 | learning_rate_power        | lr_power | 功能一致，参数名不同 |
|      | 参数3 | initial_accumulator_value  | initial_accum | 功能一致，参数名不同 |
|      | 参数4 | l1_regularization_strength | l1 | 功能一致，参数名不同 |
|      | 参数5 | l2_regularization_strength | l2 | 功能一致，参数名不同 |
|      | 参数6 | name | - | 不涉及 |
|      | 参数7 | l2_shrinkage_regularization_strength | weight_decay | 功能一致，参数名不同 |
|      | 参数8 | beta |      -     | 一个浮点值，默认值为0.0。MindSpore无此参数                 |
|      | 参数9 | **kwargs | - | 不涉及 |
|      | 参数10 | - | params | 由Parameter类组成的列表或由字典组成的列表，TensorFlow中无此参数 |
|      | 参数11 | - | use_locking | 如果为True，则更新操作使用锁保护，默认值为False。TensorFlow中无此参数 |
|      | 参数12 | - | loss_scale | 梯度缩放系数，默认值：1.0，TensorFlow中无此参数 |
|      | 参数13 | - | grads | 优化器中 `params` 的梯度，TensorFlow中无此参数 |

### 代码示例1

> learning_rate均设置为0.1，两API功能一致，用法相同。

```python
# TensorFlow
import tensorflow as tf

opt = tf.keras.optimizers.Ftrl(learning_rate=0.1)
var = tf.Variable(1.0)
val0 = var.value()
loss = lambda: (var ** 2) / 2.0
step_count = opt.minimize(loss, [var]).numpy()
val1 = var.value()
print([val1.numpy()])
# [0.6031424]
step_count = opt.minimize(loss, [var]).numpy()
val2 = var.value()
print([val2.numpy()])
# [0.5532904]

# MindSpore
import numpy as np
import mindspore.nn as nn
import mindspore as ms
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
optim = nn.FTRL(params=net.trainable_params(), learning_rate=0.1)
model = ms.Model(net, loss_fn=loss, optimizer=optim)
data_x = np.array([1.0], dtype=np.float32)
data_y = np.array([0.0], dtype=np.float32)
data = NumpySlicesDataset((data_x, data_y), ["x", "y"])
input_x = ms.Tensor(np.array([1.0], np.float32))
y0 = net(input_x)
model.train(1, data)
y1 = net(input_x)
print(y1)
# [0.6031424]
model.train(1, data)
y2 = net(input_x)
print(y2)
# [0.5532904]
```

### 代码示例2

> learning_rate均设置为0.01，两API功能一致，用法相同。

```python
# TensorFlow
import tensorflow as tf

opt = tf.keras.optimizers.Ftrl(learning_rate=0.01)
var = tf.Variable(1.0)
val0 = var.value()
loss = lambda: (var ** 2) / 2.0
step_count = opt.minimize(loss, [var]).numpy()
val1 = var.value()
print([val1.numpy()])
# [0.688954]
step_count = opt.minimize(loss, [var]).numpy()
val2 = var.value()
print([val2.numpy()])
# [0.6834637]

# MindSpore
import numpy as np
import mindspore.nn as nn
import mindspore as ms
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
optim = nn.FTRL(params=net.trainable_params(), learning_rate=0.01)
model = ms.Model(net, loss_fn=loss, optimizer=optim)
data_x = np.array([1.0], dtype=np.float32)
data_y = np.array([0.0], dtype=np.float32)
data = NumpySlicesDataset((data_x, data_y), ["x", "y"])
input_x = ms.Tensor(np.array([1.0], np.float32))
y0 = net(input_x)
model.train(1, data)
y1 = net(input_x)
print(y1)
# [0.688954]
model.train(1, data)
y2 = net(input_x)
print(y2)
# [0.6834637]
```
