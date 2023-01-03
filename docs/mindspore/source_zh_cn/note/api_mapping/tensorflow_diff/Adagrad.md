# 比较与tf.keras.optimizers.Adagrad的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/Adagrad.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.keras.optimizers.Adagrad

```text
tf.keras.optimizers.Adagrad(
    learning_rate=0.001,
    initial_accumulator_value=0.1,
    epsilon=1e-07,
    name='Adagrad',
    **kwargs
) -> Tensor
```

更多内容详见[tf.keras.optimizers.Adagrad](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/keras/optimizers/Adagrad)。

## mindspore.nn.Adagrad

```text
class mindspore.nn.Adagrad(
    params,
    accum=0.1,
    learning_rate=0.001,
    update_slots=True,
    loss_scale=1.0,
    weight_decay=0.0
)(grads) -> Tensor
```

更多内容详见[mindspore.nn.Adagrad](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.Adagrad.html)。

## 差异对比

TensorFlow：Adagrad是一个具有特定参数学习率的优化器，用来实现Adagrad算法，它根据训练期间参数更新的频率进行调整。参数接收的更新越多，更新越小。

MindSpore：MindSpore此API实现功能与TensorFlow基本一致，部分参数名不一样，并且比TensorFlow多出update_slots、loss_scale、weight_decay参数。

| 分类 | 子类   | TensorFlow                | MindSpore     | 差异                                                         |
| ---- | ------ | ------------------------- | ------------- | ------------------------------------------------------------ |
| 参数 | 参数1  | learning_rate             | learning_rate | -                                                            |
|      | 参数2  | initial_accumulator_value | accum         | 功能一致，参数名不同                                         |
|      | 参数3  | epsilon                   | -             | TensorFlow用于保持数值稳定性的小浮点值，MindSpore无此参数  |
|      | 参数4  | name                      | -             | 不涉及                                                       |
|      | 参数5  | **kwargs                  | -             | 不涉及                                                       |
|      | 参数6  | -                         | params        | Parameter组成的列表或字典组成的列表，TensorFlow中无此参数    |
|      | 参数7  | -                         | update_slots  | 值如果为True，则更新累加器，TensorFlow中无此参数             |
|      | 参数8  | -                         | loss_scale    | 梯度缩放系数，TensorFlow中无此参数                           |
|      | 参数9  | -                         | weight_decay  | 要乘以权重的权重衰减值，TensorFlow中无此参数                 |
|      | 参数10 | -                         | grads         | 优化器中params的梯度，形状与params相同，TensorFlow中无此参数 |

### 代码示例1

> learning_rate均设置为0.1，累加器的初始值均设置为0.1，两API功能一致，用法相同。

```python
# TensorFlow
import tensorflow as tf

opt = tf.keras.optimizers.Adagrad(initial_accumulator_value=0.1, learning_rate=0.1)
var = tf.Variable(1.0)
val0 = var.value()
loss = lambda: (var ** 2)/2.0
step_count = opt.minimize(loss, [var]).numpy()
val1 = var.value()
print([val1.numpy()])
# [0.9046537]
step_count = opt.minimize(loss, [var]).numpy()
val2 = var.value()
print([val2.numpy()])
# [0.8393387]

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
optim = nn.Adagrad(params=net.trainable_params(), accum=0.1, learning_rate=0.1)
model = ms.Model(net, loss_fn=loss, optimizer=optim)
data_x = np.array([1.0], dtype=np.float32)
data_y = np.array([0.0], dtype=np.float32)
data = NumpySlicesDataset((data_x, data_y), ["x", "y"])
input_x = ms.Tensor(np.array([1.0], np.float32))
y0 = net(input_x)
model.train(1, data)
y1 = net(input_x)
print(y1)
# [0.9046537]
model.train(1, data)
y2 = net(input_x)
print(y2)
# [0.8393387]
```
