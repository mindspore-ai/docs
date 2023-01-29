# Function Differences with tf.keras.optimizers.Ftrl

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/FTRL.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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
    **kwargs
) -> Tensor
```

For more information, see [tf.keras.optimizers.Ftrl](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/keras/optimizers/Ftrl).

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
    weight_decay=0.0
)(grads) -> Tensor
```

For more information, see [mindspore.nn.FTRL](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.FTRL.html).

## Differences

TensorFlow: An online convex optimization algorithm for shallow models with large and sparse feature feature spaces.

MindSpore: MindSpore API basically implements the same function as TensorFlow.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1 | learning_rate | learning_rate | - |
|      | Parameter 2 | learning_rate_power        | lr_power | Same function, different parameter names |
|      | Parameter 3 | initial_accumulator_value  | initial_accum | Same function, different parameter names |
|      | Parameter 4 | l1_regularization_strength | l1 | Same function, different parameter names |
|      | Parameter 5 | l2_regularization_strength | l2 | Same function, different parameter names |
|      | Parameter 6 | name | - | Not involved |
|      | Parameter 7 | l2_shrinkage_regularization_strength | weight_decay | Same function, different parameter names |
|      | Parameter 8 | beta |      -     | A floating point value, the default value is 0.0. MindSpore does not have this parameter.                 |
|      | Parameter 9 | **kwargs | - | Not involved |
|      | Parameter 10 | - | params | A list composed of Parameter or a list composed of dictionaries. TensorFlow does not have this parameter. |
|      | Parameter 11 | - | use_locking | If True, the update operation is protected with a lock, the default value is False. TensorFlow does not have this parameter. |
|      | Parameter 12 | - | loss_scale | The gradient scaling is sparse and must be greater than 0. If loss_scale is an integer, it will be converted to a floating point number. The default value is normally used, but only if the FixedLossScaleManager is used for training and the drop_overflow_update property of the FixedLossScaleManager is configured to False, this value needs to be the same as the loss_scale in the FixedLossScaleManager. The default value is 1.0. TensorFlow does not have this parameter. |
|      | Parameter 13 | - | grads | Reverse input. TensorFlow does not have this parameter. |

### Code Example 1

> The learning_rate is set to 0.1. The two APIs achieve the same function and have the same usage.

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
model.train(1, data)
y2 = net(input_x)
print(y2)
# [0.6031424]
# [0.5532904]
```

### Code Example 2

> The learning_rate is set to 0.01. The two APIs achieve the same function and have the same usage.

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
model.train(1, data)
y2 = net(input_x)
print(y2)
# [0.688954]
# [0.6834637]
```
