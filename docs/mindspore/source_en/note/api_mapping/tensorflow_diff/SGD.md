# Function Differences with tf.keras.optimizers.SGD

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/SGD.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see [tf.keras.optimizers.SGD](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/keras/optimizers/SGD).

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

For more information, see [mindspore.nn.SGD](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.SGD.html).

## Differences

TensorFlow: Implement optimizer function of stochastic gradient descent (driven volume).

MindSpore: MindSpore API implements the same functions as TensorFlow.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | learning_rate | learning_rate |Same function, different default values |
| | Parameter 2 | momentum | momentum |- |
| | Parameter 3 | nesterov | nesterov |- |
| | Parameter 4 | name | - |Not involved |
| | Parameter 5 | **kwargs | - | Not involved |
| | parameter 6 | - | params |A list composed of Parameter classes or a list composed of dictionaries, which are not available in TensorFlow. |
| | parameter 7 | - | dampening |Floating-point momentum damping value. Default value: 0.0. TensorFlow does not have this parameter. |
| | parameter 8| - | weight_decay |Weight decay (L2 penalty). Default value: 0.0. TensorFlow does not have this parameter |
| | parameter 9 | - | loss_scale |Gradient scaling factor. Default value: 1.0. TensorFlow does not have this parameter |
| | parameter 10 | - | gradients  |Gradient of `params` in the optimizer. TensorFlow does not have this parameter |

### Code Example

> Both APIs implement the same function.

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
