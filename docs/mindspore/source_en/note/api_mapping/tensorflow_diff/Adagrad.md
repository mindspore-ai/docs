# Function Differences with tf.keras.optimizers.Adagrad

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/Adagrad.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see [tf.keras.optimizers.Adagrad](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/keras/optimizers/Adagrad).

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

For more information, see [mindspore.nn.Adagrad](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Adagrad.html).

## Differences

TensorFlow: Adagrad is an optimizer with a specific parameter learning rate that is used to implement the Adagrad algorithm, and adjusts to the frequency of parameter updates during training. The more updates the parameters receive, the smaller the updates are.

MindSpore: The implementation function of API in MindSpore is basically the same as that of TensorFlow, with different parameter names and more update_slots, loss_scale and weight_decay parameters than TensorFlow.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1  | learning_rate  | learning_rate |-      |
|   | Parameter 2  | initial_accumulator_value | accum   | Same function, different parameter names        |
|      | Parameter 3  | epsilon        | -             | TensorFlow is used to maintain numerical stability of small floating point values, and MindSpore does not have this parameter  |
|  | Parameter 4  | name  | -  | Not involved                                                       |
|   | Parameter 5  | **kwargs | -| Not involved                            |
|      | Parameter 6  | -  | params        | A list of parameters or a list of dictionaries, not available in TensorFlow    |
|      | Parameter 7  | -      | update_slots  | If the value is True, the accumulator is updated, and there is no such parameter in TensorFlow             |
|      | Parameter 8  | -     | loss_scale    | Gradient scaling factor, no such parameter in TensorFlow                           |
|      | Parameter 9  | -     | weight_decay  | The weight decay value to be multiplied by the weights, no such parameter in TensorFlow                 |
|      | Parameter 10 | -   | grads         | Gradient of params in the optimizer, same shape as params, no such parameter in TensorFlow |

### Code Example 1

> The learning_rate is set to 0.1, and the initial value of the accumulator is set to 0.1. Both APIs have the same function and the same usage.

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
