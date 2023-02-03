# Function Differences with torch.optim.Adagrad

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Adagrad.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png"></a>

## torch.optim.Adagrad

```python
class torch.optim.Adagrad(
    params,
    lr=0.01,
    lr_decay=0,
    weight_decay=0,
    initial_accumulator_value=0,
    eps=1e-10
)
```

For more information, see [torch.optim.Adagrad](https://pytorch.org/docs/1.5.0/optim.html#torch.optim.Adagrad).

## mindspore.nn.Adagrad

```python
class mindspore.nn.Adagrad(
    params,
    accum=0.1,
    learning_rate=0.001,
    update_slots=True,
    loss_scale=1.0,
    weight_decay=0.0
)(grads)
```

For more information, see [mindspore.nn.Adagrad](https://mindspore.cn/docs/en/r2.0.0-alpha/api_python/nn/mindspore.nn.Adagrad.html#mindspore.nn.Adagrad).

## Differences

PyTorch: Parameters to be optimized should be put into an iterable parameter then passed as a whole. The `step` method is also implemented to perform one single step optimization and return loss.

MindSpore: The ways of the same learning rate for all parameters and different values for different parameter groups are supported.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1  | learning_rate         | learning_rate | -    |
|      | Parameter 2  | initial_accumulator_value | accum    | Same function, different parameter names             |
|      | Parameter 3  | epsilon                   | -             | TensorFlow is used to maintain numerical stability of small floating point values. MindSpore does not have this parameter |
|      | Parameter 4  | name     | -   | Not involved           |
|      | Parameter 5  | **kwargs      | -  | Not involved      |
|      | Parameter 6  | -       | params        | A list of parameters or a list of dictionaries, not available in TensorFlow |
|      | Parameter 7  | -      | update_slots  | If the value is True, the accumulator is updated. TensorFlow does not have this parameter          |
|      | Parameter 8  | -       | loss_scale    | gradient scaling factor, default value: 1.0. TensorFlow does not have this parameter           |
|      | Parameter 9  | -        | weight_decay  | weight decay (L2 penalty), default value: 0.0. TensorFlow does not have this parameter |
| Input | Single input | -   | grads         | The gradient of `params` in the optimizer. TensorFlow does not have this parameter       |

### Code Example

> The two APIs basically achieve the same function.

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
