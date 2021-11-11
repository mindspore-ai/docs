# 比较与tf.train.MomentumOptimizer的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_zh_cn/api_mapping/tensorflow_diff/Momentum.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## tf.train.MomentumOptimizer

```python
class tf.train.MomentumOptimizer(
    learning_rate,
    momentum,
    use_locking=False,
    name='Momentum',
    nesterov=False
)
```

更多内容详见[tf.train.MomentumOptimizer](http://www.tensorflow.org/versions/r1.15/api_docs/python/tf/train/MomentumOptimizer)。

## mindspore.nn.Momentum

```python
class mindspore.nn.Momentum(
    params,
    learning_rate
    momentum,
    weight_decay=0.0,
    loss_scale=1.0,
    use_nesterov=False
)(grads)
```

更多内容详见[mindspore.nn.Momentum](https://mindspore.cn/docs/api/zh-CN/r1.5/api_python/nn/mindspore.nn.Momentum.html)。

## 使用方式

TensorFlow：对所有参数使用相同的学习率，没法设定不同参数组使用不同学习率。

MindSpore：支持所有的参数使用相同的学习率以及不同的参数组使用不同的值的方式。

## 代码示例

```python
# The following implements Momentum with MindSpore.
import tensorflow as tf
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import Model

net = Net()
#1) All parameters use the same learning rate and weight decay
optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)

#2) Use parameter groups and set different values
conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True},
                {'params': no_conv_params, 'lr': 0.01},
                {'order_params': net.trainable_params()}]
optim = nn.Momentum(group_params, learning_rate=0.1, weight_decay=0.0)
# The conv_params's parameters will use default learning rate of 0.1 and weight decay of 0.01 and grad
# centralization of True.
# The no_conv_params's parameters will use learning rate of 0.01 and default weight decay of 0.0 and grad
# centralization of False.
# The final parameters order in which the optimizer will be followed is the value of 'order_params'.

loss = nn.SoftmaxCrossEntropyWithLogits()
model = Model(net, loss_fn=loss, optimizer=optim)

# The following implements MomentumOptimizer with TensorFlow.
image = tf.keras.layers.Input(shape=(28, 28, 1))
model = tf.keras.models.Model(image, net)
optim = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9)
loss = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=optim, loss=loss, metrics=['accuracy'])
```
