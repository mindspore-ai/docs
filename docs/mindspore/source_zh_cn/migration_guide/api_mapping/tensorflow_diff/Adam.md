# 比较与tf.keras.optimizers.Adam的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/tensorflow_diff/Adam.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.keras.optimizers.Adam

```python
class tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
    **kwargs
)
```

更多内容详见[tf.keras.optimizers.Adam](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/optimizers/Adam)。

## mindspore.nn.Adam

```python
class mindspore.nn.Adam(
    params,
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    update_locking=False,
    use_nesterov=False,
    weight_decay=0.0,
    loss_scale=1.0
)(grads)
```

更多内容详见[mindspore.nn.Adam](https://mindspore.cn/docs/api/zh-CN/master/api_python/nn/mindspore.nn.Adam.html)。

## 使用方式

TensorFlow：对所有参数使用相同的学习率，没法设定不同参数组使用不同学习率。

MindSpore：支持所有的参数使用相同的学习率以及不同的参数组使用不同的值的方式。

## 代码示例

```python
# The following implements Adam with MindSpore.
import tensorflow as tf
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import Model

net = Net()
#1) All parameters use the same learning rate and weight decay
optim = nn.Adam(params=net.trainable_params())

#2) Use parameter groups and set different values
conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True},
                {'params': no_conv_params, 'lr': 0.01},
                {'order_params': net.trainable_params()}]
optim = nn.Adam(group_params, learning_rate=0.1, weight_decay=0.0)
# The conv_params's parameters will use default learning rate of 0.1 and weight decay of 0.01 and grad
# centralization of True.
# The no_conv_params's parameters will use learning rate of 0.01 and default weight decay of 0.0 and grad
# centralization of False.
# The final parameters order in which the optimizer will be followed is the value of 'order_params'.

loss = nn.SoftmaxCrossEntropyWithLogits()
model = Model(net, loss_fn=loss, optimizer=optim)

# The following implements Adam with TensorFlow.
image = tf.keras.layers.Input(shape=(28, 28, 1))
model = tf.keras.models.Model(image, net)
optim = tf.keras.optimizers.Adam()
loss = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=optim, loss=loss, metrics=['accuracy'])
```
