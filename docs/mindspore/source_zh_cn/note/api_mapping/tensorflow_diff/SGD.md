# 比较与tf.keras.optimizers.SGD的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.10/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/SGD.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.10/resource/_static/logo_source.png"></a>

## tf.keras.optimizers.SGD

```python
class tf.keras.optimizers.SGD(
    learning_rate=0.001,
    momentum=0.0,
    nesterov=False,
    name='SGD',
    **kwargs
)
```

更多内容详见[tf.keras.optimizers.SGD](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/optimizers/SGD)。

## mindspore.nn.SGD

```python
class mindspore.nn.SGD(
    params,
    learning_rate=0.1,
    momentum=0.0,
    dampening=0.0,
    weight_decay=0.0,
    nesterov=False,
    loss_scale=1.0
)(gradients)
```

更多内容详见[mindspore.nn.SGD](https://mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.SGD.html)。

## 使用方式

TensorFlow：对所有参数使用相同的学习率，没法设定不同参数组使用不同学习率。

MindSpore：支持所有的参数使用相同的学习率以及不同的参数组使用不同的值的方式。

## 代码示例

```python
# The following implements SGD with MindSpore.
import tensorflow as tf
import mindspore.nn as nn
import mindspore as ms

net = Net()
#1) All parameters use the same learning rate and weight decay
optim = nn.SGD(params=net.trainable_params())

#2) Use parameter groups and set different values
conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True},
                {'params': no_conv_params, 'lr': 0.01},
                {'order_params': net.trainable_params()}]
optim = nn.SGD(group_params, learning_rate=0.1, weight_decay=0.0)
# The conv_params's parameters will use default learning rate of 0.1 and weight decay of 0.01 and grad
# centralization of True.
# The no_conv_params's parameters will use learning rate of 0.01 and default weight decay of 0.0 and grad
# centralization of False.
# The final parameters order in which the optimizer will be followed is the value of 'order_params'.

loss = nn.SoftmaxCrossEntropyWithLogits()
model = ms.Model(net, loss_fn=loss, optimizer=optim)

# The following implements SGD with TensorFlow.
image = tf.keras.layers.Input(shape=(28, 28, 1))
model = tf.keras.models.Model(image, net)
optim = tf.keras.optimizers.SGD()
loss = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=optim, loss=loss, metrics=['accuracy'])
```
