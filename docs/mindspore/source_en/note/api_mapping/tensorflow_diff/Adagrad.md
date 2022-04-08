# Function Differences with tf.keras.optimizers.Adagrad

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/Adagrad.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.keras.optimizers.Adagrad

```python
class tf.keras.optimizers.Adagrad(
    learning_rate=0.001,
    initial_accumulator_value=0.1,
    epsilon=1e-07,
    name='Adagrad',
    **kwargs
)
```

For more information, see [tf.keras.optimizers.Adagrad](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/optimizers/Adagrad).

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

For more information, see [mindspore.nn.Adagrad](https://mindspore.cn/docs/api/en/master/api_python/nn/mindspore.nn.Adagrad.html).

## Differences

TensorFlow: Using the same learning rate for all parameters and it is impossible to use different learning rates for different parameter groups.

MindSpore: Using the same learning rate for all parameters and different values for different parameter groups is supported.

## Code Example

```python
# The following implements Adagrad with MindSpore.
import tensorflow as tf
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import Model

net = Net()
#1) All parameters use the same learning rate and weight decay
optim = nn.Adagrad(params=net.trainable_params())

#2) Use parameter groups and set different values
conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True},
                {'params': no_conv_params, 'lr': 0.01},
                {'order_params': net.trainable_params()}]
optim = nn.Adagrad(group_params, learning_rate=0.1, weight_decay=0.0)
# The conv_params's parameters will use default learning rate of 0.1 and weight decay of 0.01 and grad
# centralization of True.
# The no_conv_params's parameters will use learning rate of 0.01 and default weight decay of 0.0 and grad
# centralization of False.
# The final parameters order in which the optimizer will be followed is the value of 'order_params'.

loss = nn.SoftmaxCrossEntropyWithLogits()
model = Model(net, loss_fn=loss, optimizer=optim)

# The following implements Adagrad with TensorFlow.
image = tf.keras.layers.Input(shape=(28, 28, 1))
model = tf.keras.models.Model(image, net)
optim = tf.keras.optimizers.Adagrad()
loss = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=optim, loss=loss, metrics=['accuracy'])
```
