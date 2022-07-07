# 比较与tf.keras.optimizers.RMSprop的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.8/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/RMSProp.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_source.png"></a>

## tf.keras.optimizers.RMSprop

```python
tf.keras.optimizers.RMSprop(
    learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
    name='RMSprop', **kwargs
)
```

更多内容详见[tf.keras.optimizers.RMSprop](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/optimizers/RMSprop)。

## mindspore.nn.RMSProp

```python
mindspore.nn.RMSProp(
    params, learning_rate=0.1, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False,
    centered=False, loss_scale=1.0, weight_decay=0.0)
```

更多内容详见[mindspore.nn.RMSProp](https://mindspore.cn/docs/zh-CN/r1.8/api_python/nn/mindspore.nn.RMSProp.html#mindspore.nn.RMSProp)。

## 使用方式

一般使用场景：

- MindSpore：一般情况下，在实例化一个优化器子类之后，将其作为`mindspore.model`高阶API的入参参与训练，用法请参考代码示例；或使用`mindspore.nn.TrainOneStepCell`，通过传入优化器和一个`mindspore.nn.WithLossCell`的实例，自定义训练网络，具体实现方式可以参考[官网教程](https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/train/train_eval.html#自定义训练网络)。

- TensorFlow：一般情况下，在实例化一个优化器子类之后，将其作为`tf.keras.models.Model`高阶API的入参参与训练；或调用`minimize()`（包含`compute_gradients()`和`apply_gradients()`）方法单步执行。

其他功能差异：

- 参数分组：MindSpore提供参数分组功能，且支持为不同参数组设置不同配置值，通过入参`params`传入参数组字典实现，`mindspore.nn.RMSProp`支持参数分组；TensorFlow没有此入参配置。

- 动态学习率：MindSpore支持动态学习率，分别在`nn.dynamic_lr`和`nn.learning_rate_schedule`模块中有不同的实现方法，`mindspore.nn.RMSProp`支持动态学习率；TensorFlow也支持此功能，学习率设置封装在`tf.train`模块中。

- 权重衰减和混合精度：MindSpore的`mindspore.nn.Optimizer`基类支持通过配置入参`weight_decay`和`loss_scale`来进行权重衰减及混合精度设置；TensorFlow的优化器没有相关入参配置，但提供了`tf.keras.regularizers`和`tf.keras.mixed_precision`模块提供相似的功能，配合优化器使用。

## 代码示例

MindSpore：

```python
# The following implements RMSprop with MindSpore.
import mindspore.nn as nn
import mindspore as ms

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.bn = nn.BatchNorm2d(64)

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

net = Net()

# 1) All parameters use the same learning rate and weight decay
optim = nn.RMSProp(params=net.trainable_params())

# 2) Use parameter groups and set different values
conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))

group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True}, {'params': no_conv_params, 'lr': 0.01},  {'order_params': net.trainable_params()}]

optim = nn.RMSProp(group_params, learning_rate=0.1, weight_decay=0.0)
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
model = ms.Model(net, loss_fn=loss, optimizer=optim, metrics={"accuracy"})
```

TensorFlow：

```python
# The following implements RMSprop with tensorflow.
import tensorflow as tf
from tensorflow.keras import layers
tf.enable_eager_execution()

# build model and instantiate RMSprop optimizer
model = tf.keras.Sequential()
model.add(layers.Dense(1, kernel_initializer='uniform', input_shape=(3,)))
model.add(layers.Activation('relu'))

inputs = tf.constant([[1., 2., 3.], [3., 4., 5.]], dtype=tf.float32)
outputs = tf.constant([[0.5], [0.6]], dtype=tf.float32)

optim = tf.keras.optimizers.RMSprop(learning_rate=0.01)
```

作为`tf.keras.models.Model`入参参与训练：

> 以下训练输出结果均具有随机性。

```python
loss = tf.keras.losses.MeanSquaredError()
model.compile(optimizer=optim, loss=loss)
model.fit(inputs, outputs)

# out:
# Train on 2 samples
# 2/2 [==============================] - 3s 2s/sample - loss: 0.3050
```

使用`minimize()`方法单步执行：

```python
logits = model(inputs)
loss = lambda: tf.keras.losses.mse(model(inputs), outputs)

# minimize method
optim.minimize(loss, model.trainable_weights)
print(tf.keras.losses.mse(model(inputs), outputs))

# out: tf.Tensor([0.05567991 0.00738888], shape=(2,), dtype=float32)
```
