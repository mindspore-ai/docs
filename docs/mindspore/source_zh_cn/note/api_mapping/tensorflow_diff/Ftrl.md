# 比较与tf.keras.optimizers.Ftrl的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/Ftrl.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.keras.optimizers.Ftrl

```python
tf.keras.optimizers.Ftrl(
    learning_rate=0.001, learning_rate_power=-0.5, initial_accumulator_value=0.1,
    l1_regularization_strength=0.0, l2_regularization_strength=0.0, name='Ftrl',
    l2_shrinkage_regularization_strength=0.0, **kwargs
)
```

更多内容详见[tf.keras.optimizers.Ftrl](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/optimizers/Ftrl)。

## mindspore.nn.FTRL

```python
mindspore.nn.FTRL(params, initial_accum=0.1, learning_rate=0.001,
    lr_power=-0.5, l1=0.0, l2=0.0, use_locking=False,
    loss_scale=1.0, weight_decay=0.0)
```

更多内容详见[mindspore.nn.FTRL](https://mindspore.cn/docs/api/zh-CN/master/api_python/nn/mindspore.nn.FTRL.html)。

## 使用方式

一般使用场景：

- MindSpore：一般情况下，在实例化一个优化器子类之后，将其作为`mindspore.model`高阶API的入参参与训练，用法请参考代码示例；或使用`mindspore.nn.TrainOneStepCell`，通过传入优化器和一个`mindspore.nn.WithLossCell`的实例，自定义训练网络，具体实现方式可以参考[官网教程](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/train_and_eval.html#id5)。

- TensorFlow：一般情况下，在实例化一个优化器子类之后，将其作为`tf.keras.models.Model`高阶API的入参参与训练；或调用`minimize()`（包含`compute_gradients()`和`apply_gradients()`）方法单步执行。

其他功能差异：

- 参数分组：一般情况下，MindSpore提供参数分组功能，且支持为不同参数组设置不同配置值，通过入参`params`传入参数组字典实现，但`mindspore.nn.FTRL`不支持参数分组；TensorFlow没有此入参配置。

- 动态学习率：MindSpore支持动态学习率，分别在`nn.dynamic_lr`和`nn.learning_rate_schedule`模块中有不同的实现方法，但`mindspore.nn.FTRL`不支持动态学习率；TensorFlow也支持此功能，学习率设置封装在`tf.train`模块中，`tf.keras.optimizers.Ftrl`支持动态学习率。

- 权重衰减和混合精度：MindSpore的`mindspore.nn.Optimizer`基类支持通过配置入参`weight_decay`和`loss_scale`来进行权重衰减及混合精度设置；TensorFlow的优化器没有相关入参配置，但提供了`tf.keras.regularizers`和`tf.keras.mixed_precision`模块提供相似的功能，配合优化器使用。

## 代码示例

MindSpore：

```python
# The following implements Ftrl with MindSpore.
import mindspore.nn as nn
from mindspore import Model

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

optim = nn.FTRL(params=net.trainable_params())
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
model = Model(net, loss_fn=loss, optimizer=optim, metrics={"accuracy"})
```

TensorFlow：

```python
# The following implements Ftrl with TensorFlow.
import tensorflow as tf
from tensorflow.keras import layers
tf.enable_eager_execution()

# build model and instantiate  optimizer
model = tf.keras.Sequential()
model.add(layers.Dense(1, kernel_initializer='uniform', input_shape=(3,)))
model.add(layers.Activation('relu'))

inputs = tf.constant([[1., 2., 3.], [3., 4., 5.]], dtype=tf.float32)
outputs = tf.constant([[0.5], [0.6]], dtype=tf.float32)

optim = tf.keras.optimizers.Ftrl(learning_rate=0.01)
```

作为`tf.keras.models.Model`入参参与训练：

> 以下训练输出结果均具有随机性。

```python
loss = tf.keras.losses.MeanSquaredError()
model.compile(optimizer=optim, loss=loss)
model.fit(inputs, outputs)

# out: Train on 2 samples
# 2/2 [==============================] - 2s 887ms/sample - loss: 0.2696
```

使用`minimize()`方法单步执行：

```python
logits = model(inputs)
loss = lambda: tf.keras.losses.mse(model(inputs), outputs)

# minimize method
optim.minimize(loss, model.trainable_weights)
print(tf.keras.losses.mse(model(inputs), outputs))

# out: tf.Tensor([0.16361837 0.22712846], shape=(2,), dtype=float32)
```
