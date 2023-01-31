# Function Differences with tf.keras.Model

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/Model.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.keras.Model

```python
tf.keras.Model(*args, **kwargs)
```

For more information, see [tf.keras.Model](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/keras/Model).

## mindspore.train.Model

```python
mindspore.train.Model(network, loss_fn=None, optimizer=None, metrics=None, eval_network=None, eval_indexes=None, amp_level="O0", boost_level="O0", **kwargs)
```

For more information, see [mindspore.train.Model](https://www.mindspore.cn/docs/en/master/api_python/train/mindspore.train.Model.html#mindspore.train.Model).

## Usage

The framework provides a high-level API for model training and inference, and common scenarios for instantiating a Model can be found in the code examples.

## Code Example

TensorFlow:

1. Two ways to instantiate a Model:

  Create a forward pass that creates a Model instance based on the input and output.

  ```python
  import tensorflow as tf

  inputs = tf.keras.Input(shape=(3,))
  x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
  outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  ```

  Inherit the Model class, define the model layer in __init__, and explicitly execute the logic in the call.

  ```python
  import tensorflow as tf

  class MyModel(tf.keras.Model):

    def __init__(self):
      super(MyModel, self).__init__()
      self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
      self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

    def call(self, inputs):
      x = self.dense1(inputs)
      return self.dense2(x)

  model = MyModel()
  ```

2. Use the compile method for model configuration

  ```python
  model.compile(loss='mae', optimizer='adam')
  ```

MindSpore：

```python
import mindspore as ms
from mindspore.train import Model
from mindspore import nn
from mindspore.common.initializer import Normal

class LinearNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(1, 1, Normal(0.02), Normal(0.02))

    def construct(self, x):
        return self.fc(x)

net = LinearNet()
crit = nn.MSELoss()
opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)

model = Model(network=net, loss_fn=crit, optimizer=opt, metrics={"mae"})
```
