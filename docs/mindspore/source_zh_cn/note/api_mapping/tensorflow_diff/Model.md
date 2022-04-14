# 比较与tf.keras.Model的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/Model.md " target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.keras.Model

```python
tf.keras.Model(*args, **kwargs)
```

更多内容详见[tf.keras.Model](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/Model)。

## mindspore.Model

```python
mindspore.Model(network, loss_fn=None, optimizer=None, metrics=None, eval_network=None, eval_indexes=None, amp_level="O0", boost_level="O0", **kwargs)
```

更多内容详见[mindspore.Model](https://mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore/mindspore.Model.html#mindspore.Model)。

## 使用方式

框架提供的模型训练和推理的高阶API，实例化一个Model的常见场景可参考代码示例。

## 代码示例

TensorFlow：

1. 实例化Model的两种方法：

  创建一个前向传递，根据输入输出创建一个Model实例：

  ```python
  import tensorflow as tf

  inputs = tf.keras.Input(shape=(3,))
  x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
  outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  ```

  继承Model类，在__init__中定义模型层，在call中明确执行逻辑。

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

2. 使用compile方法进行模型配置：

 ```python
 model.compile(loss='mae', optimizer='adam')
 ```

MindSpore：

```python
from mindspore import nn, Model
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
