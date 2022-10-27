# 比较与tf.keras.Model.fit、tf.keras.Model.fit_generator的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.9/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/ModelTrain.md " target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source.png"></a>

## tf.keras.Model.fit

```python
tf.keras.Model.fit(
    x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
    validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
    sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    validation_steps=None, validation_freq=1, max_queue_size=10, workers=1,
    use_multiprocessing=False, **kwargs
)
```

更多内容详见[tf.keras.Model.fit](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/Model#fit)。

## tf.keras.Model.fit_generator

```python
tf.keras.Model.fit_generator(
    generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None,
    validation_data=None, validation_steps=None, validation_freq=1,
    class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False,
    shuffle=True, initial_epoch=0
)
```

更多内容详见[tf.keras.Model.fit_generator](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/Model#fit_generator)。

## mindspore.train.Model.train

```python
mindspore.train.Model.train(epoch, train_dataset, callbacks=None, dataset_sink_mode=True, sink_size=-1)
```

更多内容详见[mindspore.train.Model.train](https://www.mindspore.cn/docs/zh-CN/r1.9/api_python/mindspore/mindspore.Model.html)。

## 使用方式

`tf.keras.Model.fit`和`tf.keras.Model.fit_generator`分别支持数据集的不同载入方式，除基本的`epoch`、`callbacks`外；通过`verbose`设置训练过程中的输出信息格式；通过`validation*`等参数配置验证集，用于训练时的同步验证；通过`workers`、 `use_multiprocessing`配置多线程场景下的进程数；通过`shuffle`设置训练时数据集是否混洗；其他入参不做详细说明，具体请参考官网API文档。

`mindspore.train.Model.train`除了基本的训练参数`epoch`、`train_dataset`、`callback`，还可以通过`dataset_sink_mode`，`sink_size`进行下沉配置。其他功能暂未提供。

## 代码示例

> 以下代码结果具有随机性。

```python
import tensorflow as tf
import numpy as np

inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss='mae', optimizer='adam')

# fit
inputs_x = np.random.rand(10, 3)
inputs_y = np.random.rand(10, 5)
model.fit(inputs_x, inputs_y, batch_size=2)
# output:
# 10/10 [==============================] - 0s 18ms/sample - loss: 0.3080


# fit generator
def generate_data(data_num):
    for _ in range(data_num):
        yield np.random.rand(2, 3), np.random.rand(2, 5)
model.fit_generator(generate_data(5), steps_per_epoch=5)

# output:
# 5/5 [==============================] - 0s 77ms/step - loss: 0.3292
```

```python
import mindspore as ms
from mindspore import nn
import numpy as np
from mindspore import dataset as ds

def get_data(num):
    for _ in range(num):
        yield np.random.rand(3).astype(np.float32), np.random.rand(4).astype(np.float32)

def create_dataset(num_data=16, batch_size=4):
    dataset = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
    dataset = dataset.batch(batch_size)
    return dataset

class LinearNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(3, 4)

    def construct(self, x):
        return self.fc(x)

net = LinearNet()
crit = nn.MSELoss()
opt = nn.Momentum(net.trainable_params(), learning_rate=0.05, momentum=0.9)
model = ms.Model(network=net, loss_fn=crit, optimizer=opt, metrics={"mae"})

train_dataset = create_dataset()
model.train(2, train_dataset)
```
