# 比较与tf.keras.Model.predict、tf.keras.Model.predict_generator的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_zh_cn/api_mapping/tensorflow_diff/ModelEval.md " target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## tf.keras.Model.predict

```python
tf.keras.Model.predict(
    x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
    workers=1, use_multiprocessing=False
)
```

更多内容详见[tf.keras.Model.predict](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/Model#predict)。

## tf.keras.Model.predict_generator

```python
tf.keras.Model.predict_generator(
    generator, steps=None, callbacks=None, max_queue_size=10, workers=1,
    use_multiprocessing=False, verbose=0
)
```

更多内容详见[tf.keras.Model.predict_generator](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/Model#predict_generator)。

## mindspore.Model.eval

```python
mindspore.Model.eval(valid_dataset, callbacks=None, dataset_sink_mode=True)
```

更多内容详见[mindspore.Model.eval](https://mindspore.cn/docs/api/zh-CN/r1.6/api_python/mindspore/mindspore.Model.html#mindspore.Model.eval)。

## 使用方式

`tf.keras.Model.predict`和`tf.keras.Model.predict_generator`分别支持数据集的不同载入方式，除基本的`callbacks`等，还可通过`workers`、 `use_multiprocessing`配置多线程场景下的进程数等。

`mindspore.Model.train`除了可配置基本的参数`valid_dataset`、`callbacks`，还可以配置`dataset_sink_mode`设置是否下沉。

## 代码示例

```python
import tensorflow as tf
import numpy as np

inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# predict
inputs_x = np.random.rand(10, 3)
pred_result = model.predict(inputs_x, batch_size=1)

# predict_generator
def generate_data(data_num):
    for _ in range(data_num):
        yield np.random.rand(2, 3), np.random.rand(2, 5)
model.predict_generator(generate_data(5), steps=2)
```

```python
from mindspore import nn, Model
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
model = Model(network=net, loss_fn=crit, optimizer=opt, metrics={"mae"})

test_dataset = create_dataset()
model.eval(test_dataset)
```
