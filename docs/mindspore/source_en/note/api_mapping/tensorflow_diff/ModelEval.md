# Function Differences with tf.keras.Model.predict and tf.keras.Model.predict_generator

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/ModelEval.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.keras.Model.predict

```python
tf.keras.Model.predict(
    x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
    workers=1, use_multiprocessing=False
)
```

For more information, see [tf.keras.Model.predict](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/keras/Model#predict).

## tf.keras.Model.predict_generator

```python
tf.keras.Model.predict_generator(
    generator, steps=None, callbacks=None, max_queue_size=10, workers=1,
    use_multiprocessing=False, verbose=0
)
```

For more information, see [tf.keras.Model.predict_generator](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/keras/Model#predict_generator).

## mindspore.train.Model.eval

```python
mindspore.train.Model.eval(valid_dataset, callbacks=None, dataset_sink_mode=True)
```

For more information, see [mindspore.train.Model.eval](https://www.mindspore.cn/docs/en/master/api_python/train/mindspore.train.Model.html#mindspore.train.Model.eval).

## Usage

`tf.keras.Model.predict` and `tf.keras.Model.predict_generator` support different ways of loading datasets, in addition to the basic `callbacks`. You can also configure the number of processes in multi-threaded scenarios through `workers` and `use_multiprocessing`.

`mindspore.train.Model.train` can be configured with the basic parameters `valid_dataset`, `callbacks`, and also `dataset_sink_mode` to set whether to sink.

## Code Example

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
import mindspore as ms
from mindspore.train import Model
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
model = Model(network=net, loss_fn=crit, optimizer=opt, metrics={"mae"})

test_dataset = create_dataset()
model.eval(test_dataset)
```
