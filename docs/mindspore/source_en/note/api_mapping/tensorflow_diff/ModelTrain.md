# Function Differences with tf.keras.Model.fit and tf.keras.Model.fit_generator

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/ModelTrain.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see [tf.keras.Model.fit](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/keras/Model#fit).

## tf.keras.Model.fit_generator

```python
tf.keras.Model.fit_generator(
    generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None,
    validation_data=None, validation_steps=None, validation_freq=1,
    class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False,
    shuffle=True, initial_epoch=0
)
```

For more information, see [tf.keras.Model.fit_generator](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/keras/Model#fit_generator).

## mindspore.train.Model.train

```python
mindspore.train.Model.train(epoch, train_dataset, callbacks=None, dataset_sink_mode=True, sink_size=-1)
```

For more information, see [mindspore.train.Model.train](https://www.mindspore.cn/docs/en/master/api_python/train/mindspore.train.Model.html).

## Usage

`tf.keras.Model.fit` and `tf.keras.Model.fit_generator` support different methods for loading datasets, in addition to the basic `epoch` and `callbacks` respectively. Set the output information format during training by `verbose` and configure the validation set by `validation*` and other parameters for simultaneous validation during training. The number of processes in multi-threaded scenarios is configured by `workers` and `use_multiprocessing`. Whether the data set is shuffled during training is set by `shuffle`. Other inputs are not detailed, and please refer to the official API documentation for details.

`mindspore.train.Model.train` can be configured with `dataset_sink_mode`, `sink_size` for sinking in addition to the basic training parameters `epoch`, `train_dataset`, `callback`. Other functions are not provided yet.

## Code Example

> The following code results are random.

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

train_dataset = create_dataset()
model.train(2, train_dataset)
```
