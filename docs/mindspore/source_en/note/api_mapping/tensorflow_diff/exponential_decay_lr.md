# Function Differences with tf.compat.v1.train.exponential_decay

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/exponential_decay_lr.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.compat.v1.train.exponential_decay

```text
tf.compat.v1.train.exponential_decay(
    learning_rate,
    global_step,
    decay_steps,
    decay_rate,
    staircase=False,
    name=None
) -> Tensor
```

For more information, see [tf.compat.v1.train.exponential_decay](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/compat/v1/train/exponential_decay).

## mindspore.nn.exponential_decay_lr

```text
mindspore.nn.exponential_decay_lr(
    learning_rate,
    decay_rate,
    total_step,
    step_per_epoch,
    decay_epoch,
    is_stair=False
) -> list[float]
```

For more information, see [mindspore.nn.exponential_decay_lr](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.exponential_decay_lr.html).

## Differences

TensorFlow: calculate the learning rate based on the exponential decay function.

MindSpore: MindSpore API basically implements the same function as TensorFlow.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1 | learning_rate | learning_rate  | -                      |
|      | Parameter 2 | global_step   | total_step     | Same function, different parameter names |
|      | Parameter 3 | decay_steps   | decay_epoch    | Same function, different parameter names |
|      | Parameter 4 | decay_rate    | decay_rate     | -                      |
|      | Parameter 5 | staircase     | is_stair       | Same function, different parameter names |
|      | Parameter 6 |     name          | -| Not involved    |
|      | Parameter 7 |     -          | step_per_epoch | The number of steps per epoch, TensorFlow does not have this parameter    |

### Code Example

> The two APIs achieve the same function and have the same usage.

```python
# TensorFlow
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

# Create an instance of the model
model = MyModel()
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
epochs = 1
global_step = tf.Variable(0, trainable=False, dtype= tf.int32)
starter_learning_rate = 1.0
lr = []
for epoch in range(epochs):
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        learning_rate = tf.compat.v1.train.exponential_decay(
                    starter_learning_rate,
                    global_step,
                    2,
                    0.9,
                    staircase=True)
        tf.keras.optimizers.SGD(learning_rate=learning_rate).apply_gradients(zip(grads, model.trainable_weights))
        lr.append(learning_rate().numpy())
        global_step.assign_add(1)
        if global_step == 6:
            break
print(lr)
# [1.0, 1.0, 0.9, 0.9, 0.80999994, 0.80999994]

# MindSpore
import mindspore.nn as nn

learning_rate = 1.0
decay_rate = 0.9
total_step = 6
step_per_epoch = 2
decay_epoch = 1
output = nn.exponential_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch)
print(output)
# [1.0, 1.0, 0.9, 0.9, 0.81, 0.81]
```
