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

For more information, see [tf.compat.v1.train.exponential_decay](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/train/exponential_decay).

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

learning_rate = 1.0
decay_rate = 0.9
step_per_epoch = 2
epochs = 3
lr = []
for epoch in range(epochs):
    learning_rate = tf.compat.v1.train.exponential_decay(learning_rate, epoch, step_per_epoch, decay_rate, staircase=True)
    learning_rate = learning_rate().numpy().item()
    lr.append(round(float(learning_rate), 2))
print(lr)
# [1.0, 1.0, 0.9]

# MindSpore
import mindspore.nn as nn

learning_rate = 1.0
decay_rate = 0.9
total_step = 3
step_per_epoch = 2
decay_epoch = 1
output = nn.exponential_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch)
print(output)
# [1.0, 1.0, 0.9]
```
