# Function Differences with tf.train.linear_cosine_decay

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/api_mapping/tensorflow_diff/CosineDecayLR.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## tf.train.linear_cosine_decay

```python
class tf.train.linear_cosine_decay(
    learning_rate,
    global_step,
    decay_steps,
    num_periods=0.5,
    alpha=0.0,
    beta=0.001,
    name=None
)
```

For more information, see[tf.train.linear_cosine_decay](http://www.tensorflow.org/versions/r1.15/api_docs/python/tf/train/linear_cosine_decay).

## mindspore.nn.CosineDecayLR

```python
class mindspore.nn.CosineDecayLR(
    min_lr,
    max_lr,
    decay_steps
)(global_step)
```

For more information, see[mindspore.nn.CosineDecayLR](https://mindspore.cn/docs/api/en/master/api_python/nn/mindspore.nn.CosineDecayLR.html).

## Differences

TensorFlow: The formulas are as follows：
`global_step = min(global_step, decay_steps)
linear_decay = (decay_steps - global_step) / decay_steps
cosine_decay = 0.5 \* (1 + cos(pi \* 2 \* num_periods \* global_step / decay_steps))
decayed = (alpha + linear_decay) \* cosine_decay + beta
decayed_learning_rate = learning_rate \* decayed`

MindSpore：The calculation logic is different from Tensorflow, the formulas are as follows：
`current_step = min(global_step, decay_step)
decayed_learning_rate = min_lr + 0.5 \* (max_lr - min_lr) \*
        (1 + cos(pi \* current_step / decay_steps))`

## Code Example

```python
# The following implements CosineDecayLR with MindSpore.
import numpy as np
import tensorflow as tf
import mindspore
import mindspore.nn as nn
from mindspore import Tensor

min_lr = 0.01
max_lr = 0.1
decay_steps = 4
global_steps = Tensor(2, mindspore.int32)
cosine_decay_lr = nn.CosineDecayLR(min_lr, max_lr, decay_steps)
result = cosine_decay_lr(global_steps)
print(result)
# Out：
# 0.055


# The following implements linear_cosine_decay with TensorFlow.
learging_rate = 0.01
global_steps = 2
output = tf.train.linear_cosine_decay(learging_rate, global_steps, decay_steps)
ss = tf.Session()
ss.run(output)
# out
# 0.0025099998
```
