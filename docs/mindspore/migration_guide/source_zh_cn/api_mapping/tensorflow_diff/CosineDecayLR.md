# 比较与tf.train.linear_cosine_decay的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_zh_cn/api_mapping/tensorflow_diff/CosineDecayLR.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

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

更多内容详见[tf.train.linear_cosine_decay](http://www.tensorflow.org/versions/r1.15/api_docs/python/tf/train/linear_cosine_decay)。

## mindspore.nn.CosineDecayLR

```python
class mindspore.nn.CosineDecayLR(
    min_lr,
    max_lr,
    decay_steps
)(global_step)
```

更多内容详见[mindspore.nn.CosineDecayLR](https://mindspore.cn/docs/api/zh-CN/r1.5/api_python/nn/mindspore.nn.CosineDecayLR.html)。

## 使用方式

TensorFlow: 计算公式如下：

`global_step = min(global_step, decay_steps)`

`linear_decay = (decay_steps - global_step) / decay_steps`

`cosine_decay = 0.5 * (1 + cos(pi * 2 * num_periods * global_step / decay_steps))`

`decayed = (alpha + linear_decay) * cosine_decay + beta`

`decayed_learning_rate = learning_rate * decayed`

MindSpore：计算逻辑和Tensorflow不一样，计算公式如下：
`current_step = min(global_step, decay_step)`

`decayed_learning_rate = min_lr + 0.5 * (max_lr - min_lr) *(1 + cos(pi * current_step / decay_steps))`

## 代码示例

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
