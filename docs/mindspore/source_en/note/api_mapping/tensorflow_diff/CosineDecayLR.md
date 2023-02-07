# Function Differences with tf.compat.v1.train.linear_cosine_decay

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/CosineDecayLR.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.compat.v1.train.linear_cosine_decay

```text
tf.compat.v1.train.linear_cosine_decay(
    learning_rate,
    global_step,
    decay_steps,
    num_periods=0.5,
    alpha=0.0,
    beta=0.001,
    name=None
) -> Tensor
```

For more information, see [tf.compat.v1.train.linear_cosine_decay](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/compat/v1/train/linear_cosine_decay).

## mindspore.nn.CosineDecayLR

```text
class mindspore.nn.CosineDecayLR(
    min_lr,
    max_lr,
    decay_steps
)(global_step) -> Tensor
```

For more information, see [mindspore.nn.CosineDecayLR](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.CosineDecayLR.html).

## Differences

TensorFlow: The learning rate is calculated based on the cosine decay function.

MindSpore: The same function as TensorFlow now, based on a different calculation formula.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | learning_rate | - |Initial learning rate. MindSpore does not have this parameter |
| | Parameter 2 | global_step | global_step |- |
| | Parameter 3 | decay_steps | decay_steps |- |
| | Parameter 4 | num_periods | - | The number of periods of cosine decay. MindSpore does not have this parameter |
| | Parameter 5 | alpha | - |Calculate the α parameter in the formula. MindSpore does not have this parameter |
| | Parameter 6 | beta | - |Calculate the β parameter in the formula. MindSpore does not have this parameter |
| | Parameter 7 | name | - | Not involved |
| | Parameter 8 | - | min_lr |The minimum value of learning rate. TensorFlow does not have this parameter|
| | Parameter 9 | - | max_lr |The maximum value of learning rate. TensorFlow does not have this parameter |

### Code Example

> The two APIs achieve the same function, based on different calculation formulas.

```python
# TensorFlow
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
learning_rate = 0.01
global_steps = 2
decay_steps = 4
output = tf.compat.v1.train.linear_cosine_decay(learning_rate, global_steps, decay_steps)
ss = tf.compat.v1.Session()
print(ss.run(output))
#0.0025099998

# MindSpore
import mindspore
from mindspore import Tensor, nn

min_lr = 0.01
max_lr = 0.1
decay_steps = 4
global_steps = Tensor(2, mindspore.int32)
cosine_decay_lr = nn.CosineDecayLR(min_lr, max_lr, decay_steps)
output = cosine_decay_lr(global_steps)
print(output)
#0.055
```
