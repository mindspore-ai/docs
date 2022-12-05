# 比较与tf.compat.v1.train.linear_cosine_decay的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/CosineDecayLR.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

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

更多内容详见[tf.compat.v1.train.linear_cosine_decay](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/compat/v1/train/linear_cosine_decay)。

## mindspore.nn.CosineDecayLR

```text
class mindspore.nn.CosineDecayLR(
    min_lr,
    max_lr,
    decay_steps
)(global_step) -> Tensor
```

更多内容详见[mindspore.nn.CosineDecayLR](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.CosineDecayLR.html)。

## 差异对比

TensorFlow：基于余弦衰减函数计算学习率。

MindSpore：与TensorFlow现同样的功能，依据的计算公式不同。

| 分类 | 子类 |TensorFlow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | learning_rate | - |初始学习速率，MindSpore无此参数 |
| | 参数2 | global_step | global_step |- |
| | 参数3 | decay_steps | decay_steps |- |
| | 参数4 | num_periods | - |余弦部分衰减的周期数，MindSpore无此参数 |
| | 参数5 | alpha | - |计算公式中的α参数，MindSpore无此参数 |
| | 参数6 | beta | - |计算公式中的β参数，MindSpore无此参数 |
| | 参数7 | name | - | 不涉及 |
| | 参数8 | - | min_lr |学习率的最小值 |
| | 参数9 | - | max_lr |学习率的最大值 |

### 代码示例1

> 两API实现功能相同，计算逻辑不同。

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
