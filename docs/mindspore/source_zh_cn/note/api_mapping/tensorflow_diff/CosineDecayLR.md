# 比较与tf.compat.v1.train.cosine_decay的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/CosineDecayLR.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## tf.compat.v1.train.cosine_decay

```text
tf.compat.v1.train.cosine_decay(
    learning_rate,
    global_step,
    decay_steps,
    alpha=0.0,
    name=None
) -> Tensor
```

更多内容详见[tf.compat.v1.train.cosine_decay](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/compat/v1/train/cosine_decay)。

## mindspore.nn.CosineDecayLR

```text
class mindspore.nn.CosineDecayLR(
    min_lr,
    max_lr,
    decay_steps
)(global_step) -> Tensor
```

更多内容详见[mindspore.nn.CosineDecayLR](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.CosineDecayLR.html)。

## 差异对比

TensorFlow：基于余弦衰减函数计算学习率。

MindSpore：与TensorFlow实现基本一致功能。在MindSpore的max_lr固定为1后，TensorFlow输出的是衰减后的学习率，而MindSpore输出的是衰减的比率。也就是说，MindSpore输出结果乘上与TensorFlow相同的learning_rate，两者就能得出一致的结果。

| 分类 | 子类 |TensorFlow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | learning_rate | - |初始学习速率，MindSpore无此参数 |
| | 参数2 | global_step | global_step |- |
| | 参数3 | decay_steps | decay_steps |- |
| | 参数4 | alpha | min_lr |功能一致，参数名不同 |
| | 参数5 | name | - | 不涉及 |
| | 参数6 | - | max_lr |学习率的最大值，TensorFlow无此参数 |

### 代码示例

> MindSpore接口的max_lr固定为1，其输出结果乘上与TensorFlow相同的learning_rate，两API实现功能相同。

```python
# TensorFlow
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
learning_rate = 0.01
global_steps = 2
decay_steps = 4
output = tf.compat.v1.train.cosine_decay(learning_rate, global_steps, decay_steps)
ss = tf.compat.v1.Session()
print(ss.run(output))
#0.009999999

# MindSpore
import mindspore
from mindspore import Tensor, nn

min_lr = 0.01
max_lr = 1
decay_steps = 4
global_steps = Tensor(2, mindspore.int32)
cosine_decay_lr = nn.CosineDecayLR(min_lr, max_lr, decay_steps)
output = cosine_decay_lr(global_steps)
print(output)
#0.0101
```