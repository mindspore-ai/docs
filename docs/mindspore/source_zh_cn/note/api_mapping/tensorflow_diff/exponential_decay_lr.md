# 比较与tf.compat.v1.train.exponential_decay的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/exponential_decay_lr.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

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

更多内容详见[tf.compat.v1.train.exponential_decay](https://www.tensorflow.org/versions/r2.6/api_docs/python/tf/compat/v1/train/exponential_decay)。

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

更多内容详见[mindspore.nn.exponential_decay_lr](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.exponential_decay_lr.html)。

## 差异对比

TensorFlow：基于指数衰减函数计算学习率。

MindSpore：MindSpore此API实现功能与TensorFlow基本一致。

| 分类 | 子类  | TensorFlow    | MindSpore      | 差异                   |
| ---- | ----- | ------------- | -------------- | ---------------------- |
| 参数 | 参数1 | learning_rate | learning_rate  | -                      |
|      | 参数2 | global_step   | total_step     | 功能一致，参数名称不同 |
|      | 参数3 | decay_steps   | decay_epoch    | 功能一致，参数名称不同 |
|      | 参数4 | decay_rate    | decay_rate     | -                      |
|      | 参数5 | staircase     | is_stair       | 功能一致，参数名称不同 |
|      | 参数6 |     name          | -| 不涉及    |
|      | 参数7 |     -          | step_per_epoch | 每个epoch的step数，TensorFlow无此参数    |

### 代码示例

> 两API实现功能一致，用法相同。

```python
# TensorFlow
import tensorflow as tf

learning_rate = 1.0
decay_rate = 0.9
step_per_epoch = 2
epochs = 6
lr = []
for epoch in range(epochs):
    learning_rate = tf.compat.v1.train.exponential_decay(learning_rate, epoch, step_per_epoch, decay_rate, staircase=True)
    lr.append(round(float(learning_rate().numpy()), 2))
print(lr)
# [1.0, 1.0, 0.9, 0.9, 0.81, 0.81]

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