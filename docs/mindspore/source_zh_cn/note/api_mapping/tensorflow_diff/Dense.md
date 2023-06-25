# 比较与tf.compat.v1.layers.Dense的差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/Dense.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.compat.v1.layers.Dense

```text
class tf.compat.v1.layers.Dense(
    units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.compat.v1.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    **kwargs
)(x) -> Tensor
```

更多内容详见[tf.compat.v1.layers.Dense](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/compat/v1/layers/Dense)。

## mindspore.nn.Dense

```text
class mindspore.nn.Dense(
    in_channels,
    out_channels,
    weight_init='normal',
    bias_init='zeros',
    has_bias=True,
    activation=None
)(x) -> Tensor
```

更多内容详见[mindspore.nn.Dense](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Dense.html)。

## 差异对比

TensorFlow：全连接层，实现矩阵相乘的运算。

MindSpore：MindSpore此API实现功能与TensorFlow基本一致。

| 分类 | 子类   | TensorFlow           | MindSpore    | 差异                                                         |
| ---- | ------ | -------------------- | ------------ | ------------------------------------------------------------ |
| 参数 | 参数1  | units                | out_channels | 功能一致，参数名称不同                                       |
|      | 参数2  | activation           | activation   | -                                                            |
|      | 参数3  | use_bias             | has_bias     | 功能一致，参数名称不同                                       |
|      | 参数4  | kernel_initializer   | weight_init  | 功能一致，参数名称不同                                       |
|      | 参数5  | bias_initializer     | bias_init    | 功能一致，参数名称不同                                       |
|      | 参数6  | kernel_regularizer   |    -          | 权重矩阵的正则函数。MindSpore无此参数                       |
|      | 参数7  | bias_regularizer     |    -          | 偏差的正则化函数。MindSpore无此参数                         |
|      | 参数8  | activity_regularizer |    -          | 输出的正则化函数。MindSpore无此参数                         |
|      | 参数9  | kernel_constraint    |    -          | 在 `Optimizer` 程序更新后将应用于内核的可选投影函数（例如，用于实现规范约束或层权重的值约束）。该函数必须将未投影的变量作为输入，并且必须返回投影变量（形状必须相同）。在进行异步分布式训练时，使用约束并不安全。MindSpore无此参数 |
|      | 参数10 | bias_constraint      |     -         | 由 `Optimizer`更新后要应用于偏差的可选投影函数。MindSpore无此参数 |
|      | 参数11 | trainable            |     -         | 布尔值，如果为 `True` ，则还将变量添加到图形集合 `GraphKeys.TRAINABLE_VARIABLES`。MindSpore无此参数 |
|      | 参数12 | name                 |     -         | 不涉及    |
|      | 参数13 | **kwargs                 |     -         | 不涉及    |
|      | 参数14 | -                 |     in_channels         | 输入的空间维度，TensorFlow无此参数    |
|  输入   | 单输入 | x                 |     x         | -    |

### 代码示例

> 两API实现功能一致，用法相同。

```python
# TensorFlow
import tensorflow as tf
from tensorflow.compat.v1 import layers
import numpy as np

model = layers.Dense(4)
x = tf.constant(np.array([[180, 234, 154], [244, 48, 247]]),dtype=tf.float32)
output = model(x)
print(output.shape)
# (2, 4)

# MindSpore
import mindspore
from mindspore import Tensor, nn
import numpy as np

x = Tensor(np.array([[180, 234, 154], [244, 48, 247]]), mindspore.float32)
net = nn.Dense(3, 4)
output = net(x)
print(output.shape)
# (2, 4)
```