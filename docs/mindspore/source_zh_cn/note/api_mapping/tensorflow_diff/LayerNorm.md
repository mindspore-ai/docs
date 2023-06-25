# 比较与tf.keras.layers.LayerNormalization的差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/LayerNorm.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.keras.layers.LayerNormalization

```python
class tf.keras.layers.LayerNormalization(
    axis=-1,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    **kwargs
)(x) -> Tensor
```

更多内容详见[tf.keras.layers.LayerNormalization](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/keras/layers/LayerNormalization)。

## mindspore.nn.LayerNorm

```python
class mindspore.nn.LayerNorm(
    normalized_shape,
    begin_norm_axis=-1,
    begin_params_axis=-1,
    gamma_init='ones',
    beta_init='zeros',
    epsilon=1e-7
)(x) -> Tensor
```

更多内容详见[mindspore.nn.LayerNorm](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.LayerNorm.html)。

## 差异对比

TensorFlow：在mini-batch输入上应用层归一化（Layer Normalization），其中，参数`center`和`scale`控制是否使用beta和gamma，`beta_regularizer`和`gamma_regularizer`用于控制是否采用beta和gamma的可选正则化器，参数`beta_constraint`和`gamma_constraint`用于控制是否采用beta和gamma的可选约束。

MindSpore：TensorFlow的此API中各参数均为默认时，MindSpore此API实现功能与TensorFlow基本一致。但MindSpore中不存在参数`center`和`scale`，不能实现忽略beta和gamma的功能；不存在参数`beta_regularizer`，`gamma_regularizer`，`beta_constraint`，和`gamma_constraint`，暂不支持实现对应功能；同时MindSpore此API增加了参数`begin_params_axis`控制第一个参数(beta, gamma)的维度，以及参数`normalized_shape`用来控制平均值和标准差计算的特定维度。

| 分类 | 子类 |TensorFlow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | axis | begin_norm_axis | 功能一致，参数名不同|
| | 参数2 | epsilon | epsilon | 功能一致，参数名相同，默认值不同 |
| | 参数3 | center | - | TensorFlow中此参数用于控制是否使用beta偏移量，MindSpore无此参数|
| | 参数4 | scale | - | TensorFlow中此参数用于控制是否使用gamma，MindSpore无此参数|
| | 参数5 | beta_initializer | beta_init | 功能一致，参数名不同|
| | 参数6 | gamma_initializer | gamma_init | 功能一致，参数名不同|
| | 参数7 | beta_regularizer | - | TensorFlow中此参数用于控制是否采用beta权重的可选正则化器，MindSpore无此参数|
| | 参数8 | gamma_regularizer | - | TensorFlow中此参数用于控制是否采用gamma权重的可选正则化器，MindSpore无此参数|
| | 参数9 | beta_constraint | - | TensorFlow中此参数用于控制是否采用beta权重的可选约束，MindSpore无此参数|
| | 参数10 | gamma_constraint | - | TensorFlow中此参数用于控制是否采用gamma权重的可选约束，MindSpore无此参数|
| | 参数11 | **kwargs | - | 不涉及|
| | 参数12 | - | normalized_shape | TensorFlow中无此参数，MindSpore中的此参数控制平均值和标准差计算的特定维度|
| | 参数13 | - | begin_params_axis | TensorFlow中无此参数，MindSpore中的此参数确定第一个参数(beta, gamma)的维度，用于内置参数scale和centering的广播|
|输入 | 单输入 | x | x | - |

### 代码示例

> TensorFlow的此API中各参数均为默认时，两API功能一致，用法相同。

```python
# TensorFlow
import tensorflow as tf

inputs = tf.ones([20, 5, 10, 10])
layer = tf.keras.layers.LayerNormalization(axis=-1)
output = layer(inputs)
print(output.shape)
# (20, 5, 10, 10)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.numpy as np
import mindspore.nn as nn

x = Tensor(np.ones([20, 5, 10, 10]), mindspore.float32)
shape1 = x.shape[1:]
m = nn.LayerNorm(shape1, begin_norm_axis=1, begin_params_axis=1)
output = m(x).shape
print(output)
# (20, 5, 10, 10)
```
