# 比较与tf.keras.layers.BatchNormalization的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/BatchNorm2d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.keras.layers.BatchNormalization

```text
tf.keras.layers.BatchNormalization(
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    **kwargs
)(x) -> Tensor
```

更多内容详见 [tf.keras.layers.BatchNormalization](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/keras/layers/BatchNormalization)。

## mindspore.nn.BatchNorm2d

```text
class mindspore.nn.BatchNorm2d(
    num_features,
    eps=1e-5,
    momentum=0.9,
    affine=True,
    gamma_init='ones',
    beta_init='zeros',
    moving_mean_init='zeros',
    moving_var_init='ones',
    use_batch_statistics=None,
    data_format='NCHW'
)(x) -> Tensor
```

更多内容详见 [mindspore.nn.BatchNorm2d](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.BatchNorm2d.html)。

## 差异对比

TensorFlow：对输入的数据进行批归一化。

MindSpore：对输入的四维数据进行批归一化(Batch Normalization Layer)。在四维输入（具有额外通道维度的小批量二维输入）上应用批归一化处理，以避免内部协变量偏移。

| 分类 | 子类 |TensorFlow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | input | input | - |
| | 参数2 | axis | - | 应规范化的轴（通常是特征轴），MindSpore无此参数 |
| | 参数3 | moving_mean_initializer | moving_mean_init | 功能相同，参数名不同 |
| | 参数4 | moving_variance_initializer | moving_var_init | 功能相同，参数名不同 |
| | 参数5 | gamma_initializer | gamma_init | 功能相同，参数名不同 |
| | 参数6 | beta_initializer | beta_init | 功能相同，参数名不同 |
| | 参数7 | momentum | momentum | - |
| | 参数8 | variance_epsilon | eps | 功能相同，参数名不同 |
| | 参数9 | center | - | 如果为True，则将偏移量beta添加到归一化张量。如果为False，则beta被忽略。MindSpore无此参数 |
| | 参数10 | scale | - | 如果为True，则乘以gamma。如果为False，则不使用gamma。MindSpore无此参数 |
| | 参数11 | beta_regularizer | - | beta权重的可选正则器。MindSpore无此参数 |
| | 参数12 | gamma_regularizer | - | gamma权重的可选正则器。MindSpore无此参数 |
| | 参数13 | beta_constraint | - | beta权重的可选约束。MindSpore无此参数 |
| | 参数14 | gamma_constraint | - | gamma权重的可选约束。MindSpore无此参数 |
| | 参数15 | - | num_features | 通道数量，输入Tensor shape (N,C,H,W)(N,C,H,W) 中的C |
| | 参数16 | - | affine | bool类型。设置为True时，可学习 γ 和 β 值。默认值：True。 |
| | 参数17 | - | use_batch_statistics | 如果为True，则使用当前批处理数据的平均值和方差值，并跟踪运行平均值和运行方差。<br /> 如果为False，则使用指定值的平均值和方差值，不跟踪统计值。<br /> 如果为None，则根据训练和验证模式自动设置 use_batch_statistics 为True或False。在训练时，use_batch_statistics会 设置为True。在验证时，use_batch_statistics 会自动设置为False。默认值：None |
| | 参数18 | - | data_format | 数据格式可为"NHWC"或"NCHW"。默认值："NCHW" |
| | 参数19 | x | x | - |

### 代码示例1

> 两API功能相同，使用方法相同。

```python
# TensorFlow
import tensorflow as tf

inputx = [[[[1, 2],
           [2, 1]],
          [[3, 4],
           [4, 3]]],
         [[[5, 6],
           [6, 5]],
          [[7, 8],
           [8, 7]]
         ]]
input_tf = tf.constant(inputx, dtype=tf.float32)
output_tf = tf.keras.layers.BatchNormalization(axis=3,momentum=0.1,epsilon=1e-5)
output = output_tf(input_tf,training=False)
print(output.numpy())
# [[[[0.999995  1.99999  ]
#    [1.99999   0.999995 ]]
#
#   [[2.999985  3.99998  ]
#    [3.99998   2.999985 ]]]
#
#
#  [[[4.999975  5.99997  ]
#    [5.99997   4.999975 ]]
#
#   [[6.9999647 7.99996  ]
#    [7.99996   6.9999647]]]]

# MindSpore
from mindspore import Tensor,nn
import numpy as np

m = nn.BatchNorm2d(num_features=2,momentum=0.9)
inputx = Tensor(np.array([[[[1, 2],[2,1]],
                          [[3, 4],[4,3]]],
                          [[[5, 6],[6,5]],[[7,8],[8,7]]]]).astype(np.float32))
output = m(inputx)
print(output)
# [[[[0.99999493 1.9999899 ]
#    [1.9999899  0.99999493]]
#
#   [[2.9999847  3.9999797 ]
#    [3.9999797  2.9999847 ]]]
#
#
#  [[[4.9999747  5.9999695 ]
#    [5.9999695  4.9999747 ]]
#
#   [[6.9999647  7.9999595 ]
#    [7.9999595  6.9999647 ]]]]
```
