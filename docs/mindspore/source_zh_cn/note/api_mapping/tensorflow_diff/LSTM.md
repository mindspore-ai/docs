# 比较与tf.keras.layers.LSTM的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/LSTM.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.keras.layers.LSTM

```text
class tf.keras.layers.LSTM(
    units, activation='tanh', recurrent_activation='sigmoid',
    use_bias=True, kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    bias_initializer='zeros', unit_forget_bias=True,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
    activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
    bias_constraint=None, dropout=0.0, recurrent_dropout=0.0,
    return_sequences=False, return_state=False, go_backwards=False, stateful=False,
    time_major=False, unroll=False)(inputs, mask, training, initial_state) -> Tensor
```

更多内容详见[tf.keras.layers.LSTM](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/keras/layers/LSTM)。

## mindspore.nn.LSTM

```text
class mindspore.nn.LSTM(
    input_size,
    hidden_size,
    num_layers=1,
    has_bias=True,
    batch_first=False,
    dropout=0.0,
    bidirectional=False)(x, hx, seq_length=None) -> Tensor
```

更多内容详见[mindspore.nn.LSTM](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.LSTM.html)。

## 差异对比

TensorFlow：当设定好参数return_sequences和return_state时，可以根据输入序列计算输出序列和最终状态。

MindSpore：MindSpore可以根据输入序列和给定的初始状态计算输出序列和最终状态，并且可以实现多层和双向的LSTM网络。但不可以像TensorFlow一样指定计算过程中的一些函数（如激活函数，正则化函数，约束函数等），并且TensorFlow的该API只可以实现单向一层的LSTM网络，因此会导致俩API最后的状态张量形状不同。

| 分类 | 子类   | TensorFlow            | MindSpore     | 差异                                                                                                                                                   |
| --- |------|:----------------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
|参数 | 参数1  | units                 | hidden_size   | 功能一致，参数名不同                                                                                                                                           |
| | 参数2  | activation            | -             | 指定要使用的激活函数，默认值:tanh。MindSpore无此参数，但在计算过程中默认使用同样的激活函数                                                                                   |
| | 参数3  | recurrent_activation  | -             | 指定递归步骤中使用的激活函数，默认值:sigmoid。MindSpore无此参数，但在计算过程中默认使用同样的激活函数                                                                            |
| | 参数4  | use_bias              | has_bias      | 功能一致，参数名不同                                                                                                                                          |
| | 参数5  | kernel_initializer    | -             | 初始化kernel的权重矩阵，用于输入的线性变换。默认值：glorot_uniform。MindSpore无此参数                                                                              |
| | 参数6  | recurrent_initializer | -             | 初始化recurrent_kernel的权重矩阵，用于递归状态的线性变换。默认值：orthogonal。MindSpore无此参数                                                                     |
| | 参数7  | bias_initializer      | -             | 初始化偏置向量，默认值：zeros。MindSpore无此参数                                                                                                        |
| | 参数8  | unit_forget_bias      | -             | 选择是否在初始化时将遗忘门的偏置加1，默认值：True。MindSpore无此参数                                                                                              |
| | 参数9  | kernel_regularizer    | -             | 应用于kernel权重矩阵的正则化函数，默认值：None。MindSpore无此参数                                                                                             |
| | 参数10 | recurrent_regularizer | -             | 应用于recurrent_kernel权重矩阵的正则化函数，默认值：None。MindSpore无此参数                                                                                  |
| | 参数11 | bias_regularizer      | -             | 应用于偏置向量的正则化函数，默认值：None。MindSpore无此参数                                                                                                  |
| | 参数12 | activity_regularizer  | -             | 应用于激活后的层输出的正则化函数，默认值：None。MindSpore无此参数                                                                                               |
| | 参数13 | kernel_constraint     | -             | 应用于kernel权重矩阵的约束函数，默认值：None。MindSpore无此参数                                                                                              |
| | 参数14 | recurrent_constraint  | -             | 应用于recurrent_kernel权重矩阵的约束函数，默认值：None。MindSpore无此参数                                                                                    |
| | 参数15 | bias_constraint       | -             | 应用于偏置向量的约束函数，默认值：None。MindSpore无此参数                                                                                                   |
| | 参数16 | dropout               | dropout       | -  |
| | 参数17 | recurrent_dropout     | -       | 递变状态下使用的丢弃概率，MindSpore使用dropout|
| | 参数18 | return_sequences      | -             | 是否返回在输出序列或完整序列中的最后一次输出，默认值：False。MindSpore无此参数，但默认为True                                                                                   |
| | 参数19 | return_state          | -             | 是否返回最后的状态，默认值：False。MindSpore无此参数，但默认为True                                                                                                |
| | 参数20 | go_backwards          | -             | 是否反向处理输入序列并返回反向序列，默认值：False。MindSpore无此参数                                                                                              |
| | 参数21 | stateful              | -             | 是否将批次中索引i处每个样本的最后状态用作下一批次中索引i处样本的初始状态，默认值：False。MindSpore无此参数                                                                         |
| | 参数22 | time_major            | -             | 选择输入和输出张量的形状格式。如果为True，输入和输出将为[timesteps, batch, feature]，而在False的情况下，将为[batch, timesteps, feature]。默认值：False。MindSpore无此参数，但默认两种形状均可以 |
| | 参数23 | unroll                | -             | 如果为True，网络将被展开，否则将使用符号循环，默认值：False。MindSpore无此参数                                                                                      |
| | 参数24 | inputs                | x             | 功能一致，参数名不同                                                                                                                                          |
| | 参数25 | mask                  | -             | 形状为[batch，timesteps]的二进制张量，指示是否应屏蔽给定的时间步长(可选，默认为None)。单个True条目指示应该利用相应的时间步长，而False条目指示应该忽略相应的时间步长。MindSpore无此参数                        |
| | 参数26 | training              | -             | Python布尔值，指示layer应在训练模式还是推理模式下运行。调用cell时，该参数被传递给单元格。这仅在使用dropout或recurrent_dropout时才有意义(可选，默认为None)。MindSpore无此参数                      |
| | 参数27 | initial_state         | hx            | 要传递给cell第一次调用的初始状态张量列表(可选，默认为None，这将导致创建零填充的初始状态张量)。MindSpore中作用是给定初始状态张量                                                               |
| | 参数28 | -                     | input_size    | 自动判断输入大小，TensorFlow无此参数|
| | 参数29 | -                     | num_layers    | 设置网络层数，默认值：1。TensorFlow无此参数|
| | 参数30 | -                     | batch_first   | 默认输入的第一个维度为batch_size，TensorFlow无此参数|
| | 参数31 | -                     | bidirectional | 功能为设置双向LSTM，TensorFlow无此参数|
| | 参数32 | -                     | seq_length    | 指定输入batch的序列长度，TensorFlow无此参数|

### 代码示例

> TensorFlow的该API一般默认初始状态张量为零填充张量，因此我们可以将MindSpore的输入状态张量设置为零张量。另外TensorFlow的该API只可以实现一层单向的LSTM网络，并且输出状态的形状为[batch_size, hidden_size]，而MindSpore的输出状态的形状为[num_directions * num_layers, batch_size, hidden_size]，因此，我们可以将MindSpore该API的参数bidirectional取默认值False，使得num_directions为1，将参数num_layers也取默认值1，使得MindSpore输出状态张量形状的第一维为1，然后再搭配mindspore.ops.Squeeze去掉第一维，就可以得到和TensorFlow的API相同的结果，并且实现相同的功能。

```python
# TensorFlow
import tensorflow as tf
import numpy as np

inputs = np.ones([3, 5, 10])
lstm = tf.keras.layers.LSTM(16, return_sequences=True, return_state=True)
whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
print(whole_seq_output.shape)
# (3, 5, 16)
print(final_memory_state.shape)
# (3, 16)
print(final_carry_state.shape)
# (3, 16)

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

net = mindspore.nn.LSTM(10, 16, 1, has_bias=True, batch_first=True, bidirectional=False)
x = Tensor(np.ones([3, 5, 10]).astype(np.float32))
h0 = Tensor(np.zeros([1 * 1, 3, 16]).astype(np.float32))
c0 = Tensor(np.zeros([1 * 1, 3, 16]).astype(np.float32))
output, (hn, cn) = net(x, (h0, c0))
print(output.shape)
# (3, 5, 16)
squeeze = mindspore.ops.Squeeze(0)
hn_ = squeeze(hn)
print(hn_.shape)
# (3, 16)
cn_ = squeeze(cn)
print(cn_.shape)
# (3, 16)
```
