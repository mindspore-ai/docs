# 比较与tf.keras.layers.LSTM的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.9/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/LSTM.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source.png"></a>

## tf.keras.layers.LSTM

```python
class tf.keras.layers.LSTM(
    units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True,
    kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
    bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None,
    recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False,
    return_state=False, go_backwards=False, stateful=False, unroll=False, **kwargs
)
```

更多内容详见[tf.keras.layers.LSTM](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/layers/LSTM)。

## mindspore.nn.LSTM

```python
class mindspore.nn.LSTM(*args, **kwargs)
```

更多内容详见[mindspore.nn.LSTM](https://mindspore.cn/docs/zh-CN/r1.9/api_python/nn/mindspore.nn.LSTM.html)。

## 使用方式

| diff parameters    | MindSpore                                                                                                                                            | TensorFlow                                                                                                                                                                               |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input_size         | 初始化时输入大小。                                                                                                                                            | 无此参数，自动判断输入大小。                                                                                                                                                                           |
| hidden_size        | 初始化隐藏状态大小。                                                                                                                                           | 对应units。                                                                                                                                                                                 |
| num_layers         | 网络层数，默认：1。                                                                                                                                           | 无此参数，默认：1，需要自己构建1层以上的网络。                                                                                                                                                                 |
| batch_first        | 输入x的第一维度是否为batch_size，默认：False 。                                                                                                                     | 无此参数， 默认输入第一个维度为batch_size。                                                                                                                                                              |
| bidirectional      | bidirectional = True，设置双向LSTM。                                                                                                                       | 设置双向LSTM，需要使用 [tf.keras.layers.Bidirectional] (https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/layers/Bidirectional)。                                                 |
| output、hn、cn       | 默认输出一个序列，output形状为(seq_len, batch_size, num_directions * hidden_size)的Tensor，hn、cn是最后一个状态，都形如(num_directions * num_layers, batch_size, hidden_size)。 | 设置参数return_sequences = True，输出返回序列，return_state = True，返回最后一个状态，默认都是：False。当这两个全部设置为True，排布方式与MindSpore不同， output形如(batch_size, seq_len, hidden_size)，hn、cn都形如(batch_size, hidden_size)。 |

## 代码示例

```python
import numpy as np
import tensorflow as tf
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor

# The following implements LSTM with MindSpore.

# bidirectional LSTM
net = nn.LSTM(10, 16, 1,  bidirectional=True)
x = Tensor(np.ones([5, 3, 10]).astype(np.float32))
h0 = Tensor(np.ones([1 * 1, 3, 16]).astype(np.float32))
c0 = Tensor(np.ones([1 * 1, 3, 16]).astype(np.float32))
output, (hn, cn) = net(x, (h0, c0))
print(output.shape)
# (5, 3, 32)
print(hn, cn)
# (2, 3, 16), (2, 3, 16)


# The following implements LSTM with TensorFlow.

# default LSTM
inputs = tf.random.normal([3, 5, 10])
lstm = tf.keras.layers.LSTM(16, return_sequences=True, return_state=True)
output_tf, hn_tf, cn_tf = lstm(inputs)
print(output_tf.shape)
#(3, 5, 16)
print(hn_tf, cn_tf)
# (3, 16), (3, 16)

# bidirectional LSTM
bidirectional_tf = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM())

```
