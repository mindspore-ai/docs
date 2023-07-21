# Function Differences with tf.keras.layers.LSTM

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.10/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.10/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/LSTM.md)

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

For more information, see [tf.keras.layers.LSTM](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/layers/LSTM).

## mindspore.nn.LSTM

```python
class mindspore.nn.LSTM(*args, **kwargs)(x, hx, seq_length)
```

For more information, see [mindspore.nn.LSTM](https://mindspore.cn/docs/en/r1.10/api_python/nn/mindspore.nn.LSTM.html).

## Differences

| diff parameters  | MindSpore                                                                                                                                                                                    | TensorFlow                                                                                                                                                                                                                                                                                    |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input_size       | input size at initialization.                                                                                                                                                                | no parameter, the input size is automatically determined.                                                                                                                                                                                                                                     |
| hidden_size      | initialize hidden state size.                                                                                                                                                                | corresponds to units.                                                                                                                                                                                                                                                                         |
| num_layers       | the number of network layers, default: 1.                                                                                                                                                    | no parameter, default: 1, build network with more than 1 layer by yourself.                                                                                                                                                                                                                   |
| batch_first      | if the first dimension of input x is batch_size, default: False.                                                                                                                             | no parameter, the first dimension defaults to batch_size.                                                                                                                                                                                                                                     |
| bidirectional    | bidirectional = True, set a bidirectional LSTM.                                                                                                                                              | set bidirectional LSTM, refer to [tf.keras.layers.Bidirectional] (https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/layers/Bidirectional).                                                                                                                                   |
| output、hn、cn     | a sequence by default, output tensor such as (seq_len, batch_size, num_directions * hidden_size), hn、cn is the last state, shape like(num_directions * num_layers, batch_size, hidden_size). | if set return_sequences = True，return a sequence，if set return_state = True, return last state, default：False. When both of these are set to True, the arrangement is different from MindSpore, output such as (batch_size, seq_len, hidden_size), hn、cn shape like(batch_size, hidden_size). |

## Code Example

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
