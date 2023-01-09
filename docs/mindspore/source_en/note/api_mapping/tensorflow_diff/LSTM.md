# Function Differences with tf.keras.layers.LSTM

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/LSTM.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see [tf.keras.layers.LSTM](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/keras/layers/LSTM).

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

For more information, see [mindspore.nn.LSTM](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.LSTM.html).

## Differences

TensorFlow: When the parameters return_sequences and return_state are set, the output sequence and final state can be calculated based on the input sequence.

MindSpore: MindSpore can compute output sequences and final states based on input sequences and given initial states, and can implement multi-layer and bi-directional LSTM networks. However, it is not possible to specify some functions (such as activation function, regularization function, constraint function) in the computation process like TensorFlow, and the API of TensorFlow can only implement one-way one-layer LSTM networks, so it will lead to different shapes of the final state tensor between the two APIs.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1  | units                 | hidden_size   | Same function, different parameter names     |
| | Parameter 2  | activation            | -             | Specify the activation function to be used. Default value: tanh. MindSpore does not have this parameter, but the same activation function is used by default during the calculation      |
| | Parameter 3  | recurrent_activation  | -             | Specify the activation function used in the recursion step. Default value: sigmoid. MindSpore does not have this parameter, but the same activation function is used by default during the calculation   |
| | Parameter 4  | use_bias              | has_bias      | Same function, different parameter names    |
| | Parameter 5  | kernel_initializer    | -             | Initialize the kernel weight matrix for the linear transformation of the input. Default value: glorot_uniform. MindSpore does not have this parameter.   |
| | Parameter 6  | recurrent_initializer | -             | Initialize the weight matrix of recurrent_kernel for linear transformation of recursive states. Default value: orthogonal. MindSpore does not have this parameter    |
| | Parameter 7  | bias_initializer      | -             | Initialize the bias vector. Default value: zeros. MindSpore does not have this parameter.           |
| | Parameter 8  | unit_forget_bias      | -             | Select whether to add 1 to the offset of the forget gate at initialization. Default value: True. MindSpore does not have this parameter.         |
| | Parameter 9  | kernel_regularizer    | -             | The regularization function applied to the kernel weight matrix. Default value: None. MindSpore does not have this parameter.     |
| | Parameter 10 | recurrent_regularizer | -             | The regularization function applied to the recurrent_kernel weight matrix. Default value: None. MindSpore does not have this parameter.      |
| | Parameter 11 | bias_regularizer      | -             | The regularization function applied to the bias vector. Default value: None. MindSpore does not have this parameter.         |
| | Parameter 12 | activity_regularizer  | -             | The regularization function applied to the output of the activated layer. Default value: None. MindSpore does not have this parameter      |
| | Parameter 13 | kernel_constraint     | -             | Constraint function applied to the kernel weight matrix. Default value: None. MindSpore does not have this parameter       |
| | Parameter 14 | recurrent_constraint  | -             | Constraint function applied to the recurrent_kernel weight matrix. Default value: None. MindSpore does not have this parameter   |
| | Parameter 15 | bias_constraint       | -             | Constraint function applied to the weight vector. Default value: None. MindSpore does not have this parameter     |
| | Parameter 16 | dropout               | dropout       | -  |
| | Parameter 17 | recurrent_dropout     | -       | The dropout probability used in the recursive state. MindSpore uses dropout |
| | Parameter 18 | return_sequences      | -             | Whether to return the last output in the output sequence or the complete sequence. Default value: False. MindSpore does not have this parameter, but defaults to True.    |
| | Parameter 19 | return_state          | -             | Whether to return the last state. Default value: False. MindSpore does not have this parameter, but defaults to True |
| | Parameter 20 | go_backwards          | -             | Whether to reverse the input sequence and return the reverse sequence. Default value: False. MindSpore does not have this parameter. |
| | Parameter 21 | stateful              | -             | Whether to use the last state of each sample at index i in the batch as the initial state of the samples at index i in the next batch. Default value: False. MindSpore does not have this parameter.    |
| | Parameter 22 | time_major     | -             | Selects the shape format of the input and output tensor. If True, the input and output will be [timesteps, batch, feature], while in the case of False, it will be [batch, timesteps, feature]. Default value: False. MindSpore does not have this parameter, but by default both shapes are possible |
| | Parameter 23 | unroll                | -             | If True, the network will be expanded, otherwise a symbolic loop will be used. Default value: False. MindSpore does not have this parameter.    |
| | Parameter 24 | inputs   | x        | Same function, different parameter names    |
| | Parameter 25 | mask        | -             | A binary tensor of the shape [batch, timesteps] indicating whether the given time step should be masked or not (optional, default is None). A single True entry indicates that the corresponding time step should be utilized, while a False entry indicates that the corresponding time step should be ignored. MindSpore does not have this parameter      |
| | Parameter 26 | training              | -             | Python bool indicating whether layer should be run in training mode or inference mode. This parameter is passed to the cell when the cell is called. This is only relevant when using dropout or recurrent_dropout (optional, default is None). MindSpore does not have this parameter                      |
| | Parameter 27 | initial_state         | hx            | The initial state tensor list to be passed to the cell for the first call (optional, default is None, which will result in the creation of a zero-padded initial state tensor). The role in MindSpore is to give the initial state tensor.    |
| | Parameter 28 | -                     | input_size    | Automatically determine the input size. TensorFlow does not have this parameter|
| | Parameter 29 | -                     | num_layers    | Set the number of network layers. Default value: 1. TensorFlow does not have this parameter|
| | Parameter 30 | -                     | batch_first   | The first dimension of the default input is batch_size, and TensorFlow does not have this parameter |
| | Parameter 31 | -                     | bidirectional | The function is to set the bi-directional LSTM, and TensorFlow does not have this parameter |
| | Parameter 32 | -                     | seq_length    | Specify the sequence length of the input batch. TensorFlow does not have this parameter |

### Code Example

> This API of TensorFlow generally defaults to a zero-padding tensor for the initial state tensor, so we can set MindSpore input state tensor to a zero tensor. In addition, TensorFlow API can only implement one layer of one-way LSTM network, and the shape of the output state is [batch_size, hidden_size], while the shape of MindSpore output state is [num_directions * num_layers, batch_size, hidden_size]. Therefore, we can take the default value of False for the parameter bidirectional of the MindSpore API, so that num_directions is 1. By taking the default value of 1 for the parameter num_layers as well, making the first dimension of MindSpore output state tensor shape 1, and then removing the first dimension with mindspore.ops.Squeeze, we can get the same result as TensorFlow API and achieve the same function.

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
