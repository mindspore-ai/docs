# Function Differences with tf.nn.ctc_greedy_decoder

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/ctc_greedy_decoder.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.nn.ctc_greedy_decoder

```text
tf.nn.ctc_greedy_decoder(
    inputs,
    sequence_length,
    merge_repeated=True,
    blank_index=None
)(decoded, neg_sum_logits) -> Tuple
```

For more information, see [tf.nn.ctc_greedy_decoder](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/ctc_greedy_decoder).

## mindspore.ops.ctc_greedy_decoder

```text
mindspore.ops.ctc_greedy_decoder(
    inputs,
    sequence_length,
    merge_repeated=True
)(decoded_indices, decoded_values, decoded_shape, log_probability) -> Tuple
```

For more information, see [mindspore.ops.ctc_greedy_decoder](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.ctc_greedy_decoder.html).

## Differences

TensorFlow: Perform greedy decoding of the given logits in the input, and return a tuples consisting of SparseTesnor and float matrices where SparseTesnor contains 3 dense tensors: indices, values, and sense_shape.

MindSpore: MindSpore API implements the same functions as TensorFlow, with different parameter names and different returned parameters.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | inputs | inputs        | -           |
|  | Parameter 2 | sequence_length       | sequence_length         | - |
|  | Parameter 3 | merge_repeated       | merge_repeated         | - |
|  | Parameter 4 | blank_index       | -         | Define the class index used for blank labels. The default value for Tensorflow is None, and the operator is used in the same way as MindSpore. |
|Returned Parameters| Parameter 5 | decoded       | decoded_indices, decoded_values, decoded_shape          | TensorFlow decoded is SparseTesnor, which contains three dense tensor, indices, values, sense_shape, corresponding to three outputs of MindSpore decoded_indices, decoded_values, and decoded_shape  |
|  | Parameter 6 | neg_sum_logits       | log_probability          | Same function, different parameter names |

### Code Example 1

The outputs of MindSpore and TensorFlow are consistent.

```python
# TensorFlow
import tensorflow as tf
import numpy as np

inputs = tf.constant(np.array([[[0.6, 0.4, 0.2], [0.8, 0.6, 0.3]],[[0.0, 0.6, 0.0], [0.5, 0.4, 0.5]]]), dtype=tf.float32)
seq_lens = tf.constant([2, 2])
output = tf.nn.ctc_greedy_decoder(inputs, seq_lens)

print(output[0][0])
# SparseTensor(indices=tf.Tensor(
# [[0 0]
#  [0 1]
#  [1 0]], shape=(3, 2), dtype=int64), values=tf.Tensor([0 1 0], shape=(3,), dtype=int64), dense_shape=tf.Tensor([2 2], shape=(2,), dtype=int64))

print(output[1].numpy())
# [[-1.2]
#  [-1.3]]

# MindSpore
import mindspore
import numpy as np
from mindspore.ops.function import nn_func as ops
from mindspore import Tensor

inputs = Tensor(np.array([[[0.6, 0.4, 0.2], [0.8, 0.6, 0.3]],

                          [[0.0, 0.6, 0.0], [0.5, 0.4, 0.5]]]), mindspore.float32)
sequence_length = Tensor(np.array([2, 2]), mindspore.int32)
decoded_indices, decoded_values, decoded_shape, log_probability = ops.ctc_greedy_decoder(inputs, sequence_length)
print(decoded_indices)
# [[0 0]
#  [0 1]
#  [1 0]]
print(decoded_values)
# [0 1 0]
print(decoded_shape)
# [2 2]
print(log_probability)
# [[-1.2]
#  [-1.3]]
```
