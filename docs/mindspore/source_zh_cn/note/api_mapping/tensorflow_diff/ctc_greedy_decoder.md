# 比较与tf.nn.ctc_greedy_decoder的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.11/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/ctc_greedy_decoder.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.11/resource/_static/logo_source.png"></a>

## tf.nn.ctc_greedy_decoder

```text
tf.nn.ctc_greedy_decoder(
    inputs,
    sequence_length,
    merge_repeated=True,
    blank_index=None
)(decoded, neg_sum_logits) -> Tuple
```

更多内容详见[tf.nn.ctc_greedy_decoder](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/ctc_greedy_decoder)。

## mindspore.ops.ctc_greedy_decoder

```text
mindspore.ops.ctc_greedy_decoder(
    inputs,
    sequence_length,
    merge_repeated=True
)(decoded_indices, decoded_values, decoded_shape, log_probability) -> Tuple
```

更多内容详见[mindspore.ops.ctc_greedy_decoder](https://www.mindspore.cn/docs/zh-CN/r1.11/api_python/ops/mindspore.ops.ctc_greedy_decoder.html)。

## 差异对比

TensorFlow：对输入中给定的logits执行贪婪解码，返回一个由SparseTesnor和float矩阵组成的tuple，其中，SparseTesnor包含3个密集张量，它们为：indices、values、dense_shape。

MindSpore：MindSpore此API实现功能与TensorFlow一致，部分参数名不同，且返回参数不同。

| 分类 | 子类 |TensorFlow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | inputs | inputs        | -           |
|  | 参数2 | sequence_length       | sequence_length         | - |
|  | 参数3 | merge_repeated       | merge_repeated         | - |
|  | 参数4 | blank_index       | -         | 定义用于空白标签的类索引，Tensorflow默认值为None，此时该算子和MindSpore用法一致。 |
|返回参数| 参数5 | decoded       | decoded_indices, decoded_values, decoded_shape          | TensorFlow的decoded为SparseTesnor，包含三个密集张量，为indices、values、dense_shape，对应MindSpore的decoded_indices 、decoded_values 、decoded_shape三个输出。 |
|  | 参数6 | neg_sum_logits       | log_probability          | 功能一致，参数名不同 |

### 代码示例1

MindSpore和TensorFlow输出结果一致。

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
