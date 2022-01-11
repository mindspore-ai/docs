# 比较与tf.nn.softmax_cross_entropy_with_logits的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_zh_cn/api_mapping/tensorflow_diff/SoftmaxCrossEntropyWithLogits.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## tf.nn.softmax_cross_entropy_with_logits

```python
class tf.nn.softmax_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,
    logits=None,
    dim=-1,
    name=None,
    axis=None
)
```

更多内容详见[tf.nn.softmax_cross_entropy_with_logits](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/softmax_cross_entropy_with_logits)。

## mindspore.nn.SoftmaxCrossEntropyWithLogits

```python
class mindspore.nn.SoftmaxCrossEntropyWithLogits(
    sparse=False,
    reduction='none'
)(logits, labels)
```

更多内容详见[mindspore.nn.SoftmaxCrossEntropyWithLogits](https://mindspore.cn/docs/api/zh-CN/r1.6/api_python/nn/mindspore.nn.SoftmaxCrossEntropyWithLogits.html)。

## 使用方式

TensorFlow：labels和logits的shape需一致，未提供reduction参数对loss求mean或sum。

MindSpore：支持labels是稀疏矩阵，且通过reduction参数可对loss求mean或sum。

## 代码示例

```python
# The following implements SoftmaxCrossEntropyWithLogits with MindSpore.
import numpy as np
import tensorflow as tf
import mindspore
import mindspore.nn as nn
from mindspore import Tensor

loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='sum')
logits = Tensor(np.array([[3, 5, 6, 9], [42, 12, 32, 72]]), mindspore.float32)
labels_np = np.array([1, 0]).astype(np.int32)
labels = Tensor(labels_np)
output = loss(logits, labels)
print(output)
# Out：
# 34.068203


# The following implements softmax_cross_entropy_with_logits with TensorFlow.
logits = tf.constant([[3, 5, 6, 9], [42, 12, 32, 72]], dtype=tf.float32)
labels = tf.constant([[0, 1, 0, 0], [1, 0, 0, 0]], dtype=tf.float32)
output = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
ss = tf.Session()
ss.run(output)
# out
# array([ 4.068202, 30.  ], dtype=float32)
```
