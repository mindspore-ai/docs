# Function Differences with tf.nn.softmax_cross_entropy_with_logits

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/SoftmaxCrossEntropyWithLogits.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.nn.softmax_cross_entropy_with_logits

```text
tf.nn.softmax_cross_entropy_with_logits(labels, logits, axis=-1, name=None) -> Tensor
```

For more information, see [tf.nn.softmax_cross_entropy_with_logits](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/softmax_cross_entropy_with_logits).

## mindspore.nn.SoftmaxCrossEntropyWithLogits

```text
class mindspore.nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction='none')(logits, labels) -> Tensor
```

For more information, see [mindspore.nn.SoftmaxCrossEntropyWithLogits](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.SoftmaxCrossEntropyWithLogits.html).

## Differences

TensorFlow: The operator is functional in TensorFlow and can be called directly to accept `logits` and `labels` inputs and return the output. The parameters `labels` and `logits` need to have the same shape, and the `axis` parameter can be specified to specify the dimension in which the 'class' is located.

MindSpore: The operator needs to be instantiated in MindSpore, and the `sparse` parameter can be accepted when instantiating to indicate whether the input `labels` are sparse. The default is `False`. The `reduction` parameter can be accepted to indicate the way the input result is statuted, taking the values `mean`, `sum` or `none`. The default is `none`.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|  Parameters    | Parameter 1 | labels     | labels    | MindSpore receives this parameter in the instantiation function, with consistent function             |
|      | Parameter 2 | logits     | logits    | MindSpore receives this parameter in the instantiation function, with consistent function              |
|      | Parameter 3 | axis       | -         | TensorFlow `axis` parameter specifies the dimension in which the 'class' is located, e.g. `axis=-1` means that the last dimension is the dimension of the 'class'. MindSpore does not have this parameter |
|      | Parameter 4 | name       | -         | Not involved                                     |
|      | Parameter 5 | -          | sparse    | MindSpore can accept `sparse` to specify whether the input `labels` are sparse during instantiation. TensorFlow does not have this parameter |
|      | Parameter 6 | -          | reduction | MindSpore can statute the output results, and TensorFlow does not have this parameter      |

### Code Example 1

> The two APIs implement the same function, but the TensorFlow operator is functional and accepts input directly, which needs to be instantiated first in MindSpore.

```python
# TensorFlow
import tensorflow as tf
from tensorflow import nn

logits = tf.constant([[4.0, 2.0, 1.0], [0.0, 5.0, 1.0]])
labels = tf.constant([[1.0, 0.0, 0.0], [0.0, 0.8, 0.2]])

out = nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
print(out.numpy())
# [0.16984604 0.82474494]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor, nn

logits = Tensor(np.array([[4.0, 2.0, 1.0], [0.0, 5.0, 1.0]]), mindspore.float32)
labels = Tensor(np.array([[1.0, 0.0, 0.0], [0.0, 0.8, 0.2]]), mindspore.float32)

loss = nn.SoftmaxCrossEntropyWithLogits()
out = loss(logits, labels)
print(out)
# [0.16984606 0.82474494]
```

### Code Example 2

> The `axis` parameter can be accepted in TensorFlow to specify the dimension in which the 'class' is located. MindSpore defaults to the last dimension because the accepted `shape` of `logits` is `[batch_size, num_classes]` and MindSpore can achieve the same function by calling the Transpose operator to swap `axis`.

```python
# TensorFlow
import tensorflow as tf
from tensorflow import nn
logits = tf.constant([[4.0, 0.0],[2.0, 5.0],[1.0, 1.0]])
labels = tf.constant([[1.0, 0.0],[0.0, 0.8],[0.0, 0.2]])
out = nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, axis=0)
print(out.numpy())
# [0.16984604 0.82474494]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor, nn, ops

logits_ = Tensor(np.array([[4.0, 0.0],[2.0, 5.0],[1.0, 1.0]]), mindspore.float32)
labels_ = Tensor(np.array([[1.0, 0.0],[0.0, 0.8],[0.0, 0.2]]), mindspore.float32)
transpose = ops.Transpose()
logits = transpose(logits_, (1,0))
labels = transpose(labels_, (1,0))
loss = nn.SoftmaxCrossEntropyWithLogits()
out = loss(logits, labels)
print(out)
# [0.16984606 0.82474494]
```
