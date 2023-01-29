# 比较与tf.nn.softmax_cross_entropy_with_logits的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/SoftmaxCrossEntropyWithLogits.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## tf.nn.softmax_cross_entropy_with_logits

```text
tf.nn.softmax_cross_entropy_with_logits(labels, logits, axis=-1, name=None) -> Tensor
```

更多内容详见[tf.nn.softmax_cross_entropy_with_logits](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/softmax_cross_entropy_with_logits)。

## mindspore.nn.SoftmaxCrossEntropyWithLogits

```text
class mindspore.nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction='none')(logits, labels) -> Tensor
```

更多内容详见[mindspore.nn.SoftmaxCrossEntropyWithLogits](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.SoftmaxCrossEntropyWithLogits.html)。

## 差异对比

TensorFlow：TensorFlow中该算子是函数式，可以直接调用算子接受`logits`和`labels`输入，并返回输出结果。参数`labels`和`logits`的shape需一致，可指定`axis`参数指定‘类’所在的维度。

MindSpore：MindSpore中该算子需要实例化，实例化时可接受`sparse`参数表示输入的`labels`是否是稀疏表示，默认为`False`；可接受`reduction`参数表示输入结果的规约方式，取值为`mean`、`sum`或`none`，默认为`none`。

| 分类 | 子类  | TensorFlow | MindSpore | 差异                                                         |
| ---- | ----- | ---------- | --------- | ------------------------------------------------------------ |
|  参数    | 参数1 | labels     | labels    | MindSpore在实例化函数中接收此参数，功能一致              |
|      | 参数2 | logits     | logits    | MindSpore在实例化函数中接收此参数，功能一致              |
|      | 参数3 | axis       | -         | TensorFlow`axis`参数指定‘类’所在的维度，如`axis=-1`表示最后一个维度作为‘类’的维度，MindSpore无此参数|
|      | 参数4 | name       | -         | 不涉及                                     |
|      | 参数5 | -          | sparse    | MindSpore实例化时可以接受`sparse`指定输入的`labels`是否为稀疏表示，TensorFlow无此参数 |
|      | 参数6 | -          | reduction | MindSpore可对输出结果进行规约，TensorFlow无此参数                              |

### 代码示例1

> 两API实现功能一致，但是TensorFlow该算子是函数式的，直接接受输入。MindSpore中需要先实例化。

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

### 代码示例2

> TensorFlow中可接受`axis`参数指定'类'所在维度。MindSpore默认最后一维，因为接受的`logits`的`shape`为`[batch_size, num_classes]`，MindSpore可以通过调用Transpose算子交换`axis`实现相同的功能。

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
