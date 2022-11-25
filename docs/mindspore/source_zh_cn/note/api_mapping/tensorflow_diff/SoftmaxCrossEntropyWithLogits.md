# 比较与tf.nn.softmax_cross_entropy_with_logits的功能差异

## tf.nn.softmax_cross_entropy_with_logit

```text
tf.nn.softmax_cross_entropy_with_logits(
    labels, logits, axis=-1
) -> Tensor
```

更多内容详见[tf.nn.softmax_cross_entropy_with_logits](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/softmax_cross_entropy_with_logits)。

## mindspore.nn.SoftmaxCrossEntropyWithLogits

```text
class mindspore.nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction='none')(logits, labels) -> Tensor
```

更多内容详见[mindspore.nn.SoftmaxCrossEntropyWithLogits](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.SoftmaxCrossEntropyWithLogits.html)。

## 差异对比

Tensorflow：tensorflow中该算子是函数式，可以直接调用算子接受`logits`和`labels`输入，并返回输出结果。参数`labels`和`logits`的shape需一致。可指定`axis`参数指定‘类’所在的维度。

Mindspore：Mindspore中该算子需要实例化，实例化时可接受`sparse`参数表示输入的`labels`是否是稀疏表示，默认为`False`；可接受`reduction`参数表示输入结果的规约方式，取值为`mean`,`sum`或`none`,默认为`none`。

| 分类 | 子类  | Tensorflow | MindSpore | 差异                                                         |
| ---- | ----- | ---------- | --------- | ------------------------------------------------------------ |
| 参数 | 参数1 | logits     | logits    | mindspore在实例化函数中接收此参数，功能无差异。              |
|      | 参数2 | labels     | labels    | mindspore在实例化函数中接收此参数，功能无差异。              |
|      | 参数3 | axis       | -         | tensorflow可指定`axis`参数指定‘类’所在的维度，mindspore无此参数。如`axis=-1`表示最后一个维度作为‘类’的维度。 |
|      | 参数4 | name       | -         | mindspore无此参数。                                          |
|      | 参数5 | -          | sparse    | mindspore实例化时可以接受`sparse`指定输入的`labels`是否为稀疏表示。 |
|      | 参数6 | -          | reduction | mindspore可对输出结果进行规约。                              |

## 差异分析与示例

### 代码示例1

> tensorflow该算子是函数式的，直接接受输入。mindspore中需要先实例化。

```python
# tensorflow
import tensorflow as tf
from tensorflow import nn
logits = tf.constant([[4.0, 2.0, 1.0], [0.0, 5.0, 1.0]])
labels = tf.constant([[1.0, 0.0, 0.0], [0.0, 0.8, 0.2]])

out = nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
print(out.numpy())
# [0.16984604 0.82474494]

# mindspore
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

> tensorflow中可接受`axis`参数指定'类'所在维度。Mindspore默认最后一维，因为接受的`logits`的`shape`为`[batch_size, num_classes]`，mindspore可以通过交换`axis`实现相同的功能。

```python
# tensorflow
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

