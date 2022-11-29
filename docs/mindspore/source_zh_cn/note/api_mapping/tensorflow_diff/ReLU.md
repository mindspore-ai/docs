# 比较与tf.nn.relu的功能差异

## tf.nn.relu

```text
tf.nn.relu(features, name=None) -> Tensor
```

更多内容详见 [tf.nn.relu](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/relu)。

## mindspore.nn.ReLU

```text
class mindspore.nn.ReLU()(x) -> Tensor
```

更多内容详见 [mindspore.nn.ReLU](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.ReLU.html)。

## 差异对比

Tensorflow：ReLU激活函数。

MindSpore: MindSpore此算子实现功能与TensorFlow一致，仅参数名不同。

| 分类 | 子类 | Tensorflow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | features | x | 输入Tensor |
| | 参数2 | name | - | 不涉及 |

### 代码示例1

> 两API实现功能一致，用法相同。

```python
# Tensorflow
import tensorflow as tf

x = tf.constant([[-1.0, 2.2], [3.3, -4.0]])
out = tf.nn.relu(x).numpy()
print(out)
# [[0.  2.2]
#  [3.3 0. ]]

# MindSpore
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np

x = Tensor(np.array([[-1.0, 2.2], [3.3, -4.0]]), mindspore.float16)
relu = nn.ReLU()
output = relu(x)
print(output)
# [[0.  2.2]
#  [3.3 0. ]]
```