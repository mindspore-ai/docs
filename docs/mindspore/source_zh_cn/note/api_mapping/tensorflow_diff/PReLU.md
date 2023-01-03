# 比较与tf.keras.layers.PReLU的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/PReLU.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.keras.layers.PReLU

```text
tf.keras.layers.PReLU(
  alpha_initializer='zeros',
  alpha_regularizer=None,
  alpha_constraint=None,
  shared_axes=None
)(x) -> Tensor
```

更多内容详见[tf.keras.layers.PReLU](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/keras/layers/PReLU)。

## mindspore.nn.PReLU

```text
class mindspore.nn.PReLU(channel=1, w=0.25)(x) -> Tensor
```

更多内容详见[mindspore.nn.PReLU](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.PReLU.html)。

## 差异对比

TensorFlow：PReLU激活函数。

MindSpore：MindSpore此接口功能与TensorFlow基本一致。

| 分类 | 子类 | TensorFlow | MindSpore | 差异 |
| --- | --- | :-- | --- |---|
|参数 | 参数1 | alpha_initializer | w | 权重的初始化函数，参数功能一致，默认值不同，参数名不同 |
| | 参数2 | alpha_regularizer | - | 权重的正则化器。MindSpore无此参数 |
| | 参数3 | alpha_constraint | - | 权重的约束。MindSpore无此参数 |
| | 参数4 | shared_axes | - | 共享激活函数的可学习参数的轴。MindSpore无此参数 |
| | 参数5 | x | x | - |
| | 参数6 | - | channel | 输入张量的通道数，默认值为1。TensorFlow无此参数 |

### 代码示例1

> TensorFlow的alpha_initializer参数与MindSpore的参数功能一致，默认值不同，参数名不同，TensorFlow默认alpha为0.0，故使用MindSpore只需将w设置为0.0即可实现相同功能。

```python
# TensorFlow
import tensorflow as tf
from keras.layers import PReLU
import numpy as np

x = tf.constant([[-1.0, 2.2], [3.3, -4.0]])
m = PReLU()
out = m(x)
print(out.numpy())
# [[0.  2.2]
#  [3.3 0. ]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

x = Tensor(np.array([[-1.0, 2.2], [3.3, -4.0]]), mindspore.float32)
prelu = nn.PReLU(w=0.0)
output = prelu(x)
print(output)
# [[0.   2.2]
#  [ 3.3 0. ]]
```

### 代码示例2

> TensorFlow的alpha_initializer参数可以通过初始化函数改变alpha值，MindSpore只需将w设置为对应值即可实现相同功能。

```python
# TensorFlow
import tensorflow as tf
from keras.layers import PReLU
import numpy as np
x = tf.constant([[-1.0, 2.2], [3.3, -4.0]])
m = PReLU(alpha_initializer=tf.constant_initializer(0.5))
out = m(x)
print(out.numpy())
# [[-0.5  2.2]
#  [ 3.3 -2. ]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np
x = Tensor(np.array([[-1.0, 2.2], [3.3, -4.0]]), mindspore.float32)
prelu = nn.PReLU(w=0.5)
output = prelu(x)
print(output)
# [[-0.5  2.2]
#  [ 3.3 -2. ]]
```
