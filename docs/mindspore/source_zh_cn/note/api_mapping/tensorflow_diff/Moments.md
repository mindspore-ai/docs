# 比较与tf.nn.moments的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/Moments.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.nn.moments

```text
tf.nn.moments(x, axes, shift=None, keepdims=False, name=None) -> Tensor
```

更多内容详见[tf.nn.moments](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/moments)。

## mindspore.nn.Moments

```text
class mindspore.nn.Moments(axis=None, keep_dims=None)(x) -> Tensor
```

更多内容详见[mindspore.nn.Moments](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Moments.html)。

## 差异对比

TensorFlow：沿指定轴axes计算输入x的均值和方差。

MindSpore：MindSpore此API实现功能与TensorFlow一致。

| 分类 | 子类  | TensorFlow | MindSpore | 差异                                            |
| --- |-----|------------|-----------|-----------------------------------------------|
|参数 | 参数1 | x          | x         | -                                             |
| | 参数2 | axes        | axis      | 功能一致，参数名不同，TensorFlow中该参数无默认值，MindSpore中该参数默认值为None |
| | 参数3 |   shift            | -         | TensorFlow的该参数在当前实现中未使用，是无用参数。MindSpore无此参数 |
| | 参数4 | keepdims      | keep_dims | 功能一致，参数名不同                                   |
| | 参数5 |   name            | -         | 不涉及 |

### 代码示例1

> 两API用于计算Tensor指定轴的均值和方差，用法相同。

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), dtype='float32')
mean, variance = tf.nn.moments(x, axes=0, keepdims=True)
print(mean.numpy())
# [[[3. 4.]
#   [5. 6.]]]
print(variance.numpy())
# [[[4. 4.]
#   [4. 4.]]]

# MindSpore
import mindspore
from mindspore import Tensor
from mindspore import nn
import numpy as np

x = Tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), mindspore.float32)
net = nn.Moments(axis=0, keep_dims=True)
mean, variance = net(x)
print(mean)
# [[[3. 4.]
#   [5. 6.]]]
print(variance)
# [[[4. 4.]
#   [4. 4.]]]
```

### 代码示例2

> 两API用于计算Tensor所有值的均值和方差，TensorFlow的API需要手动指定所有轴，MindSpore的API默认指定所有轴。

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x = tf.constant(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), dtype='float32')
mean, variance = tf.nn.moments(x, axes=[0, 1, 2])
print(mean.numpy())
# 4.5
print(variance.numpy())
# 5.25

# MindSpore
import mindspore
from mindspore import Tensor
from mindspore import nn
import numpy as np

x = Tensor(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), mindspore.float32)
net = nn.Moments()
mean, variance = net(x)
print(mean)
# 4.5
print(variance)
# 5.25
```
