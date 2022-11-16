# 比较与tf.math.divide的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/Div.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.math.divide

```text
tf.math.divide(x, y, name=None) -> Tensor
```

更多内容详见 [tf.math.divide](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/math/divide)。

## mindspore.ops.Div

```text
class mindspore.ops.Div()(x, y) -> Tensor
```

更多内容详见 [mindspore.ops.Div](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Div.html)。

## 差异对比

TensorFlow：将两个Tensor进行逐元素相除取商的操作。

MindSpore：MindSpore此API实现功能与TensorFlow一致。

| 分类 | 子类 | TensorFlow | MindSpore | 差异                                                                     |
| --- | --- |------------|-----------|------------------------------------------------------------------------|
|参数 | 参数1 | x          | x         | -                                           |
| | 参数2 | y          | y         |  -                                           |

### 代码示例

> 两API实现的功能一致，用法相同。

```python
# TensorFlow
import tensorflow as tf
import numpy

x = tf.constant([[2, 4, 6, 8], [1, 2, 3, 4]])
y = tf.constant([5, 8, 8, 16])
out = tf.math.divide(x, y).numpy()
print(out)
# [[0.4   0.5   0.75  0.5  ]
#  [0.2   0.25  0.375 0.25 ]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

x_ = np.array([[2, 4, 6, 8], [1, 2, 3, 4]])
y_ = np.array([5, 8, 8, 16])
x = Tensor(x_, mindspore.float64)
y = Tensor(y_, mindspore.float64)
div = ops.Div()
output = div(x, y)
print(output)
# [[0.4   0.5   0.75  0.5  ]
#  [0.2   0.25  0.375 0.25 ]]
```
