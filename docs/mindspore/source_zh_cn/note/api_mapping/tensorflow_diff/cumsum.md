# 比较与tf.math.cumsum的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/cumsum.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.math.cumsum

```text
tf.math.cumsum(x, axis=0, exclusive=False, reverse=False, name=None) -> Tensor
```

更多内容详见 [tf.math.cumsum](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/math/cumsum)。

## mindspore.ops.cumsum

```text
mindspore.ops.cumsum(x, axis, dtype=None) -> Tensor
```

更多内容详见 [mindspore.ops.CumSum](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.cumsum.html)。

## 差异对比

TensorFlow：计算输入Tensor在指定轴上的累加和。

MindSpore：MindSpore此API实现功能与TensorFlow基本一致，参数设定上有所差异。

| 分类 | 子类 |TensorFlow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | x | input |功能一致，参数名不同 |
| | 参数2 | axis | axis | - |
| | 参数3 | exclusive | - | MindSpore无此参数 |
| | 参数4 | reverse | - | MindSpore无此参数 |
| | 参数5 | name | - | 不涉及 |
| | 参数6 | - | dtype | mindspore中设置输出数据类型 |

### 代码示例1

> 相同输入tensor，轴为-1，对输入tensor最内层从左到右进行累加，两API实现相同的功能。

```python
# TensorFlow
import tensorflow as tf
a = tf.constant([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]])
y = tf.cumsum(a, -1)
print(y.numpy())
#[[ 3  7 13 23]
# [ 1  7 14 23]
# [ 4  7 15 22]
# [ 1  4 11 20]]

# MindSpore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np
x = Tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]]))
y = ops.cumsum(x, -1)
print(y)
#[[ 3  7 13 23]
# [ 1  7 14 23]
# [ 4  7 15 22]
# [ 1  4 11 20]]
```
