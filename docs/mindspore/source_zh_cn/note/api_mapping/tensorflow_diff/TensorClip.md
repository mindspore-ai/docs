# 比较与tf.clip_by_value的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/TensorClip.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

## tf.clip_by_value

```python
tf.clip_by_value(
    t, clip_value_min, clip_value_max, name=None
)
```

更多内容详见[tf.clip_by_value](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/clip_by_value)。

## mindspore.Tensor.clip

```python
mindspore.Tensor.clip(xmin, xmax, dtype=None)
```

更多内容详见[mindspore.Tensor.clip](https://www.mindspore.cn/docs/en/r1.7/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor.clip)。

## 使用方式

主要功能一致。`tf.clip_by_value`在`t`为`int32`，`clip_value_min`或`clip_value_max`为`float32`类型时，抛出类型错误，`mindspore.Tensor.clip`没有此限制。

## 代码示例

```python
import mindspore
from mindspore import Tensor

x = Tensor([1, 2, 3, -4, 0, 3, 2, 0]).astype(mindspore.int32)
print(x.clip(0, 2))
# [1 2 2 0 0 2 2 0]
print(x.clip(0., 2.))
# [1 2 2 0 0 2 2 0]
print(x.clip(Tensor([1, 1, 1, 1, 1, 1, 1, 1]), 2))
# [1 2 2 1 1 2 2 1]

import tensorflow as tf
tf.enable_eager_execution()

A = tf.constant([1, 2, 3, -4, 0, 3, 2, 0])
B = tf.clip_by_value(A, clip_value_min=0, clip_value_max=2)
print(B.numpy())
# [1 2 2 0 0 2 2 0]
C = tf.clip_by_value(A, clip_value_min=0., clip_value_max=2.)
# throws `TypeError`
D = tf.clip_by_value(A, [1, 1, 1, 1, 1, 1, 1, 1], 2)
print(D.numpy())
# [1 2 2 1 1 2 2 1]
```
