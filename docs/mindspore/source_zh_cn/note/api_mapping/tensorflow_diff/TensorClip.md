# 比较与tf.clip_by_value的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/TensorClip.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## tf.clip_by_value

```python
tf.clip_by_value(
    t, clip_value_min, clip_value_max, name=None
)
```

更多内容详见[tf.clip_by_value](https://tensorflow.google.cn/versions/r1.15/api_docs/python/tf/clip_by_value)。

## mindspore.Tensor.clip

```python
mindspore.Tensor.clip(xmin, xmax, dtype=None)
```

更多内容详见[mindspore.Tensor.clip](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/mindspore/Tensor/mindspore.Tensor.clip.html#mindspore.Tensor.clip)。

## 使用方式

主要功能一致。`tf.clip_by_value`在`t`为`int32`，`clip_value_min`或`clip_value_max`为`float32`类型时，抛出类型错误，`mindspore.Tensor.clip`没有此限制。

## 代码示例

```python
import mindspore as ms

x = ms.Tensor([1, 2, 3, -4, 0, 3, 2, 0]).astype(ms.int32)
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
