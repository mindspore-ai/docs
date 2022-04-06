# 比较与tf.math.reduce_sum的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/tensorflow_diff/TensorSum.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.math.reduce_sum

```python
tf.math.reduce_sum(
    input_tensor, axis=None, keepdims=None, name=None, reduction_indices=None,
    keep_dims=None
)
```

更多内容详见[tf.math.reduce_sum](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/reduce_sum)。

## mindspore.Tensor.sum

```python
 mindspore.Tensor.sum(self, axis=None, dtype=None, keepdims=False, initial=None)
```

更多内容详见[mindspore.Tensor.sum](https://mindspore.cn/docs/api/zh-CN/master/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor.sum)。

## 使用方式

两接口基本功能相同，都是计算某个维度上Tensor的和。不同点在于，`mindspore.Tensor.sum`多一个入参`initial`用于设置起始值。

## 代码示例

```python
from mindspore import Tensor
import mindspore

a = Tensor([10, -5], mindspore.float32)
print(a.sum()) # 5.0
print(a.sum(initial=2)) # 7.0

import tensorflow as tf
tf.enable_eager_execution()

b = tf.constant([10, -5])
print(tf.math.reduce_sum(b).numpy()) # 5
```
