# 比较与tf.math.reduce_std的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/tensorflow_diff/TensorStd.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.math.reduce_std

```python
tf.math.reduce_std(input_tensor, axis=None, keepdims=False, name=None)
```

更多内容详见[tf.math.reduce_std](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/reduce_std)。

## mindspore.Tensor.std

```python
 mindspore.Tensor.std(self, axis=None, ddof=0, keepdims=False)
```

更多内容详见[mindspore.Tensor.std](https://mindspore.cn/docs/api/zh-CN/master/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor.std)。

## 使用方式

两接口基本功能相同，都是计算某个维度上Tensor的标准差，计算公式为：std = sqrt(mean(x)), 其中x = abs(a - a.mean())**2.。

不同点在于，`mindspore.Tensor.std`多一个入参`ddof`。一般情况下，均值为x.sum() / N, 其中N=len(x)，如果`ddof`被配置，分母将由N变为N-ddof。

## 代码示例

```python
import mindspore
from mindspore import Tensor
import numpy as np

a = Tensor(np.array([[1, 2], [3, 4]]), mindspore.float32)
print(a.std()) # 1.118034
print(a.std(axis=0)) # [1. 1.]
print(a.std(axis=1)) # [0.5 0.5]
print(a.std(ddof=1)) # 1.2909944

import tensorflow as tf
tf.enable_eager_execution()

x = tf.constant([[1., 2.], [3., 4.]])
print(tf.math.reduce_std(x).numpy())  # 1.118034
print(tf.math.reduce_std(x, 0).numpy())  # [1., 1.]
print(tf.math.reduce_std(x, 1).numpy())  # [0.5,  0.5]
```
