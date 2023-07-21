# 比较与tf.math.reduce_variance的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.10/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.10/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/TensorVar.md)

## tf.math.reduce_variance

```python
tf.math.reduce_variance(input_tensor, axis=None, keepdims=False, name=None)
```

更多内容详见[tf.math.reduce_variance](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/reduce_variance)。

## mindspore.Tensor.var

```python
mindspore.Tensor.var(axis=None, ddof=0, keepdims=False)
```

更多内容详见[mindspore.Tensor.var](https://mindspore.cn/docs/zh-CN/r1.10/api_python/mindspore/Tensor/mindspore.Tensor.var.html#mindspore.Tensor.var)。

## 使用方式

两接口基本功能相同，都是计算某个维度上Tensor的方差，计算公式为：var = mean(x), 其中x =  abs(a - a.mean())**2。

不同点在于，`mindspore.Tensor.var`多一个入参`ddof`。一般情况下，均值为x.sum() / N, 其中N=len(x)，如果`ddof`被配置，分母将由N变为N-ddof。

## 代码示例

```python
import mindspore as ms
import numpy as np

a = ms.Tensor(np.array([[1, 2], [3, 4]]), ms.float32)
print(a.var()) # 1.25
print(a.var(axis=0)) # [1. 1.]
print(a.var(axis=1)) # [0.25 0.25]
print(a.var(ddof=1)) # 1.6666666

import tensorflow as tf
tf.enable_eager_execution()

x = tf.constant([[1., 2.], [3., 4.]])
print(tf.math.reduce_variance(x).numpy())  # 1.25
print(tf.math.reduce_variance(x, 0).numpy())  # [1., 1.]
print(tf.math.reduce_variance(x, 1).numpy())  # [0.25,  0.25]
```
