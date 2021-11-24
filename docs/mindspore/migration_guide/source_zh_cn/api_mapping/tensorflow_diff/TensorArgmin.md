# 比较与tf.arg_min的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/tensorflow_diff/TensorArgmin.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## tf.arg_min

```python
tf.arg_min(input, dimension, output_type=tf.dtypes.int64, name=None)
```

更多内容详见[tf.arg_min](http://www.tensorflow.org/versions/r1.15/api_docs/python/tf/arg_min)。

## mindspore.Tensor.argmin

```python
 mindspore.Tensor.argmin(axis=None)
```

更多内容详见[mindspore.Tensor.argmin](https://mindspore.cn/docs/api/zh-CN/master/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor.argmin)。

## 使用方式

功能相同，MindSpore和TensorFlow的两接口分别通过参数`axis`和`dimension`入参决定基于哪个维度返回最大值的索引。

不同点在于，默认状态下，MindSpore的`axis=None`，返回最小值的全局索引；TensorFlow的`dimension`不传入数值时，默认返回`dimension=0`的最小值索引。

## 代码示例

```python
import mindspore
from mindspore import Tensor

a = Tensor([[1, 10, 166.32, 62.3], [1, -5, 2, 200]], mindspore.float32)
print(a.argmin())
print(a.argmin(axis=0))
print(a.argmin(axis=1))
# output:
# 5
# [0 1 1 0]
# [0 1]

import tensorflow as tf
tf.enable_eager_execution()

b = tf.constant([[1, 10, 166.32, 62.3], [1, -5, 2, 200]])
print(tf.argmin(b).numpy())
print(tf.argmin(b, dimension=0).numpy())
print(tf.argmin(b, dimension=1).numpy())
# output:
# [0 1 1 0]
# [0 1 1 0]
# [0 1]
```
