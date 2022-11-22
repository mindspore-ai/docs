# 比较与tf.fill的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/fill.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.fill

```text
tf.fill(dims, value, name=None) -> Tensor
```

更多内容详见 [tf.fill](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/fill)。

## mindspore.ops.fill

```text
mindspore.ops.fill(type, shape, value) -> Tensor
```

更多内容详见 [mindspore.ops.fill](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.fill.html)。

## 差异对比

TensorFlow：‎用于生成具有标量值的张量

MindSpore：与TensorFlow实现同样的功能，仅参数名不同

| 分类 | 子类 |TensorFlow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | dims | shape |功能一致，参数名不同 |
|  | 参数2 | value | value | - |
|  | 参数3 | name | - | 不涉及 |
|  | 参数4 | - | type | 指定输出Tensor的数据类型 |

### 代码示例1

> 两个API实现功能相同，MindSpore仅多一个指定输出的类型参数，其余参数用法相同

```python
# TensorFlow
import tensorflow as tf
import numpy as np

dims = np.array([2,3])
value = 9
output = tf.fill(dims, value)
output_m = output.numpy()
print(output_m)
#[[9 9 9]
# [9 9 9]]

# MindSpore
import mindspore
import mindspore.ops as ops

type = mindspore.int32
shape = tuple((2,3))
value = 9
output = ops.fill(type, shape, value)
print(output)
#[[9 9 9]
# [9 9 9]]
```
