# 比较与tf.eye的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/eye.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## tf.eye

```text
tf.eye(
    num_rows,
    num_columns=None,
    batch_shape=None,
    dtype=tf.dtypes.float32
    name=None
) -> Tensor
```

更多内容详见[tf.eye](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/eye)。

## mindspore.ops.eye

```text
mindspore.ops.eye(n, m, t) -> Tensor
```

更多内容详见[mindspore.ops.eye](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.eye.html)。

## 差异对比

Tensorflow：TensorFlow中可以在参数中接受`batch_shape`，使输出具有这样的形状。

MindSpore：列数和数据类型不可缺省，功能上无差异。

| 分类 | 子类  | Tensorflow  | MindSpore | 差异                                                         |
| ---- | ----- | ----------- | --------- | ------------------------------------------------------------ |
| 参数 | 参数1 | num_rows    | n         | 功能一致， 参数名不同                                        |
|      | 参数2 | num_columns | m         | 指定张量的列数。TensorFlow中是可选的，如果没有该参数，那么返回一个列数和行数相同的张量；MindSpore中是必须的 |
|      | 参数3 | batch_shape | -       | 使输出具有指定的形状，MindSpore无此参数。如`batch_shape=[3]` |
|      | 参数4 | dtype       | t         | 名称不同，Tensorflow中是可选的，如果没有默认为`tf.dtypes.float32`；MindSpore中是必须的 |
|      | 参数5 | name       | -        | 不涉及 |

## 差异分析与示例

### 代码示例1

> Tensorflow可以缺省`num_columns`，MindSpore不可以缺省。

```python
# Tensorflow
import tensorflow as tf

e1 = tf.eye(3)
print(e1.numpy())
# [[1., 0., 0.]
#  [0., 1., 0.]
#  [0., 0., 1.]]

# MindSpore
import mindspore
import mindspore.ops as ops
e1 = ops.eye(3, 3, mindspore.float32)
print(e1.numpy())
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

### 代码示例2

> Tensorflow可以缺省`dtype`，MindSpore不可以缺省。

```python
# Tensorflow
import tensorflow as tf
e2 = tf.eye(3, 2)
print(e2.numpy())
# [[1, 0]
#  [0, 1]
#  [0, 0]]

# MindSpore
import mindspore
import mindspore.ops as ops
e2 = ops.eye(3, 2, mindspore.float32)
print(e2)
# [[1. 0.]
#  [0. 1.]
#  [0. 0.]]
```
