# 比较与tf.nn.avg_pool2d的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/AvgPool2d.md)

## tf.nn.avg_pool2d

```text
tf.nn.avg_pool2d(
    input,
    ksize,
    strides,
    padding,
    data_format='NHWC',
    name=None
) -> Tensor
```

更多内容详见[tf.nn.avg_pool2d](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/avg_pool2d)。

## mindspore.nn.AvgPool2d

```text
mindspore.nn.AvgPool2d(
    kernel_size=1,
    stride=1,
    pad_mode='valid',
    data_format='NCHW'
)(x) -> Tensor
```

更多内容详见[mindspore.nn.AvgPool2d](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.AvgPool2d.html)。

## 差异对比

TensorFlow：对输入的Tensor执行平均池化。

MindSpore：MindSpore此API实现功能与TensorFlow一致，仅参数名不同以及使用输入Tensor的方式不同。

| 分类 | 子类  | TensorFlow  | MindSpore   | 差异                                                  |
| ---- | ----- | ----------- | ----------- | ----------------------------------------------------- |
| 参数 | 参数1 | input       | x           | 功能一致，用于输入一个4维的Tensor，数据的输入格式不同 |
|      | 参数2 | ksize       | kernel_size | 功能一致，参数名不同，TensorFlow无默认值              |
|      | 参数3 | strides     | stride      | 功能一致，参数名不同，TensorFlow无默认值              |
|      | 参数4 | padding     | pad_mode    | 功能一致，参数名不同，TensorFlow无默认值              |
|      | 参数5 | data_format | data_format | 功能一致，参数默认值不同                                 |
|      | 参数6 | name | - | 不涉及        |

### 代码示例

> 两API实现功能一致，用法相同。

```python
# TensorFlow
import tensorflow as tf
import numpy as np

y = tf.constant([[[[1, 0, 1], [0, 1, 1]]]], dtype=tf.float32)
out = tf.nn.avg_pool2d(input=y, ksize=1, strides=1, padding='SAME')
print(out.numpy())
# [[[[1. 0. 1.]
#    [0. 1. 1.]]]]

# MindSpore
import mindspore
import mindspore.nn as nn
from mindspore import Tensor

pool = nn.AvgPool2d(kernel_size=1, stride=1, pad_mode='SAME')
x = Tensor([[[[1, 0, 1], [0, 1, 1]]]], dtype=mindspore.float32)
output = pool(x)
print(output)
# [[[[1. 0. 1.]
#    [0. 1. 1.]]]]
```
