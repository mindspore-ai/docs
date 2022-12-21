# 比较与tf.nn.avg_pool2d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/tensorflow_diff/AvgPool2d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

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

更多内容详见[mindspore.nn.AvgPool2d](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.AvgPool2d.html)。

## 差异对比

TensorFlow：对输入的Tensor执行平均池化。

MindSpore：MindSpore此API实现功能与TensorFlow一致，仅参数名不同以及使用输入Tensor的方式不同。

| 分类 | 子类  | TensorFlow  | MindSpore   | 差异                              |
| ---- | ----- | ----------- | ----------- | --------------------------------- |
| 参数 | 参数1 | input       | x           | TensorFlow用于输入一个4-D的Tensor |
|      | 参数2 | ksize       | kernel_size | 功能一致，参数名不同，TensorFlow无默认值              |
|      | 参数3 | strides     | stride      | 功能一致，参数名不同，TensorFlow无默认值              |
|      | 参数4 | padding     | pad_mode    | 功能一致，参数名不同，TensorFlow无默认值              |
|      | 参数5 | data_format | data_format | 功能一致，参数名默认值不同                                 |

### 代码示例1

> 两API实现功能一致，用法相同。

```python
# TensorFlow
import tensorflow as tf
import numpy as np

y = tf.constant(10*np.random.random(size=(2,3,4,4)), dtype=tf.float16)
out = tf.nn.avg_pool2d(input=y, ksize=3, strides=1, padding='SAME')
print(out.shape)
# (2, 3, 4, 4)

# MindSpore
import mindspore
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor

pool = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='SAME')
x = Tensor(10*np.random.random(size=(2,3,4,4)), dtype=mindspore.float16)
output = pool(x)
print(output.shape)
# (2, 3, 4, 4)
```
