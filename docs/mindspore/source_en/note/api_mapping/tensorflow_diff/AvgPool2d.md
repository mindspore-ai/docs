# Function Differences with tf.nn.avg_pool2d

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/AvgPool2d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see [tf.nn.avg_pool2d](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/avg_pool2d).

## mindspore.nn.AvgPool2d

```text
mindspore.nn.AvgPool2d(
    kernel_size=1,
    stride=1,
    pad_mode='valid',
    data_format='NCHW'
)(x) -> Tensor
```

For more information, see [mindspore.nn.AvgPool2d](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.AvgPool2d.html).

## Differences

TensorFlow: Performs average pooling on the input Tensor.

MindSpore: MindSpore API implements the same function as TensorFlow, and only the parameter names and the way of using input Tensor are different.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1 | input       | x           | Same function, used to input a 4-dimensional Tensor. The data input format is different |
|      | Parameter 2 | ksize       | kernel_size | Same function, different parameter names, no default values for TensorFlow              |
|      | Parameter 3 | strides     | stride      | Same function, different parameter names, no default values for TensorFlow              |
|      | Parameter 4 | padding     | pad_mode    | Same function, different parameter names, no default values for TensorFlow              |
|      | Parameter 5 | data_format | data_format | Same function, different default values of parameters                                 |
|      | Parameter 6 | name | - | Not involved    |

### Code Example

> The two APIs achieve the same function and have the same usage.

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
