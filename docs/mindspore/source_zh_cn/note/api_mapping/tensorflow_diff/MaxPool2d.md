# 比较与tf.nn.max_pool2d的功能差异

## tf.nn.max_pool2d

```text
tf.nn.max_pool2d(
    input,
    ksize,
    strides,
    padding,
    data_format='NHWC',
    name=None
) -> Tensor
```

更多内容详见 [tf.nn.max_pool2d](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/max_pool2d)。

## mindspore.nn.MaxPool2d

```text
class mindspore.nn.MaxPool2d(
    kernel_size=1,
    stride=1,
    pad_mode='valid',
    data_format='NCHW'
)(x) -> Tensor
```

更多内容详见 [mindspore.nn.MaxPool2d](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.MaxPool2d.html)。

## 差异对比

TensorFlow：对输入的多维数据进行二维的最大池化运算。

MindSpore：MindSpore此API实现功能与TensorFlow基本一致。

| 分类 | 子类 |TensorFlow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | input | x |功能一致，参数名不同 |
| | 参数2 | ksize | kernel_size | 功能一致，参数名不同 |
| | 参数3 | strides | stride | 功能一致，参数名不同 |
| | 参数4 | padding | pad_mode | 功能一致，参数名不同 |
| | 参数5 | data_format | data_format | - |
| | 参数6 | name | - | 不涉及 |

### 代码示例1

> 在TensorFlow中，当padding="VALID"，data_format="NCHW"时，对应MindSpore中pad_mode和data_format的默认值，再设置ksize=3，strides=1，对输入数据进行二维的最大池化运算，两API实现相同的功能。

```python
# TensorFlow
import tensorflow as tf
x = tf.ones((1, 2, 4, 4), dtype=tf.float32, name=None)
output = tf.nn.max_pool2d(x, ksize=3, strides=1, padding="VALID", data_format='NCHW')
print(output.shape)
# (1, 2, 2, 2)

# MindSpore
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np
pool = nn.MaxPool2d(kernel_size=3, stride=1)
x = Tensor(np.random.randint(0, 10, [1, 2, 4, 4]), mindspore.float32)
output = pool(x)
print(output.shape)
# (1, 2, 2, 2)
```

### 代码示例2

> 在TensorFlow中，当padding="SAME"时，对应MindSpore中pad_mode="same"，data_format="NHWC"，再设置ksize=3，strides=2，对输入数据进行二维的最大池化运算，两API实现相同的功能。

```python
# TensorFlow
import tensorflow as tf
x = tf.constant([[[[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]]]])
output = tf.nn.max_pool2d(x, ksize=3, strides=2, padding="SAME")
print(output.shape)
# (1, 1, 1, 10)

# MindSpore，Ascend环境
import mindspore
import numpy as np
max_pool = mindspore.nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
x = mindspore.Tensor([[[[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]]]],mindspore.float32)
x = mindspore.ops.transpose(x, (0, 3, 2, 1))
output = max_pool(mindspore.Tensor(x))
output = mindspore.ops.transpose(output, (0, 3, 2, 1))
print(output.shape)
# (1, 1, 1, 10)

# MindSpore，CPU和GPU环境
import mindspore
import numpy as np
max_pool = mindspore.nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same', data_format='NHWC')
x = Tensor([[[[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]]]],mindspore.float32)
output = max_pool(x)
print(output.shape)
# (1, 1, 1, 10)
```
