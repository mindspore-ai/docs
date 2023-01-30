# Function Differences with tf.nn.max_pool2d

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/MaxPool2d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see [tf.nn.max_pool2d](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/max_pool2d).

## mindspore.nn.MaxPool2d

```text
class mindspore.nn.MaxPool2d(
    kernel_size=1,
    stride=1,
    pad_mode='valid',
    data_format='NCHW'
)(x) -> Tensor
```

For more information, see [mindspore.nn.MaxPool2d](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.MaxPool2d.html).

## Differences

TensorFlow: Perform two-dimensional maximum pooling operations on the input multidimensional data.

MindSpore: MindSpore API basically implements the same function as TensorFlow.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | input | x |Same function, different parameter names |
| | Parameter 2 | ksize | kernel_size | Same function, different parameter names, no default values for TensorFlow |
| | Parameter 3 | strides | stride | Same function, different parameter names, no default values for TensorFlow |
| | Parameter 4 | padding | pad_mode | Same function, different parameter names, no default values for TensorFlow |
| | Parameter 5 | data_format | data_format | - |
| | Parameter 6 | name | - | Not involved |

### Code Example 1

> In TensorFlow, when padding="SAME", corresponding to MindSpore with pad_mode="same" and data_format="NHWC", and then set ksize=3 and strides=2 to perform the maximum pooling operation on the input data in two dimensions. The two APIs achieve the same function.

```python
# TensorFlow
import tensorflow as tf

x = tf.constant([[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]]])
output = tf.nn.max_pool2d(x, ksize=3, strides=2, padding="SAME")
print(output.shape)
# (1, 1, 1, 10)

# MindSpore
import mindspore
import numpy as np
from mindspore import context
from mindspore import Tensor

device = context.get_context("device_target")
x = Tensor(np.array([[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]]]).astype(np.float32))
if device == "Ascend" or device == "CPU":
    max_pool = mindspore.nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
    x = mindspore.ops.transpose(x, (0, 3, 2, 1))
    output = max_pool(mindspore.Tensor(x))
    output = mindspore.ops.transpose(output, (0, 3, 2, 1))
    print(output.shape)
# (1, 1, 1, 10)
else:
    max_pool = mindspore.nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same', data_format='NHWC')
    output = max_pool(x)
    print(output.shape)
# (1, 1, 1, 10)
```
