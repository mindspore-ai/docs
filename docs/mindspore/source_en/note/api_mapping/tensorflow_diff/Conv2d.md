# Function Differences with tf.nn.conv2d

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/Conv2d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.nn.conv2d

```text
tf.nn.conv2d(
    input,
    filters,
    strides,
    padding,
    data_format='NHWC',
    dilations=None,
    name=None
) -> Tensor
```

For more information, see [tf.nn.conv2d](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/conv2d).

## mindspore.nn.Conv2d

```text
class mindspore.nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    pad_mode='same',
    padding=0,
    dilation=1,
    group=1,
    has_bias=False,
    weight_init='normal',
    bias_init='zeros',
    data_format='NCHW'
)(x) -> Tensor
```

For more information, see [mindspore.nn.Conv2d](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Conv2d.html).

## Differences

TensorFlow: To compute a two-dimensional convolution on the input Tensor, typically the output values with input size $\left(N, C_{\mathrm{in}}, H, W\right)$ and output size $\left(N, C_{\text {out }}, H_{\text {out }}, W_{\text {out }}\right)$ can be described as:
$\operatorname{out}\left(N_{i}, C_{\text {out }_{j}}\right)=\operatorname{bias}\left(C_{\text {out }_{j}}\right)+\sum_{k=0}^{C_{i n}-1} \text { weight }\left(C_{\text {out }_{j}}, k\right) \star \operatorname{input}\left(N_{i}, k\right)$,
where $\star$ is the 2D cross-correlation operator, $N$ is the batch size, $C$ is the number of channels, and $H$ and $W$ are the height and width of the feature layer, respectively.

MindSpore: MindSpore API basically implements the same function as TensorFlow. However, some of the parameters have different structures, support dimensions, and default values. MindSpore and TensorFlow both contain 'same', 'valid' in their fill modes, but MindSpore has more 'pad' (zero fill) compared to TensorFlow.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1 |input | x | Same function, different parameter names |
| | Parameter 2 | filters | kernel_size |Same function, different parameter names, different data structure |
| | Parameter 3 | strides | stride |Same function, different parameter names, different supported dimensions, different default values |
| | Parameter 4 | padding | pad_mode |Same function, different parameter names, different options, different default values|
| | Parameter 5 | data_format | data_format |Same function, different default value|
| | Parameter 6 | dilations | dilation |Same function, different parameter names, different supported dimensions, different default values |
| | Parameter 7 | name | - |Not involved|
| | Parameter 8 | - | in_channels |Spatial dimension of the input Tensor |
| | Parameter 9 | - | out_channels |Spatial dimension of the output Tensor |
| | Parameter 10 | - | padding |Number of padding in the direction of height and width of the input|
| | Parameter 11 | -           | group |Splitting filters into groups|
| | Parameter 12 | -           | has_bias |Whether to add bias parameters|
| | Parameter 13 | - | weight_init |Initialization method of weight parameters|
| | Parameter 14 | - | bias_init |Initialization method of bias parameters|

### Code Example 1

> The default value of data_format for TensorFlow is 'NHWC', which means the input and output Tensor format is [batchsize, in_height, in_width, in_channels]. The default value of data_format for MindSpore is 'NCHW', which means the input and output Tensor format is [batchsize, in_height, in_width, in_channels]. MindSpore 'NHWC' data format can only be used on GPU. On other platforms, when the input data format is 'NHWC', you can use ops.transpose to modify the data format to 'NCHW' and then perform the convolution operation, and finally convert the result to 'NHWC' by ops.transpose again.

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x_ = tf.ones((1, 3, 3, 5))
x = tf.convert_to_tensor(x_, dtype=tf.float32)
filters_ = tf.ones((2, 2, 5, 1))
filters = tf.convert_to_tensor(filters_, dtype=tf.float32)
output = tf.nn.conv2d(x, filters, strides=1, padding='SAME').shape
print(output)
# (1, 3, 3, 1)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np

x_ = np.ones((1, 3, 3, 5))
x_NHWC = Tensor(x_, mindspore.float32)
x = ops.transpose(x_NHWC, (0, 3, 1, 2))
net = nn.Conv2d(5, 1, 2, stride=1, pad_mode='same')
output = ops.transpose(net(x), (0, 2, 3, 1)).shape
print(output)
# (1, 3, 3, 1)
```

### Code Example 2

> The TensorFlow parameter filters is a four-dimensional Tensor, including [filter_height, filter_width, in_channels, out_channels], i.e. [height of convolution kernel, width of convolution kernel, number of image channels, number of convolution kernels]. MindSpore parameter kernel_size is an integer or two integer tuples, one integer means that both the height and width of the convolution kernel are of that value. The two integer tuples represent the height and width of the convolution kernel, respectively.

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x_ = tf.ones((1, 4, 4, 5))
x = tf.convert_to_tensor(x_, dtype=tf.float32)
filters_ = tf.ones((2, 3, 5, 1))
filters = tf.convert_to_tensor(filters_, dtype=tf.float32)
output = tf.nn.conv2d(x, filters, strides=1, padding='VALID').shape
print(output)
# (1, 3, 2, 1)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np

x_ = np.ones((1, 4, 4, 5))
x_NHWC = Tensor(x_, mindspore.float32)
x = ops.transpose(x_NHWC, (0, 3, 1, 2))
net = nn.Conv2d(5, 1, (2, 3), stride=1, pad_mode='valid')
output = ops.transpose(net(x), (0, 2, 3, 1)).shape
print(output)
# (1, 3, 2, 1)
```

### Code Example 3

> The TensorFlow parameter strides is a one-dimensional vector, which can be 1, 2, or 4 in length, indicating the step length of each dimension during convolution. One integer indicates that the strides are in both height and width directions, two integers indicate the strides in height and width directions, and the strides in the remaining two dimensions are 1 by default. There is no default value for this parameter. MindSpore parameter stride is an integer or two integer tuples. One integer means the strides in both the height and width directions. The two integer tuples indicate the strides in the height and width directions respectively, and the default value of the parameter is 1.

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x_ = tf.ones((1, 4, 4, 5))
x = tf.convert_to_tensor(x_, dtype=tf.float32)
filters_ = tf.ones((2, 3, 5, 1))
filters = tf.convert_to_tensor(filters_, dtype=tf.float32)
output = tf.nn.conv2d(x, filters, strides=[1,1,1,1], padding='VALID').shape
print(output)
# (1, 3, 2, 1)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np

x_ = np.ones((1, 4, 4, 5))
x_NHWC = Tensor(x_, mindspore.float32)
x = ops.transpose(x_NHWC, (0, 3, 1, 2))
net = nn.Conv2d(5, 1, (2, 3), pad_mode='valid')
output = ops.transpose(net(x), (0, 2, 3, 1)).shape
print(output)
# (1, 3, 2, 1)
```

### Code Example 4

> The TensorFlow parameter dilations is a one-dimensional vector, which can be of length 1, 2, or 4, indicating the convolution kernel expansion size, and must have a value of 1 in the H and C dimensions. The MindSpore parameter dilations is an integer or a tuples of two integers.

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x_ = tf.ones((1, 6, 6, 5))
x = tf.convert_to_tensor(x_, dtype=tf.float32)
filters_ = tf.ones((2, 3, 5, 1))
filters = tf.convert_to_tensor(filters_, dtype=tf.float32)
output = tf.nn.conv2d(x, filters, strides=1, dilations=[1,2,2,1], padding='VALID').shape
print(output)
# (1, 4, 2, 1)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np

x_ = np.ones((1, 6, 6, 5))
x_NHWC = Tensor(x_, mindspore.float32)
x = ops.transpose(x_NHWC, (0, 3, 1, 2))
net = nn.Conv2d(5, 1, (2, 3), dilation=(2,2), pad_mode='valid')
output = ops.transpose(net(x), (0, 2, 3, 1)).shape
print(output)
# (1, 4, 2, 1)
```

### Code Example 5

> The TensorFlow parameter padding indicates the padding mode and has no default value. The default value of MindSpore parameter pad_mode is 'same'.

```python
# TensorFlow
import tensorflow as tf
import numpy as np

x_ = tf.ones((1, 4, 4, 5))
x = tf.convert_to_tensor(x_, dtype=tf.float32)
filters_ = tf.ones((2, 3, 5, 1))
filters = tf.convert_to_tensor(filters_, dtype=tf.float32)
output = tf.nn.conv2d(x, filters, strides=1, padding='SAME').shape
print(output)
# (1, 4, 4, 1)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np

x_ = np.ones((1, 4, 4, 5))
x_NHWC = Tensor(x_, mindspore.float32)
x = ops.transpose(x_NHWC, (0, 3, 1, 2))
net = nn.Conv2d(5, 1, (2, 3), stride=1)
output = ops.transpose(net(x), (0, 2, 3, 1)).shape
print(output)
# (1, 4, 4, 1)
```
