# 比较与tf.nn.conv2d的功能差异

## tf.nn.conv2d

``` text
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

更多内容详见 [tf.nn.conv2d](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/conv2d)。

## mindspore.nn.Conv2d

``` text
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

更多内容详见 [mindspore.nn.Conv2d](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Conv2d.html)。

## 差异对比

TensorFlow：对输入Tensor计算二维卷积，通常情况下，输入大小为 $\left(N, C_{\mathrm{in}}, H, W\right)$ 、输出大小为 $\left(N, C_{\text {out }}, H_{\text {out }}, W_{\text {out }}\right)$ 的输出值可以描述为：
$\operatorname{out}\left(N_{i}, C_{\text {out }_{j}}\right)=\operatorname{bias}\left(C_{\text {out }_{j}}\right)+\sum_{k=0}^{C_{i n}-1} \text { weight }\left(C_{\text {out }_{j}}, k\right) \star \operatorname{input}\left(N_{i}, k\right)$
其中，$\star$ 为2D cross-correlation 算子，$N$ 是batch size，$C$ 是通道数量，$H$ 和 $W$ 分别是特征层的高度和宽度。

MindSpore：与TensorFlow实现的功能基本一致，但部分参数结构、支持维度、默认值不同。MindSpore和TensorFlow的填充模式都包含了'same'、'valid'，但MindSpore相较于TensorFlow多了'pad'（零填充）。

| 分类 | 子类 |TensorFlow | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 参数 | 参数1 |input | x | 功能一致，参数名不同 |
| | 参数2 | filters | kernel_size |功能一致，参数名不同，数据结构不同 |
| | 参数3 | strides | stride |功能一致，参数名不同，支持维度不同，默认值不同 |
| | 参数4 | padding | pad_mode |功能一致，参数名不同，可选项不同，默认值不同|
| | 参数5 | data_format | data_format |功能一致，默认值不同|
| | 参数6 | dilations | dilation |功能一致，参数名不同，支持维度不同，默认值不同 |
| | 参数7 | name | - |不涉及|
| | 参数8 | - | in_channels |输入Tensor的空间维度 |
| | 参数9 | - | out_channels |输出Tensor的空间维度 |
| | 参数10 | - | padding |输入的高度和宽度方向上填充的数量|
| | 参数11 | -           | group |将过滤器拆分为组|
| | 参数12 | -           | has_bias |是否添加偏置参数|
| | 参数13 | - | weight_init |权重参数的初始化方法|
| | 参数14 | - | bias_init |偏置参数的初始化方法|

### 代码示例1

> TensorFlow的参数data_format默认值为'NHWC'，表示输入和输出的Tensor格式为[batchsize，in_height，in_width，in_channels]。MindSpore的参数data_format默认值为'NCHW'，表示输入和输出的Tensor格式为[batchsize，in_channels，in_height，in_width]。MindSpore的'NHWC'数据格式只能在GPU上使用，其它平台上，当输入数据格式为'NHWC'时，可以使用ops.transpose将数据格式修改为'NCHW'再进行卷积操作，最后将结果再通过ops.transpose转化为'NHWC'。

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

### 代码示例2

> TensorFlow的参数filters是一个四维Tensor，包括[filter_height，filter_width，in_channels，out_channels]，即[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]。MindSpore的参数kernel_size为整型或两个整型的tuple，一个整数表示卷积核的高度和宽度均为该值。两个整数的tuple分别表示卷积核的高度和宽度。

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

### 代码示例3

> TensorFlow的参数strides是一个一维向量，长度可以为1、2、4，表示卷积时每一维的步长。一个整数表示在高度和宽度方向的移动步长均为该值，两个整数分别表示在高度和宽度方向的移动步长，剩下两维移动步长默认为1，此参数无默认值。MindSpore的参数stride为整型或两个整型的tuple。一个整数表示在高度和宽度方向的移动步长均为该值。两个整数的tuple分别表示在高度和宽度方向的移动步长，参数默认值为1。

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

### 代码示例4

> TensorFlow的参数dilations是一个一维向量，长度可以为1、2、4，表示卷积核膨胀尺寸，在H和C维度上的值必须为1。MindSpore的参数dilation为整型或两个整型的tuple。

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

### 代码示例5

> TensorFlow的参数padding表示填充模式，没有默认值。MindSpore的参数pad_mode默认值为'same'。

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
