# Function Differences with tf.nn.conv2d_transpose

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/Conv2dTranspose.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.nn.conv2d_transpose

```text
tf.nn.conv2d_transpose(
    input,
    filters,
    output_shape,
    strides,
    padding='SAME',
    data_format='NHWC',
    dilations=None,
    name=None
) -> Tensor
```

For more information, see [tf.nn.conv2d_transpose](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/conv2d_transpose).

## mindspore.nn.Conv2dTranspose

```text
class mindspore.nn.Conv2dTranspose(
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
    bias_init='zeros'
)(x) -> Tensor
```

For more information, see [mindspore.nn.Conv2dTranspose](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Conv2dTranspose.html).

## Differences

TensorFlow: Computing a two-dimensional transposed convolution can be thought of as conv2d solving for the gradient of the input, also known as deconvolution (which is not really deconvolution). The input shape is usually $(N,C,H,W)$ or $(N,H,W,C)$, where $N$ is the batch size, $C$ is the spatial dimension, and $H_{in},W_{in}$ are the height and width, respectively. There are three different types of padding: "SAME", "VALID", and a custom list [[0, 0], [pad_top,pad_bottom], [pad_left, pad_right], [0, 0]], and the output shape can be specified using output_shape (a tensor of the same size may be convolved from tensors of different shapes), but an error is reported if the shape cannot be computed from the given parameters.

MindSpore: MindSpore API basically implements the same function as TensorFlow. The scope and data types of some parameters are different from competitors. MindSpore cannot specify the output shape, but the weights and bias can be initialized directly using the parameters weight_init and bias_init, and the filters can be grouped.

| Categories | Subcategories |TensorFlow | MindSpore | Differences |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1 | input| x | Same function, different parameter names |
|      | Parameter 2 | filters  |  kernel_size  | Describe the size of the convolution kernel. TensorFlow: [height,width, output_channels, in_channels] is the height, width and number of convolution kernels. in_channels must be consistent with the input; MindSpore is int type or tuple (int, int). An integer indicates that the height and width of the convolution kernel are the same value. Two integer tuples indicate the height and width of the convolution kernel, respectively |
|      | Parameter 3 | output_shape | - | TensorFlow is a one-dimensional Tensor [N,H,W,C] of length 4, and specifies the output shape (an error will occur if the size is wrong). MindSpore output dimension needs to be calculated. |
|      | Parameter 4 | strides   |  stride           | Transpose the stride of each dimension of the convolution. TensorFlow represents strides in width and height if it is an int, and defaults to 0 on N and C. If it is an int list of length 1, 2 or 4, the order is the same as data_format.MindSpore is int type or tuple(int, int). An integer indicates the value of the move step in both height and width directions. Two integer tuples mean move steps in height and width respectively. |
|      | Parameter 5 | padding   |  padding           | TensorFlow indicates the padding mode with optional values of "SAME", "VALID", [[0, 0], [pad_top,pad_bottom], [pad_left, pad_right], [0, 0]] (NHWC) or [[0, 0], [0, 0], [pad_top, pad_bottom ], [pad_left, pad_right]] (NCHW). If padding is an integer in MindSpore, the top, bottom, left, and right padding are all equal to padding. If the padding is tuple(int,int,int,int), the top, bottom, left and right padding are equal to padding[0], padding[1], padding[2] and padding[3] respectively. The value should be greater than or equal to 0. The default is 0. |
|      | Parameter 6 | data_format   |    | Set the format. Optional values are "NHWC" and "NCHW", and the default is "NHWC". MindSpore defaults to "NCHW". |
|      | Parameter 7 | dilations   |  dilation           | 2D convolutional kernel expansion size. In TensorFlow a list of length 4, must be 1 in D and C dimensions (format consistent with data_format)  |
|      | Parameter 8 |   name  | -        | Not involved |
|      | Parameter 9 | - | in_channels | The spatial dimension of the input. TensorFlow does not have this parameter. |
|      | Parameter 10 | - | out_channels | The spatial dimension of the output. TensorFlow does not have this parameter. |
|      | Parameter 11 |   -  | pad_mode       | Specify the padding mode. The optional values "same", "valid" and "pad" corresponding to the TensorFlow padding parameters. In "same" and "valid" mode, padding must be set to 0, and default is "same". |
|      | Parameter 12 |  -  |  group           | Split the filter into groups, and in_channels and out_channels must be divisible by group. Default is 1. TensorFlow does not have this parameter.  |
|      | Parameter 13 |  -  |  has_bias           | Whether to add a bias function. Default is False. TensorFlow does not have this parameter. |
|      | Parameter 14 |   -  | weight_init        | The initialization method for the weights parameter. Can be Tensor, str, Initializer or numbers.Number. When using str, the values of the "TruncatedNormal", "Normal", "Uniform", "HeUniform" and "XavierUniform" distributions and the constants "One" and "Zero" distributions can be selected. Default is "normal". TensorFlow does not have this parameter. |
|      | Parameter 15 |   -  | bias_init        | The initialization method for the bias parameter. The initialization method is the same as "weight_init", the default is "zeros". TensorFlow does not have this parameter. |

### Code Example 1

> Both APIs implement two-dimensional transposed convolutional operations, and MindSpore needs to be instantiated first when used. The default order in TensorFlow is NHWC, while MindSpore is NCHW. Set the padding of TensorFlow to [[0,0], [0,0], [0,0], [0,0]], [0,0]], corresponding to set the pad_mode of MindSpore to "pad", padding=[0,0,0,0]. The input Tensor is [1,3,16,50] --> the output Tensor will be [1,64,19,53], and in TensorFlow it will also check if the output_shape is the same as the one computed with the given parameters, otherwise it will report an error.

```python
# TensorFlow
import tensorflow as tf
import numpy as np

k = 4
x_ = np.ones([1, 16, 50, 3])
x = tf.convert_to_tensor(x_, dtype=tf.float32)
f = np.ones((k, k, 64, 3), dtype=np.float32)
output = tf.nn.conv2d_transpose(x, filters=f, output_shape=[1, 19, 53, 64], strides=1, padding=[[0, 0], [0,0], [0, 0], [0, 0]])
print(tf.transpose(output, [0, 3, 1, 2]).shape)
# (1, 64, 19, 53)


# MindSpore
import mindspore as ms
import mindspore.nn as nn
import numpy as np

k = 4
x_ = np.ones([1, 3, 16, 50])
x = ms.Tensor(x_, ms.float32)
net = nn.Conv2dTranspose(3, 64, kernel_size=k, weight_init='normal', pad_mode='pad')
output = net(x)
print(output.shape)
# (1, 64, 19, 53)
```

### Code Example 2

> To make the output width the same as the input after dividing stride, TensorFlow first specifies output_shape = [1,64,16,50] with padding set to "SAME", while MindSpore sets pad_mode = "same" and padding = 0.

```python
# TensorFlow
import tensorflow as tf
import numpy as np

k = 5
x_ = np.ones([1, 16, 50, 3])
x = tf.convert_to_tensor(x_, dtype=tf.float32)
f = np.ones((k, k, 64, 3), dtype=np.float32)
output = tf.nn.conv2d_transpose(x, filters=f, output_shape=[1, 16, 50, 64], strides=1, padding='SAME')
print(tf.transpose(output, [0, 3, 1, 2]).shape)
# (1, 64, 16, 50)


# MindSpore
import mindspore as ms
import mindspore.nn as nn
import numpy as np

k = 5
x_ = np.ones([1, 3, 16, 50])
x = ms.Tensor(x_, ms.float32)
net = nn.Conv2dTranspose(3, 64, kernel_size=k, stride=1, weight_init='normal', pad_mode='same', padding=0)
output = net(x)
print(output.shape)
# (1, 64, 16, 50)
```

### Code Example 3

> If you do not do any padding on the original image, you may discard part of the data if stride>1. Set padding to "VALID" in TensorFlow, set pad_mode = "valid" in MindSpore, and set padding to 0.

```python
# TensorFlow
import tensorflow as tf
import numpy as np

k = 5
s = 3
x_ = np.ones([1, 16, 50, 3])
x = tf.convert_to_tensor(x_, dtype=tf.float32)
f = np.ones((k, k, 64, 3), dtype=np.float32)
output = tf.nn.conv2d_transpose(x, filters=f, output_shape=[1, 50, 152, 64], strides=s, padding='VALID')
print(tf.transpose(output, [0, 3, 1, 2]).shape)
# (1, 64, 50, 152)


# MindSpore
import mindspore as ms
import mindspore.nn as nn
import numpy as np

k = 5
s = 3
x_ = np.ones([1, 3, 16, 50])
x = ms.Tensor(x_, ms.float32)
net = nn.Conv2dTranspose(3, 64, kernel_size=k, stride=s, weight_init='normal', pad_mode='valid', padding=0)
output = net(x)
print(output.shape)
# (1, 64, 50, 152)
```
