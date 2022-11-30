# 比较与tf.nn.conv2d_transpose的功能差异

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

更多内容详见[tf.nn.conv2d_transpose](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/nn/conv2d_transpose)。

## mindspore.nn.Conv2dTranspose

``` text
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

更多内容详见 [mindspore.nn.Conv2dTranspose](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Conv2dTranspose.html)。

## 差异对比

TensorFlow：计算二维转置卷积，可以视为conv2d对输入求梯度，也称为反卷积（实际不是真正的反卷积）。输入的shape通常是$(N,C,H,W)$或$(N,H,W,C)$，其中$N$是batch size，$C$是空间维度，$H_{in},W_{in}$分别为高度和宽度。有三种不同的填充方式："SAME"、"VALID"以及自定义列表[[0, 0], [pad_top,pad_bottom], [pad_left, pad_right], [0, 0]]，可以利用output_shape指定输出shape（同一大小的tensor可能由不同shape的tensor卷积而来），但如果不能由给定的参数计算出该shape则报错。

MindSpore：MindSpore此API实现功能与TensorFlow基本一致，部分参数的作用范围和数据类型与友商不同。MindSpore不能指定输出shape，但可利用参数weight_init和bias_init对权重和偏置直接初始化，也可对过滤器进行分组。

| 分类 | 子类  | TensorFlow | MindSpore | 差异 |
| ---- | ----- | ------- | --------- | -------------------------------------- |
| 参数 | 参数1 | input| x | 功能一致，参数名不同 |
|      | 参数2 | filters  |  kernel_size  | 描述卷积核的尺寸。TensorFlow为[height,width, output_channels, in_channels]分别表示卷积核的高度、宽度和个数，in_channels必须与input的一致，MindSpore为int型或tuple(int，int)，一个整数表示卷积核的高度和宽度均为该值。两个整数的tuple分别表示卷积核的高度和宽度 |
|      | 参数3 | output_shape | - | TensorFlow为长度为4的一维Tensor[N,H,W,C]，指定输出shape（尺寸错误则会发生报错），MindSpore输出维度需要计算得出 |
|      | 参数4 | strides   |  stride           | 转置卷积每一维的步长。TensorFlow若为一个int则代表宽度和高度上的步长，N和C上默认为0，若为长度为1,2或4的int型list，顺序与data_format一致。MindSpore为int型或tuple(int，int)，一个整数表示在高度和宽度方向移动步长均为该值。两个整数的tuple分别表示在高度和宽度的移动步长  |
|      | 参数5 | padding   |  padding           | TensorFlow表示填充模式，可选值为"SAME"，"VALID"，[[0, 0], [pad_top,pad_bottom], [pad_left, pad_right], [0, 0]] (NHWC)或[[0, 0], [0, 0],[pad_top, pad_bottom], [pad_left, pad_right]] (NCHW)。MindSpore中若padding是一个整数，那么上、下、左、右的填充都等于padding。如果padding是tuple(int,int,int,int)，那么上、下、左、右的填充分别等于padding[0]、padding[1]、padding[2]和padding[3]。值应该要大于等于0，默认为0 |
|      | 参数6 | data_format   |    | 设置格式，可选"NHWC"和"NCHW"，默认为"NHWC"  |
|      | 参数7 | dilations   |  dilation           | 二维卷积核膨胀尺寸，TensorFlow中若为长度为4的list，D和C维度上必须为1(格式与data_format一致)  |
|      | 参数8 |   name  | -        | 不涉及 |
|  | 参数1 | - | in_channels | 输入的空间维度，TensorFlow无此参数 |
|      | 参数9 | - | out_channels | 输出的空间维度，TensorFlow无此参数 |
|      | 参数10 |   -  | pad_mode       | 指定填充模式。可选值"same"、"valid"、"pad"与TensorFlow的padding参数对应一致。在"same"和"valid"模式下，padding必须设置为0，默认为"same" |
|      | 参数11 |  -  |  group           | 将过滤器拆分为组，in_channels和out_channels必须可被group整除。默认为1  |
|      | 参数12 |  -  |  has_bias           | 是否添加偏置函数，默认为False |
|      | 参数13 |   -  | weight_init        | 权重参数的初始化方法。可为Tensor，str，Initializer或numbers.Number。当使用str时，可选"TruncatedNormal"，"Normal"，"Uniform"，"HeUniform"和"XavierUniform"分布以及常量"One"和"Zero"分布的值。默认为"normal" |
|      | 参数14 |   -  | bias_init        | 偏置参数的初始化方法。初始化方法与"weight_init"相同，默认为"zeros" |

### 代码示例1

> 两API都是实现二维转置卷积运算，MindSpore在使用时需先进行实例化。TensorFlow中默认顺序为NHWC，MindSpore为NCHW，将TensorFlow的padding设置为[[0,0], [0,0], [0,0], [0,0]]，对应将MindSpore的pad_mode设为"pad"，padding=[0,0,0,0]。输入Tensor为[1,3,16,50]-->输出Tensor将为[1,64,19,53]，在TensorFlow中还会检验output_shape是否与给定参数计算出的shape一致，否则会报错。

```python
# TensorFlow
import tensorflow as tf
import numpy as np

k = 4
x_ = np.ones([1, 16, 50, 3])
x = tf.convert_to_tensor(x_, dtype=tf.float32)
f = np.ones((k,k,64,3), dtype=np.float32)
output = tf.nn.conv2d_transpose(x, filters=f, output_shape=[1,19,53,64], strides=1, padding=[[0, 0], [0,0], [0, 0], [0, 0]])
print(tf.transpose(output,[0,3,1,2]).shape)
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

### 代码示例2

> 为使输出的宽度与输入整除stride后的值相同，TensorFlow中先指定output_shape = [1,64,16,50]，padding设置为"SAME"。MindSpore则设置pad_mode = "same"，同时padding = 0。

```python
# TensorFlow
import tensorflow as tf
import numpy as np

k = 5
x_ = np.ones([1, 16, 50, 3])
x = tf.convert_to_tensor(x_, dtype=tf.float32)
f = np.ones((k,k,64,3), dtype=np.float32)
output = tf.nn.conv2d_transpose(x, filters=f, output_shape=[1,16,50, 64], strides=1, padding="SAME")
print(tf.transpose(output,[0,3,1,2]).shape)
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

### 代码示例3

> 若不在原有图像上做任何填充，在stride>1的情况下可能舍弃一部分数据，在TensorFlow中将padding设为"VALID"，MindSpore中设置pad_mode = "valid"，同时padding设为0。

```python
# TensorFlow
import tensorflow as tf
import numpy as np

k = 5
s = 3
x_ = np.ones([1, 16, 50, 3])
x = tf.convert_to_tensor(x_, dtype=tf.float32)
f = np.ones((k,k,64,3), dtype=np.float32)
output = tf.nn.conv2d_transpose(x, filters=f, output_shape=[1,50,152, 64], strides=s, padding="VALID")
print(tf.transpose(output,[0,3,1,2]).shape)
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