# 比较与torch.nn.ConvTranspose2d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Conv2dTranspose.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.ConvTranspose2d

```text
class torch.nn.ConvTranspose2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    bias=True,
    dilation=1,
    padding_mode='zeros'
)(input) -> Tensor
```

更多内容详见[torch.nn.ConvTranspose2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ConvTranspose2d.html)。

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

更多内容详见[mindspore.nn.Conv2dTranspose](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Conv2dTranspose.html)。

## 差异对比

PyTorch：计算二维转置卷积，可以视为Conv2d对输入求梯度，也称为反卷积（实际不是真正的反卷积）。输入的shape通常是$(N,C_{in},H_{in},W_{in})$，其中$N$是batch size，$C$是空间维度，$H_{in},W_{in}$分别为高度和宽度。输出的shape为$(N,C_{out},H_{out},W_{out})$，高度和宽度分别为：
$H_{out}=(H_{in}−1)×stride[0]−2×padding[0]+dilation[0]×(kernel\underline{ }size[0]−1)+output\underline{ }padding[0]+1$
$W_{out}=(W_{in}−1)×stride[1]−2×padding[1]+dilation[1]×(kernel\underline{ }size[1]−1)+output\underline{ }padding[1]+1$

MindSpore：MindSpore此API实现功能与PyTorch基本一致，新增了填充模式参数"pad_mode"，当"pad_mode" = "pad"时与PyTorch默认方式相同，利用weight_init和bias_init参数可以配置初始化方式。此外，torch.nn.ConvTranspose2d有一个output_padding参数，其功能是指对反卷积后的特征图进行单侧补零（右侧和下侧），而mindspore.nn.Conv2dTranspose中目前没有该参数，可以对输出结果使用[nn.Pad](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Pad.html#mindspore.nn.Pad)进行补维来代替。

| 分类 | 子类   | PyTorch        | MindSpore    | 差异                                                         |
| ---- | ------ | -------------- | ------------ | ------------------------------------------------------------ |
| 参数 | 参数1  | in_channels    | in_channels  | -    |
|      | 参数2  | out_channels   | out_channels | -         |
|      | 参数3  | kernel_size    | kernel_size  | -        |
|      | 参数4  | stride         | stride       | -        |
|      | 参数5  | padding        | padding      | 功能一致，PyTorch中只能在两个维度的两侧分别填充相同的值，可为长度为2的tuple。MindSpore中可以分别设置顶部、底部、左边和右边的填充数量，可为长度为4的tuple |
|      | 参数6  | output_padding | -            | 对反卷积后的特征图进行单侧补零（右侧和下侧），通常在stride > 1的前提下使用，用来调整output shapes。例如，通常将padding设置为(kernel_size - 1)/2，此时设置output_padding = (stride - 1)可确保input shapes/output shapes = stride，MindSpore无此参数 |
|      | 参数7  | groups         | group        | 功能一致，参数名不同                                         |
|      | 参数8  | bias           | has_bias     | PyTorch默认为True，MindSpore默认为False                      |
|      | 参数9  | dilation       | dilation     | -        |
|      | 参数10 | padding_mode   | -            | 数值填充模式，只支持"zeros"即填充0。MindSpore无此参数，但默认填充0 |
|      | 参数11 | -              | pad_mode     | 指定填充模式。可选值为"same"、"valid"、"pad"，在"same"和"valid"模式下，padding必须设置为0，默认为"same"，PyTorch无此参数 |
|      | 参数12 | -              | weight_init  | 权重参数的初始化方法。可为Tensor、str、Initializer或numbers.Number。当使用str时，可选"TruncatedNormal" 、"Normal" 、"Uniform" 、"HeUniform"和"XavierUniform"分布以及常量"One"和"Zero"分布的值，默认为"normal"，PyTorch无此参数 |
|      | 参数13 | -              | bias_init    | 偏置参数的初始化方法。初始化可选参数与"weight_init"相同，默认为"zeros"，PyTorch无此参数 |
| 输入 | 单输入 | input          | x            | 功能一致，参数名不同                                         |

### 代码示例1

> 两API都是实现二维转置卷积运算，使用时需先进行实例化。PyTorch中高度和宽度的padding值在同一方向上相同，如padding设为(2,4)表示分别在高度和宽度的两侧填充2行和4列0，对应在MindSpore中将pad_mode设为"pad"，padding设置为(2,2,4,4)。PyTorch中利用net.weight.data = torch.ones()的方式将权重初始化为1，shape为$(in\underline{ }channels,\frac {out\underline{ }channels}{groups} , kernel\underline{ }size[0], kernel\underline{ }size[1])$，MindSpore直接设置参数weight_init = "ones"。

```python
# PyTorch
import torch
from torch import tensor
import torch.nn as nn
import numpy as np

k = 4
x_ = np.ones([1, 3, 16, 50])
x = tensor(x_, dtype=torch.float32)
net = nn.ConvTranspose2d(3, 64, kernel_size=k, stride=1, padding=(2, 4), output_padding=0, bias=False)
net.weight.data = torch.ones(3, 64, k, k)
output = net(x).detach().numpy()
print(output.shape)
# (1, 64, 15, 45)

# MindSpore
import mindspore as ms
import mindspore.nn as nn
import numpy as np

k = 4
x_ = np.ones([1, 3, 16, 50])
x = ms.Tensor(x_, ms.float32)
net = nn.Conv2dTranspose(3, 64, kernel_size=k, weight_init='ones', pad_mode='pad', padding=(2, 2, 4, 4))
output = net(x)
print(output.shape)
# (1, 64, 15, 45)
```

### 代码示例2

> 为使输出的宽度与输入整除stride后的值相同，PyTorch中设置output_padding = stride - 1，padding设置为(kernel_size - 1)/2。MindSpore则设置pad_mode = "same"，同时padding = 0。

```python
# PyTorch
import torch
from torch import tensor
import torch.nn as nn
import numpy as np

k = 5
s = 3
x_ = np.ones([1, 3, 16, 50])
x = tensor(x_, dtype=torch.float32)
net = nn.ConvTranspose2d(3, 64, kernel_size=k, stride=s, padding=(k-1)//2, output_padding=s-1, bias=False)
net.weight.data = torch.ones(3, 64, k, k)
output = net(x).detach().numpy()
print(output.shape)
# (1, 64, 48, 150)


# MindSpore
import mindspore as ms
import mindspore.nn as nn
import numpy as np

k = 5
s = 3
x_ = np.ones([1, 3, 16, 50])
x = ms.Tensor(x_, ms.float32)
net = nn.Conv2dTranspose(3, 64, kernel_size=k, stride=s, weight_init='ones', pad_mode='same', padding=0)
output = net(x)
print(output.shape)
# (1, 64, 48, 150)
```

### 代码示例3

> 若不在原有图像上做任何填充，在stride>1的情况下可能舍弃一部分数据，在PyTorch中将padding和output_padding设为0，MindSpore中设置pad_mode = "valid"，同时padding = 0。

```python
# PyTorch
import torch
from torch import tensor
import torch.nn as nn
import numpy as np

k = 5
s = 3
x_ = np.ones([1, 3, 16, 50])
x = tensor(x_, dtype=torch.float32)
net = nn.ConvTranspose2d(3, 64, kernel_size=k, stride=s, padding=0, output_padding=0, bias=False)
net.weight.data = torch.ones(3, 64, k, k)
output = net(x).detach().numpy()
print(output.shape)
# (1, 64, 50, 152)


# MindSpore
import mindspore as ms
import mindspore.nn as nn
import numpy as np

k = 5
s = 3
x_ = np.ones([1, 3, 16, 50])
x = ms.Tensor(x_, ms.float32)
net = nn.Conv2dTranspose(3, 64, kernel_size=k, stride=s, weight_init='ones', pad_mode='valid', padding=0)
output = net(x)
print(output.shape)
# (1, 64, 50, 152)
```

### 代码示例4

> 下面的示例实现了对输入tensor进行反卷积，并且输出反卷积后的特征图尺寸，其中PyTorch可以通过设置output_padding的值来对反卷积后的输出图像进行右侧和下侧补维，用于弥补stride大于1带来的缺失。MindSpore暂时不支持output_padding参数，需要对输出结果再使用[nn.Pad](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Pad.html#mindspore.nn.Pad)进行单侧补维。

```python
# PyTorch
import torch
import torch.nn as nn
import numpy as np

m = nn.ConvTranspose2d(in_channels=3, out_channels=32,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       output_padding=1,
                       bias=False)
input = torch.tensor(np.ones([1, 3, 48, 48]), dtype=torch.float32)
output = m(input).detach().numpy()
print(output.shape)
#(1, 32, 96, 96)

# MindSpore
import mindspore as ms
import mindspore.nn as nn
import numpy as np
input = ms.Tensor(np.ones([1, 3, 48, 48]), dtype=ms.float32)
m = nn.Conv2dTranspose(in_channels=3,
                       out_channels=32,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       pad_mode="pad",
                       has_bias=False)
output = m(input)
pad = nn.Pad(paddings=((0, 0), (0, 0), (0, 1), (0, 1)), mode="CONSTANT")
output = pad(output)
print(output.shape)
#(1, 32, 96, 96)
```
