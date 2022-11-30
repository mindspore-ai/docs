# 比较与torch.nn.ConvTranspose3d的功能差异

## torch.nn.ConvTranspose3d

```text
class torch.nn.ConvTranspose3d(
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

更多内容详见[torch.nn.ConvTranspose3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ConvTranspose3d.html)。

## mindspore.nn.Conv3dTranspose

``` text
class mindspore.nn.Conv3dTranspose(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    pad_mode='same',
    padding=0,
    dilation=1,
    group=1,
    output_padding=0,
    has_bias=False,
    weight_init='normal',
    bias_init='zeros',
    data_format='NCDHW'
)(x) -> Tensor
```

更多内容详见 [mindspore.nn.Conv3dTranspose](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Conv3dTranspose.html)。

## 差异对比

PyTorch：计算三维转置卷积，可以视为Conv3d对输入求梯度，也称为反卷积（实际不是真正的反卷积）。输入的shape通常是$(N,C_{in},D_{in},H_{in},W_{in})$，其中$N$是batch size，$C$是空间维度，$D_{in},H_{in},W_{in}$分别为特征层的深度，高度和宽度。输出的shape为$(N,C_{out},D_{out},H_{out},W_{out})$，计算公式如下：
$D_{out}=(D_{in}−1)×stride[0]−2×padding[0]+dilation[0]×(kernel\underline{ }size[0]−1)+output\underline{ }padding[0]+1$
$H_{out}=(H_{in}−1)×stride[1]−2×padding[1]+dilation[1]×(kernel\underline{ }size[1]−1)+output\underline{ }padding[1]+1$
$W_{out}=(W_{in}−1)×stride[2]−2×padding[2]+dilation[2]×(kernel\underline{ }size[2]−1)+output\underline{ }padding[2]+1$

MindSpore：MindSpore此API实现功能与PyTorch基本一致，新增了填充模式参数"pad_mode"，当"pad_mode" = "pad"时与PyTorch默认方式相同，利用weight_init 和bias_init 参数可以配置初始化方式。

| 分类 | 子类  | PyTorch | MindSpore | 差异                                   |
| ---- | ----- | ------- | --------- | -------------------------------------- |
| 参数 | 参数1 | in_channels    | in_channels          | -  |
|      | 参数2 | out_channels    | out_channels      | - |
|      | 参数3 | kernel_size  |  kernel_size          | -  |
|      | 参数4 | stride   |  stride           | -  |
|      | 参数5 | padding   |  padding           | 功能一致，PyTorch中只能在三个维度的两侧分别填充相同的值，可为长度为3的tuple。MindSpore中可以分别设置前部、尾部、顶部、底部、左边和右边的填充数量，可为长度为6的tuple  |
|      | 参数6 | output_padding   |  output_padding          | -  |
|      | 参数7 | groups   |  group           | 功能一致，参数名不同  |
|      | 参数8 | bias   |  has_bias           | PyTorch默认为True，MindSpore默认为False  |
|      | 参数9 | dilation   |  dilation           | -  |
|      | 参数10 |  padding_mode   |   -      | 数值填充模式，只支持"zeros"即填充0。MindSpore无此参数，但默认填充0|
|      | 参数11 |  -   | pad_mode       | 指定填充模式。可选值为"same"、"valid"、"pad"，在"same"和"valid"模式下，padding必须设置为0，默认为"same" |
|      | 参数12 |    - | weight_init        | 权重参数的初始化方法。可为Tensor，str，Initializer或numbers.Number。当使用str时，可选"TruncatedNormal"，"Normal"，"Uniform"，"HeUniform"和"XavierUniform"分布以及常量"One"和"Zero"分布的值。默认为"normal" |
|      | 参数13 |    -  | bias_init        | 偏置参数的初始化方法。可选填参数与"weight_init"相同，默认为"zeros" |
|      | 参数14 |   -   |   data_format  |数据格式的可选值。目前仅支持"NCDHW"，与PyTorch中默认顺序一致|
| | 参数15 | input | x | 接口输入，功能一致，仅参数名不同|

### 代码示例1

> 两API都是实现三维转置卷积运算，使用时需先进行实例化。PyTorch中深度，高度和宽度的padding值在同一方向上相同，如padding设为(1,2,4)表示分别在深度，高度和宽度的两侧填充1，2和4面0，对应在MindSpore中将pad_mode设为"pad"，padding设置为(1,1,2,2,4,4)。PyTorch中利用net.weight.data = torch.ones()的方式将权重初始化为1，shape为$(in\underline{ }channels,\frac {out\underline{ }channels}{groups}, kernel\underline{ }size[0], kernel\underline{ }size[1],kernel\underline{ }size[2])$，MindSpore直接设置参数weight_init = "ones"。

```python
# PyTorch
import torch
from torch import tensor
import torch.nn as nn
import numpy as np

k = (3, 5, 2)
x_ = np.ones([1, 3, 4, 9, 16])
x = tensor(x_, dtype=torch.float32)
net = nn.ConvTranspose3d(3, 32, kernel_size=k, padding=(1,2,4), bias=False)
net.weight.data = torch.ones(3, 32, k[0], k[1], k[2])
output = net(x).detach().numpy()
print(output.shape)
# (1, 32, 4, 9, 9)

# MindSpore
import mindspore as ms
import mindspore.nn as nn
import numpy as np

k = (3, 5, 2)
x_ = np.ones([1, 3, 4, 9, 16])
x = ms.Tensor(x_, ms.float32)
net = nn.Conv3dTranspose(3, 32, kernel_size=k, weight_init='ones', pad_mode='pad', padding=(1,1,2,2,4,4))
output = net(x)
print(output.shape)
# (1, 32, 4, 9, 9)
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
x_ = np.ones([1, 3, 4, 9, 16])
x = tensor(x_, dtype=torch.float32)
net = nn.ConvTranspose3d(3, 32, kernel_size=k, stride=s, padding=(k-1)//2, output_padding=s-1, bias=False)
net.weight.data = torch.ones(3, 32, k, k, k)
output = net(x).detach().numpy()
print(output.shape)
# (1, 32, 12, 27, 48)


# MindSpore
import mindspore as ms
import mindspore.nn as nn
import numpy as np

k = 5
s = 3
x_ = np.ones([1, 3, 4, 9, 16])
x = ms.Tensor(x_, ms.float32)
net = nn.Conv3dTranspose(3, 32, kernel_size=k, stride=s, weight_init='ones', pad_mode='same')
output = net(x)
print(output.shape)
# (1, 32, 12, 27, 48)
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
x_ = np.ones([1, 3, 4, 9, 16])
x = tensor(x_, dtype=torch.float32)
net = nn.ConvTranspose3d(3, 32, kernel_size=k, stride=s, bias=False)
net.weight.data = torch.ones(3, 32, k, k, k)
output = net(x).detach().numpy()
print(output.shape)
# (1, 32, 14, 29, 50)


# MindSpore
import mindspore as ms
import mindspore.nn as nn
import numpy as np

k = 5
s = 3
x_ = np.ones([1, 3, 4, 9, 16])
x = ms.Tensor(x_, ms.float32)
net = nn.Conv3dTranspose(3, 32, kernel_size=k, stride=s, weight_init='ones', pad_mode='valid')
output = net(x)
print(output.shape)
# (1, 32, 14, 29, 50)
```