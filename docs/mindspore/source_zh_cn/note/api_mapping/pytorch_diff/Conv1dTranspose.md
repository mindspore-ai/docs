# 比较与torch.nn.ConvTranspose1d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Conv1dTranspose.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.ConvTranspose1d

```text
class torch.nn.ConvTranspose1d(
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

更多内容详见[torch.nn.ConvTranspose1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.ConvTranspose1d.html)。

## mindspore.nn.Conv1dTranspose

```text
class mindspore.nn.Conv1dTranspose(
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
    bias_init='zeros')
)(x) -> Tensor
```

更多内容详见[mindspore.nn.Conv1dTranspose](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.Conv1dTranspose.html)。

## 差异对比

PyTorch：计算一维转置卷积，可以视为Conv1d对输入求梯度，也称为反卷积（实际不是真正的反卷积）。输入的shape通常是$(N,C_{in}, L_{in})$，其中$N$是batch size，$C$是空间维度，$L$是序列的长度。输出的shape为$(N,C_{out},L_{out})$，其中$L_{out}=(L_{in}−1)×stride−2×padding+dilation×(kernel\_size−1)+output\_padding+1$

MindSpore：MindSpore此API实现功能与PyTorch基本一致，新增了填充模式参数"pad_mode"，当"pad_mode" = "pad"时与PyTorch默认方式相同，利用weight_init 和bias_init 参数可以配置初始化方式。

| 分类 | 子类  | PyTorch | MindSpore | 差异                                   |
| ---- | ----- | ------- | --------- | -------------------------------------- |
| 参数 | 参数1 | in_channels    | in_channels          | -  |
|      | 参数2 | out_channels    | out_channels      | - |
|      | 参数3 | kernel_size  |  kernel_size          | -  |
|      | 参数4 | stride   |  stride           | -  |
|      | 参数5 | padding   |  padding           | -  |
|      | 参数6 | output_padding   |     -       | 通常在stride > 1的前提下使用，用来调整output shapes。例如，通常将padding设置为(kernel_size - 1)/2，此时设置output_padding = (stride - 1)可确保input shapes/output shapes = stride，MindSpore无此参数  |
|      | 参数7 | groups   |  group           | 功能一致，参数名不同  |
|      | 参数8 | bias   |  has_bias           | PyTorch默认为True，MindSpore默认为False |
|      | 参数9 | dilation   |  dilation           | -  |
|      | 参数10 |  padding_mode   |   -      | 数值填充模式，只支持"zeros"即填充0。MindSpore无此参数，但默认填充0|
|      | 参数11 |  -   | pad_mode       | 指定填充模式。可选值为"same"、"valid"、"pad"，在"same"和"valid"模式下，padding必须设置为0，默认为"same" |
|      | 参数12 |   -  | weight_init        | 权重参数的初始化方法。可为Tensor，str，Initializer或numbers.Number。当使用str时，可选"TruncatedNormal"，"Normal"，"Uniform"，"HeUniform"和"XavierUniform"分布以及常量"One"和"Zero"分布的值。默认为"normal" |
|      | 参数13 |   -  | bias_init        | 偏置参数的初始化方法。初始化方法与"weight_init"相同，默认为"zeros" |
|      | 参数14  | input               | x                    | 接口输入，功能一致，仅参数名不同 |

### 代码示例1

> 两API都是实现一维转置卷积运算，使用时需先进行实例化。当PyTorch将output_padding设为0，MindSpore将pad_mode设为"pad"时，输出宽度$L_{out}=(L_{in}−1)×stride−2×padding+dilation×(kernel\_size−1)+1$。PyTorch中利用net.weight.data = torch.ones()的方式将权重初始化为1，MindSpore直接设置参数weight_init = "ones"。

```python
# PyTorch
import torch
from torch import tensor
import torch.nn as nn
import numpy as np

k = 4
x_ = np.ones([1, 3, 50])
x = tensor(x_, dtype=torch.float32)
net = nn.ConvTranspose1d(3, 64, kernel_size=k, stride=1, padding=0, output_padding=0, bias=False)
net.weight.data = torch.ones(3, 64, k)
output = net(x).detach().numpy()
print(output.shape)
# (1, 64, 53)


# MindSpore
import mindspore as ms
import mindspore.nn as nn
import numpy as np

k = 4
x_ = np.ones([1, 3, 50])
x = ms.Tensor(x_, ms.float32)
net = nn.Conv1dTranspose(3, 64, kernel_size=k, weight_init='ones', pad_mode='pad')
output = net(x)
print(output.shape)
# (1, 64, 53)
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
x_ = np.ones([1, 3, 50])
x = tensor(x_, dtype=torch.float32)
net = nn.ConvTranspose1d(3, 64, kernel_size=k, stride=s, padding=(k-1)//2, output_padding=s-1, bias=False)
net.weight.data = torch.ones(3, 64, k)
output = net(x).detach().numpy()
print(output.shape)
# (1, 64, 150)


# MindSpore
import mindspore as ms
import mindspore.nn as nn
import numpy as np

k = 5
s = 3
x_ = np.ones([1, 3, 50])
x = ms.Tensor(x_, ms.float32)
net = nn.Conv1dTranspose(3, 64, kernel_size=k, stride=s, weight_init='ones', pad_mode='same', padding=0)
output = net(x)
print(output.shape)
# (1, 64, 150)
```

### 代码示例3

> 若不在原有图像上做任何填充，在stride>1的情况下可能舍弃一部分数据，输出宽度$L_{out}=(L_{in}−1)×stride+dilation×(kernel\_size−1)+1$。在PyTorch中将padding和output_padding设为0，MindSpore中设置pad_mode = "valid"。

```python
# PyTorch
import torch
from torch import tensor
import torch.nn as nn
import numpy as np

k = 5
s = 3
x_ = np.ones([1, 3, 50])
x = tensor(x_, dtype=torch.float32)
net = nn.ConvTranspose1d(3, 64, kernel_size=k, stride=s, padding=0, output_padding=0, bias=False)
net.weight.data = torch.ones(3, 64, k)
output = net(x).detach().numpy()
print(output.shape)
# (1, 64, 152)


# MindSpore
import mindspore as ms
import mindspore.nn as nn
import numpy as np

k = 5
s = 3
x_ = np.ones([1, 3, 50])
x = ms.Tensor(x_, ms.float32)
net = nn.Conv1dTranspose(3, 64, kernel_size=k, stride=s, weight_init='ones', pad_mode='valid', padding=0)
output = net(x)
print(output.shape)
# (1, 64, 152)
```