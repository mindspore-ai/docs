# 比较与torch.nn.Conv2d的功能差异

## torch.nn.Conv2d

``` text
class torch.nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode='zeros'
)(input) -> Tensor
```

更多内容详见 [torch.nn.Conv2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.Conv2d.html)。

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

PyTorch：对输入Tensor计算二维卷积，通常情况下，输入大小为 $\left(N, C_{\mathrm{in}}, H, W\right)$ 、输出大小为 $\left(N, C_{\text {out }}, H_{\text {out }}, W_{\text {out }}\right)$ 的输出值可以描述为：
$\operatorname{out}\left(N_{i}, C_{\text {out }_{j}}\right)=\operatorname{bias}\left(C_{\text {out }_{j}}\right)+\sum_{k=0}^{C_{i n}-1} \text { weight }\left(C_{\text {out }_{j}}, k\right) \star \operatorname{input}\left(N_{i}, k\right)$
其中，$\star$ 为2D cross-correlation 算子，$N$ 是batch size，$C$ 是通道数量，$H$ 和 $W$ 分别是特征层的高度和宽度。

MindSpore：与PyTorch实现的功能基本一致，但存在偏置差异和填充差异。1、偏置差异：MindSpore默认不添加偏置参数，与PyTorch相反。2、填充差异：MindSpore默认对输入进行填充，而PyTorch则默认不填充。MindSpore填充模式可选项与PyTorch不同，PyTorch的参数padding_mode可选项有‘zeros’、'reflect'、'replicate'、'circular'，含义如下：

zero：常量填充（默认零填充）。

reflect：反射填充。

replicate：复制填充。

circular：循环填充。

而MindSpore的参数pad_mode可选项有'same'、'valid'、'pad'，含义如下：

same：使用0填充使得输出的高度和宽度分别与输入整除 stride 后的值相同，使用此模式，padding的值必须为0。

valid：不填充。

pad：零填充，使用此模式，padding必须大于或等于0。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | in_channels | in_channels |- |
| | 参数2 | out_channels | out_channels |- |
| | 参数3 | kernel_size | kernel_size |- |
| | 参数4 | stride | stride |- |
| | 参数5 | padding | padding |-|
| | 参数6 | padding_mode | pad_mode |PyTorch与MindSpore可选项不同，默认值不同|
| | 参数7 | dilation | dilation |-|
| | 参数8 | groups | group |功能一致，参数名不同|
| | 参数9 | bias | has_bias |功能一致，参数名不同，默认值不同|
| | 参数10 | - | weight_init |权重参数的初始化方法，PyTorch可使用init函数进行权重初始化|
| | 参数11 | - | bias_init |偏置参数的初始化方法，PyTorch无此参数|
| | 参数12 | - | data_format |指定输入数据格式，PyTorch无此参数|
| |参数14  | input | x  | 接口输入，功能一致，仅参数名不同 |

### 代码示例1

> PyTorch的参数bias默认值为True，即默认添加偏置参数，而MindSpore的参数has_bias默认值为False，即默认不添加偏置函数，如果需要添加偏置参数，需要将has_bias的值设置为True。

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x_ = np.ones((1, 120, 1024, 640))
x = tensor(x_, dtype=torch.float32)
net = torch.nn.Conv2d(120, 240, 4)
output = net(x).detach().numpy().shape
print(output)
# (1, 240, 1021, 637)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

x_ = np.ones((1, 120, 1024, 640))
x = Tensor(x_, mindspore.float32)
net = nn.Conv2d(120, 240, 4, has_bias=True, pad_mode='valid')
output = net(x).shape
print(output)
# (1, 240, 1021, 637)
```

### 代码示例2

> Pytorch的参数padding_mode为'zero'时，表示对输入进行零填充，而MindSpore中实现零填充需设置参数pad_mode为'pad'。

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x_ = np.ones((1, 120, 1024, 640))
x = tensor(x_, dtype=torch.float32)
net = torch.nn.Conv2d(120, 240, 4, padding=1, padding_mode='zeros')
output = net(x).detach().numpy().shape
print(output)
# (1, 240, 1023, 639)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

x_ = np.ones((1, 120, 1024, 640))
x = Tensor(x_, mindspore.float32)
net = nn.Conv2d(120, 240, 4, padding=1, pad_mode='pad')
output = net(x).shape
print(output)
# (1, 240, 1023, 639)
```

### 代码示例3

> Pytorch的参数padding_mode为'reflect'时，表示对输入进行反射填充，而MindSpore中实现反射填充需通过API组合实现，首先调用nn.Pad对输入x进行反射填充，再对填充后的结果进行卷积操作。

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x_ = np.ones((1, 120, 1024, 640))
x = tensor(x_, dtype=torch.float32)
net = torch.nn.Conv2d(120, 240, 4, padding=1, padding_mode='reflect')
output = net(x).detach().numpy().shape
print(output)
# (1, 240, 1023, 639)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

x_ = np.ones((1, 120, 1024, 640))
x = Tensor(x_, mindspore.float32)
pad = nn.Pad(paddings=((0,0),(1,1),(1,1),(1,1)), mode="REFLECT")
x_pad = pad(x)
net = nn.Conv2d(122, 240, 4, padding=0, pad_mode='valid')
output = net(x_pad).shape
print(output)
# (1, 240, 1023, 639)
```

### 代码示例4

> Pytorch默认情况下不对输入进行填充，而MindSpore默认情况下需要对输入进行填充，如果不对输入进行填充，需要将pad_mode设置为'valid'。

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x_ = np.ones((1, 120, 1024, 640))
x = tensor(x_, dtype=torch.float32)
net = torch.nn.Conv2d(120, 240, 4)
output = net(x).detach().numpy().shape
print(output)
# (1, 240, 1021, 637)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

x_ = np.ones((1, 120, 1024, 640))
x = Tensor(x_, mindspore.float32)
net = nn.Conv2d(120, 240, 4, pad_mode='valid')
output = net(x).shape
print(output)
# (1, 240, 1021, 637)
```
