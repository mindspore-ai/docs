# 比较与torch.nn.Conv3d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Conv3d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.Conv3d

``` text
class torch.nn.Conv3d(
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

更多内容详见 [torch.nn.Conv3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.Conv3d.html)。

## mindspore.nn.Conv3d

``` text
class mindspore.nn.Conv3d(
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
    data_format='NCDHW'
)(x) -> Tensor
```

更多内容详见 [mindspore.nn.Conv3d](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Conv3d.html)。

## 差异对比

PyTorch：对输入Tensor计算三维卷积，通常情况下，输入大小为 $\left(N, C_{i n}, D, H, W\right)$ 、输出大小为 $\left(N, C_{\text {out }}, D_{\text {out }}, H_{\text {out }}, W_{\text {out }}\right)$ 的输出值可以描述为：
$$
\operatorname{out}\left(N_{i}, C_{\text {out }_{j}}\right)=\operatorname{bias}\left(C_{\text {out }_{j}}\right)+\sum_{k=0}^{C_{i n}-1} \text { weight }\left(C_{\text {out }_{j}}, k\right) \star \operatorname{input}\left(N_{i}, k\right)
$$
其中，$\star$ 为3d cross-correlation 算子，$N$ 是batch size，$C$ 是通道数量，$D$、$H$、$W$ 分别是特征层的深度、高度和宽度。

MindSpore：与PyTorch实现的功能基本一致，但默认不添加偏置参数，与PyTorch相反。且MindSpore默认对输入进行填充，而PyTorch则默认不填充。同时MindSpore填充模式可选项与PyTorch不同，PyTorch的参数padding_mode可选项有'zeros'、'reflect'、'replicate'、'circular'，含义如下：

zero：常量填充（默认零填充）。

reflect：反射填充，但当使用Conv3d时，无法使用这种填充方式。

replicate：复制填充。

circular：循环填充。

而MindSpore的参数pad_mode可选项有'same'、'valid'、'pad'，含义如下：

same：输出的宽度与输入整除 stride 后的值相同。

valid：不填充。

pad：零填充。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | in_channels | in_channels |- |
| | 参数2 | out_channels | out_channels |- |
| | 参数3 | kernel_size | kernel_size |- |
| | 参数4 | stride | stride |- |
| | 参数5 | padding | padding |-|
| | 参数6 | dilation | dilation |-|
| | 参数7 | groups | group |功能一致，参数名不同|
| | 参数8 | bias | has_bias |功能一致，参数名不同，默认值不同|
| | 参数9 | padding_mode | pad_mode |PyTorch与MindSpore可选项不同，默认值不同|
| | 参数10 | - | weight_init |权重参数的初始化方法，PyTorch无此参数|
| | 参数11 | - | bias_init |偏置参数的初始化方法，PyTorch无此参数|
| | 参数12 | - | data_format |输入数据格式，PyTorch无此参数|
| | 参数13 | input | x | 接口输入，功能一致，仅参数名不同|

### 代码示例1

> PyTorch的参数bias默认值为True，即默认添加偏置参数，而MindSpore的参数has_bias默认值为False，即默认不添加偏置函数，如果需要添加偏置参数，需要将has_bias的值设置为True。

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x_ = np.ones((16, 3, 10, 32, 32))
x = tensor(x_, dtype=torch.float32)
net = torch.nn.Conv3d(3, 32, (4, 3, 3))
output = net(x).detach().numpy().shape
print(output)
# (16, 32, 7, 30, 30)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

x_ = np.ones((16, 3, 10, 32, 32))
x = Tensor(x_, mindspore.float32)
net = nn.Conv3d(3, 32, (4, 3, 3), has_bias=True, pad_mode='valid')
output = net(x).shape
print(output)
# (16, 32, 7, 30, 30)
```

### 代码示例2

> PyTorch的参数padding_mode为'zero'时，表示对输入进行零填充，而MindSpore中实现零填充需设置参数pad_mode为'pad'。

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x_ = np.ones((16, 3, 10, 32, 32))
x = tensor(x_, dtype=torch.float32)
net = torch.nn.Conv3d(3, 32, (4, 3, 3), padding=1, padding_mode='zeros')
output = net(x).detach().numpy().shape
print(output)
# (16, 32, 9, 32, 32)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

x_ = np.ones((16, 3, 10, 32, 32))
x = Tensor(x_, mindspore.float32)
net = nn.Conv3d(3, 32, (4, 3, 3), padding=1, pad_mode='pad')
output = net(x).shape
print(output)
# (16, 32, 9, 32, 32)
```

### 代码示例3

> PyTorch默认情况下不对输入进行填充，而MindSpore默认情况下需要对输入进行填充，如果不对输入进行填充，需要将pad_mode设置为'valid'。

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x_ = np.ones((16, 3, 10, 32, 32))
x = tensor(x_, dtype=torch.float32)
net = torch.nn.Conv3d(3, 32, (4, 3, 3))
output = net(x).detach().numpy().shape
print(output)
# (16, 32, 7, 30, 30)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

x_ = np.ones((16, 3, 10, 32, 32))
x = Tensor(x_, mindspore.float32)
net = nn.Conv3d(3, 32, (4, 3, 3), pad_mode='valid')
output = net(x).shape
print(output)
# (16, 32, 7, 30, 30)
```
