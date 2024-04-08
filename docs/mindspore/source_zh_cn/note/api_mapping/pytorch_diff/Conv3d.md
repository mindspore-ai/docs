# 比较与torch.nn.Conv3d的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3.q1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3.q1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Conv3d.md)

## torch.nn.Conv3d

```text
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

更多内容详见[torch.nn.Conv3d](https://pytorch.org/docs/1.8.1/generated/torch.nn.Conv3d.html)。

## mindspore.nn.Conv3d

```text
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
    weight_init=None,
    bias_init=None,
    data_format='NCDHW'
)(x) -> Tensor
```

更多内容详见[mindspore.nn.Conv3d](https://www.mindspore.cn/docs/zh-CN/r2.3.0rc1/api_python/nn/mindspore.nn.Conv3d.html)。

## 差异对比

PyTorch：对输入Tensor计算三维卷积，通常情况下，输入大小为$\left(N, C_{i n}, D, H, W\right)$、输出大小为$\left(N, C_{\text {out }}, D_{\text {out }}, H_{\text {out }}, W_{\text {out }}\right)$的输出值可以描述为：

$$
\operatorname{out}\left(N_{i}, C_{\text {out }_{j}}\right)=\operatorname{bias}\left(C_{\text {out }_{j}}\right)+\sum_{k=0}^{C_{i n}-1} \text { weight }\left(C_{\text {out }_{j}}, k\right) \star \operatorname{input}\left(N_{i}, k\right)
$$

其中，$\star$为3d cross-correlation 算子，$N$是batch size，$C$是通道数量，$D$、$H$、$W$分别是特征层的深度、高度和宽度。

MindSpore：与PyTorch实现的功能基本一致，但存在偏置差异和填充差异。

1. 偏置差异：MindSpore默认不添加偏置参数，与PyTorch相反。
2. 填充差异：MindSpore默认对输入进行填充，而PyTorch则默认不填充。同时MindSpore填充模式可选项和行为与PyTorch不同，填充行为具体差异如下。

### 填充行为差异

 1. PyTorch的参数padding_mode可选项有‘zeros’、‘reflect’、‘replicate’、‘circular，默认为‘zeros’，参数padding可选项有int、tuple of ints、‘valid’、‘same’，默认为0，参数padding_mode与torch.nn.functional.pad接口的四种填充模式一致，设置过后会对卷积的输入按照指定模式的填充方式进行填充，如下：

    - zero：常量填充（默认零填充）。

    - reflect：反射填充。

    - replicate：边缘复制填充。

    - circular：循环填充。

    由padding_mode决定填充方式后，padding参数则用于控制填充的数量与位置。针对Conv2d，padding指定为int的时候，会在输入的前后上下左右进行padding次的填充(若为默认值0则代表不进行填充)；padding指定为tuple的时候，会按照tuple的输入在前后上下左右进行指定次数的填充；padding设置为‘valid’模式时，不进行填充，只会在不超出特征图的范围内进行卷积；padding设置为‘same’ 模式时，若需要padding的元素个数为偶数个，padding的元素则会均匀的分布在特征图的上下左右，若需要padding的元素个数为奇数个，PyTorch则会优先填充特征图的左侧和上侧。

2. MindSpore的参数pad_mode可选项有‘same’、‘valid’、‘pad’，参数padding只能输入int，填充参数的详细意义如下：

    pad_mode设置为‘pad’的时候，MindSpore可以设置padding参数为大于等于0的正整数，会在输入的前后上下左右进行padding次的0填充(若为默认值0则代表不进行填充)；pad_mode为另外两种模式时，padding参数必须只能设置为0，pad_mode设置为‘valid’模式时，不进行填充，只会在不超出特征图的范围内进行卷积；pad_mode设置为‘same’模式时，若需要padding的元素个数为偶数个，padding的元素则会均匀的分布在特征图的上下左右，若需要padding的元素个数为奇数个，MindSpore则会优先填充特征图的右侧和下侧(与PyTorch不同，类似TensorFlow)。

    因此MindSpore若想实现与PyTorch一致的填充模式，需要先手动使用nn.Pad或者ops.pad接口对输入进行手动填充。

### 权重初始化差异

mindspore.nn.Conv3d的 `weight_init` 是 ``None`` 时，权重使用HeUniform初始化。此时和PyTorch权重初始化方式一致。

mindspore.nn.Conv3d的 `bias_init` 是 ``None`` 时，偏差使用Uniform初始化。此时和PyTorch偏差初始化方式一致。

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
| | 参数9 | padding_mode | pad_mode |具体差异参考上文|
| | 参数10 | - | weight_init |权重参数的初始化方法，具体差异参考上文|
| | 参数11 | - | bias_init |偏置参数的初始化方法，具体差异参考上文|
| | 参数12 | - | data_format |输入数据格式，PyTorch无此参数|
| 输入 | 单输入 | input  | x | 功能一致，参数名不同 |

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
