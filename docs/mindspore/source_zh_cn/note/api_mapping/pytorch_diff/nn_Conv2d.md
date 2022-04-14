# 比较与torch.nn.Conv2d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/nn_Conv2d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.Conv2d

```python
torch.nn.Conv2d(
    in_channels=120,
    out_channels=240,
    kernel_size=4,
    stride=1,
    padding=0,
    padding_mode='zeros',
    dilation=1,
    groups=1,
    bias=True
)
```

更多内容详见[torch.nn.Conv2d](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Conv2d)。

## mindspore.nn.Conv2d

```python
class mindspore.nn.Conv2d(
    in_channels=120,
    out_channels=240,
    kernel_size=4,
    stride=1,
    pad_mode='same',
    padding=0,
    dilation=1,
    groups=1,
    has_bias=False,
    weight_init='normal',
    bias_init='zeros',
    data_format='NCHW'
)(input_x)
```

更多内容详见[mindspore.nn.Conv2d](https://mindspore.cn/docs/zh-CN/r1.7/api_python/nn/mindspore.nn.Conv2d.html#mindspore.nn.Conv2d)。

## 使用方式

PyTorch：默认不对输入进行填充，bias为True。

MindSpore：默认对输入进行填充，使输出与输入维度一致，如果不需要padding，可以将参数设为'valid'。默认has_bias为False。

## 代码示例

```python
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import torch
import numpy as np

# In MindSpore
net = nn.Conv2d(120, 240, 4, stride=2, has_bias=True, weight_init='normal')
x = Tensor(np.ones([1, 120, 1024, 640]), mindspore.float32)
output = net(x).shape
print(output)
# Out:
# (1, 240, 512, 320)

# In MindSpore
net = nn.Conv2d(120, 240, 4, stride=2, pad_mode='valid', has_bias=True, weight_init='normal')
x = Tensor(np.ones([1, 120, 1024, 640]), mindspore.float32)
output = net(x).shape
print(output)
# Out:
# (1, 240, 511, 319)

# In PyTorch
m = torch.nn.Conv2d(120, 240, 4, stride=2)
input = torch.rand(1, 120, 1024, 640)
output = m(input)
print(output.shape)
# Out：
# torch.Size([1, 240, 511, 319])
```
