# Function Differences with torch.nn.Conv2d

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/pytorch_diff/nn_Conv2d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

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

For more information, see [torch.nn.Conv2d](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Conv2d).

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

For more information, see [mindspore.nn.Conv2d](https://mindspore.cn/docs/en/r1.7/api_python/nn/mindspore.nn.Conv2d.html#mindspore.nn.Conv2d).

## Differences

PyTorch: No padding is applied to the input by default. bias is set to True by default.

MindSpore: Padding is applied to the input so the output's dimensions match with input's dimensions by default. If no padding is needed, set pad_mode to 'valid'. has_bias is set to False by default.

## Code Example

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
