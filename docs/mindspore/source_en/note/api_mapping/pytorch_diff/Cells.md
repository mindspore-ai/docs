# Comparing the function difference with torch.nn.Module.children

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Cells.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

## torch.nn.Module.children

```python
torch.nn.Module.children()
```

For more information, see [torch.nn.Module.children](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Module.children).

## mindspore.nn.Cell.cells

```python
mindspore.nn.Cell.cells()
```

For more information, see [mindspore.nn.Cell.cells](https://mindspore.cn/docs/en/r1.7/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.cells).

## Differences

PyTorch: The function returns a Generator over immediate children modules.

MindSpore: The function returns odict_values over immediate cells.

## Code Example

```python
# The following implements mindspore.nn.Cell.cells() with MindSpore.
import mindspore
import numpy as np
from mindspore import Tensor, nn

class ConvBN(nn.Cell):
  def __init__(self):
    super(ConvBN, self).__init__()
    self.conv = nn.Conv2d(3, 64, 3)
    self.bn = nn.BatchNorm2d(64)
  def construct(self, x):
    x = self.conv(x)
    x = self.bn(x)
    return x

class MyNet(nn.Cell):
  def __init__(self):
    super(MyNet, self).__init__()
    self.build_block = nn.SequentialCell(ConvBN(), nn.ReLU())
  def construct(self, x):
    return self.build_block(x)

net = MyNet()
print(net.cells())
```

```text
# Out:
odict_values([SequentialCell<
  (0): ConvBN<
    (conv): Conv2d<input_channels=3, output_channels=64, kernel_size=(3, 3),stride=(1, 1),  pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=Falseweight_init=normal, bias_init=zeros, format=NCHW>
    (bn): BatchNorm2d<num_features=64, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=build_block.0.bn.gamma, shape=(64,), dtype=Float32, requires_grad=True), beta=Parameter (name=build_block.0.bn.beta, shape=(64,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=build_block.0.bn.moving_mean, shape=(64,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=build_block.0.bn.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)>
    >
  (1): ReLU<>
  >])
```

```python
# The following implements torch.nn.Module.children() with torch.
import torch.nn as nn

class ConvBN(nn.Module):
  def __init__(self):
    super(ConvBN, self).__init__()
    self.conv = nn.Conv2d(3, 64, 3)
    self.bn = nn.BatchNorm2d(64)
  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    return x

class MyNet(nn.Module):
  def __init__(self):
    super(MyNet, self).__init__()
    self.build_block = nn.Sequential(ConvBN(), nn.ReLU())
  def construct(self, x):
    return self.build_block(x)

net = MyNet()
print(net.children())
for child in net.children():
  print(child)
```

```text
# Out:
<generator object Module.children at 0x7f5e48142bd0>
Sequential(
  (0): ConvBN(
    (conv): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
    (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (1): ReLU()
)
```
