# 比较与torch.nn.Module.children的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Cells.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.Module.children

```python
torch.nn.Module.children()
```

更多内容详见[torch.nn.Module.children](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Module.children)。

## mindspore.nn.Cell.cells

```python
mindspore.nn.Cell.cells()
```

更多内容详见[mindspore.nn.Cell.cells](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.cells)。

## 使用方式

PyTorch：获取网络中的外层子模块，返回类型为迭代器。

MindSpore：获取网络中的外层子模块，返回类型为odict_values。

## 代码示例

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
