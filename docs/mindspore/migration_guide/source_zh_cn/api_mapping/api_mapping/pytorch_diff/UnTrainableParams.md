# 比较与torch.nn.Module.buffers的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/UnTrainableParams.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## torch.nn.Module.buffers

```python
torch.nn.Module.buffers()
```

## mindspore.nn.Cell.untrainable_params

```python
mindspore.nn.Cell.untrainable_params()
```

## 使用方式

PyTorch:

- torch.nn.Module.buffers()： 获取网络中不需要被优化器优化的参数，返回类型为迭代器。

- torch.nn.Module.named_buffers()：获取网络中不需要被优化器优化的参数名称和参数，返回类型为迭代器。

MindSpore：获取网络中不需要被优化器优化的参数，返回类型为列表。在MindSpore中的`Parameter`含有属性`name`，在使用`untrainable_params`方法获取参数后，可以使用此属性获取名称。

## 代码示例

```python
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

# The following implements mindspore.nn.Cell.untrainable_params() with MindSpore.
net = MyNet()
print(type(net.untrainable_params()), "\n")
for params in net.untrainable_params():
  print("Name: ", params.name)
  print("params: ", params)
```

```text
# Out:
<class 'list'>

Name:  build_block.0.bn.moving_mean
params:  Parameter (name=build_block.0.bn.moving_mean, shape=(64,), dtype=Float32, requires_grad=False)
Name:  build_block.0.bn.moving_variance
params:  Parameter (name=build_block.0.bn.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)
```

```python
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

# The following implements torch.nn.Module.buffers() with torch.
net = MyNet()
print(type(net.buffers()), "\n")
for name, params in net.named_buffers():
  print("Name: ", name)
  print("params: ", params.size())
```

```text
# Out:
<class 'generator'>

Name:  build_block.0.bn.running_mean
params:  torch.Size([64])
Name:  build_block.0.bn.running_var
params:  torch.Size([64])
Name:  build_block.0.bn.num_batches_tracked
params:  torch.Size([])
```
