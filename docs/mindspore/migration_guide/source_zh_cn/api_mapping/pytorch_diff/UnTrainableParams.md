# 比较与torch.nn.Module.buffers的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/UnTrainableParams.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## torch.nn.Module.buffers

```python
torch.nn.Module.buffers(recurse=True)
```

## mindspore.nn.Cell.untrainable_params

```python
mindspore.nn.Cell.untrainable_params(recurse=True)
```

## 使用方式

PyTorch中，网络有`parameter`, `buffer`, `state`三种概念，其中`state`为`parameter`和`buffer`的合集。`parameter`可以通过`requires_grad`属性来区分网络中的参数是否需要优化；`buffer`多定义为网络中的不变量，例如在定义网络时，BN中的`running_mean`和`running_var`会被自动注册为`buffer`；用户也可以通过相关接口自行注册`parameter`和`buffer`。

- torch.nn.Module.buffers： 获取网络中的`buffer`，返回类型为迭代器。

- torch.nn.Module.named_buffers：获取网络中的`buffer`名称和`buffer`本身，返回类型为迭代器。

MindSpore中目前只有`parameter`的概念，通过`requires_grad`属性来区分网络中的参数是否需要优化，例如在定义网络时，BN中的`moving_mean`和`moving_var`会被定义为`requires_grad=False`的`parameter`。

- mindspore.nn.Cell.untrainable_params：获取网络中不需要被优化器优化的参数，返回类型为列表。MindSpore中的`Parameter`含有属性`name`，在使用`untrainable_params`方法获取参数后，可以使用此属性获取名称。

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
