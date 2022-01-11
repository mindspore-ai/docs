# 比较与torch.nn.Module.parameters()的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/GetParams.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## torch.nn.Module.parameters

```python
torch.nn.Module.parameters(recurse=True)
```

更多内容详见[torch.nn.Module.parameters](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Module.parameters)。

## mindspore.nn.Cell.get_parameters

```python
mindspore.nn.Cell.get_parameters(expand=True)
```

更多内容详见[mindspore.nn.Cell.get_parameters](https://mindspore.cn/docs/api/zh-CN/r1.6/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.get_parameters)。

## 使用方式

PyTorch中，网络有`parameter`, `buffer`, `state`三种概念，其中`state`为`parameter`和`buffer`的合集。`parameter`可以通过`requires_grad`属性来区分网络中的参数是否需要优化；`buffer`多定义为网络中的不变量，例如在定义网络时，BN中的`running_mean`和`running_var`会被自动注册为buffer；用户也可以通过相关接口自行注册`parameter`和`buffer`。

- `torch.nn.Module.parameters`： 获取网络中的`parameter`，返回类型为迭代器。

- `torch.nn.Module.named_parameters`：获取网络中`parameter`的名称和`parameter`本身，返回类型为迭代器。

MindSpore中目前只有`parameter`的概念，通过`requires_grad`属性来区分网络中的参数是否需要优化，例如在定义网络时，BN中的`moving_mean`和`moving_var`会被定义为`requires_grad=False`的`parameter`。

- `mindspore.nn.Cell.get_parameters`： 获取网络中的`parameter`，返回类型为迭代器。

- `mindspore.nn.Cell.trainable_paramters`：获取网络中需要被优化的`parameter`（即`requires_grad=True`），返回类型为列表。

因此，因为概念定义的差异，虽然`torch.nn.Module.parameters`和`mindspore.nn.Cell.get_parameters`都是获取网络中的 `parameter`，但是返回的内容略有不同：例如，BN中的不变量`moving_mean`和`moving_variance`，在PyTorch中被注册成`buffer`，所以不会在`torch.nn.Module.parameters`接口中返回，而在MindSpore中仍然属于`parameter`，所以会在`mindspore.nn.Cell.get_parameters`中返回。

## 代码示例

```python
from mindspore import nn

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

# The following implements mindspore.nn.Cell.get_parameters() with MindSpore.
net = MyNet()

print(type(net.get_parameters()), "\n")
for params in net.get_parameters():
  print("Name: ", params.name)
  print("params: ", params)
```

```text
# Out:

Name:  build_block.0.conv.weight
params:  Parameter (name=build_block.0.conv.weight, shape=(64, 3, 3, 3), dtype=Float32, requires_grad=True)
Name:  build_block.0.bn.moving_mean
params:  Parameter (name=build_block.0.bn.moving_mean, shape=(64,), dtype=Float32, requires_grad=False)
Name:  build_block.0.bn.moving_variance
params:  Parameter (name=build_block.0.bn.moving_variance, shape=(64,), dtype=Float32, requires_grad=False)
Name:  build_block.0.bn.gamma
params:  Parameter (name=build_block.0.bn.gamma, shape=(64,), dtype=Float32, requires_grad=True)
Name:  build_block.0.bn.beta
params:  Parameter (name=build_block.0.bn.beta, shape=(64,), dtype=Float32, requires_grad=True)

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

# The following implements torch.nn.Module.parameters() with torch.
net = MyNet()
print(type(net.parameters()), "\n")
for name, params in net.named_parameters():
  print("Name: ", name)
  print("params: ", params.size())
```

```text
# Out:
<class 'generator'>

Name:  build_block.0.conv.weight
params:  torch.Size([64, 3, 3, 3])
Name:  build_block.0.conv.bias
params:  torch.Size([64])
Name:  build_block.0.bn.weight
params:  torch.Size([64])
Name:  build_block.0.bn.bias
params:  torch.Size([64])
```
