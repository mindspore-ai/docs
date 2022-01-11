# Comparing the function difference with torch.nn.Module.parameters

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/GetParams.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

## torch.nn.Module.parameters

```python
torch.nn.Module.parameters(recurse=True)
```

For more information, see [torch.nn.Module.parameters](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Module.parameters).

## mindspore.nn.Cell.get_parameters

```python
mindspore.nn.Cell.get_parameters(expand=True)
```

For more information, see [mindspore.nn.Cell.get_parameters](https://mindspore.cn/docs/api/en/r1.6/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.get_parameters).

## Differences

In PyTorch, the network has three concepts: `parameter`, `buffer`, and `state`, where `state` is the collection of `parameter` and `buffer`. `parameter` can use the `requires_grad` attribute to distinguish whether the `parameter` in the network needs to be optimized; `buffer` is mostly defined as an invariant in the network, for example, when defining the network, the `running_mean` and `running_var` in BN will be automatically register as buffer; users can also register `parameter` and `buffer` through related interfaces.

-`torch.nn.Module.parameters`: Get the `parameter` in the network, and return a generator.

-`torch.nn.Module.named_parameters`: Get the name of  `parameter` and  `parameter` itself in the network, and return a generator.

In MindSpore, there is only the concept of `parameter` currently. The `requires_grad` attribute is used to distinguish whether the `parameter` in the network needs to be optimized. For example, when defining the network, the `moving_mean` and `moving_var` in BN will be defined as `parameter` with attribute `requires_grad=False`.

-`mindspore.nn.Cell.get_parameters`: Get the `parameter` in the network, and return a generator.

-`mindspore.nn.Cell.trainable_params`: The function returns a list of all trainable parameters(with attribute `requires_grad=True`).

Due to the difference in concept definitions, although both `torch.nn.Module.parameters` and `mindspore.nn.Cell.get_parameters` get the `parameter` in the network, the returned content is slightly different: for example, `moving_mean` and `moving_variance`  in BN are registered as `buffer` in PyTorch, so they will not be returned by `torch.nn.Module.parameters` interface, but they will be returned by  `mindspore.nn.Cell.get_parameters` because they are defined as `parameter` in MindSpore.

## Code Example

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
<class 'generator'>

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
