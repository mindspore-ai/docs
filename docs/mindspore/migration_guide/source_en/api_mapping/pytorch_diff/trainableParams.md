# Comparing the function difference with torch.nn.Module.parameters

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/trainableParams.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## torch.nn.Module.parameters

```python
torch.nn.Module.parameters()
```

For more information, see[torch.nn.Module.parameters](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Module.parameters).

## mindspore.nn.Cell.trainable_params

```python
mindspore.nn.Cell.trainable_params()
```

For more information, see[mindspore.nn.Cell.trainable_params](https://mindspore.cn/docs/api/en/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.trainable_params).

## Differences

PyTorch:

- torch.nn.Module.parameters(): The function returns a Generator over module parameters to be trained.

- torch.nn.Module.named_parameters(): The function returns an Generator over trainable parameters in the module, yielding both names of the parameters as well as the parameters themselves.

MindSpore: The function returns a list of all trainable parameters. `Parameter` has an attribute `name` in MindSpore, names of parameters can be obtained after getting parameters by using the `trainable_params` method.

## Code Example

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

# The following implements mindspore.nn.Cell.trainable_params() with MindSpore.
net = MyNet()
print(type(net.trainable_params()), "\n")
for params in net.trainable_params():
  print("Name: ", params.name)
  print("params: ", params)
```

```text
# Out:
<class 'list'>

Name:  build_block.0.conv.weight
params:  Parameter (name=build_block.0.conv.weight, shape=(64, 3, 3, 3), dtype=Float32, requires_grad=True)
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
