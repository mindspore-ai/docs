# Comparing the function difference with torch.nn.Module.buffers

## torch.nn.Module.buffers

```python
torch.nn.Module.buffers()
```

## mindspore.nn.Cell.untrainable_params

```python
mindspore.nn.Cell.untrainable_params()
```

## Differences

PyTorch:

- torch.nn.Module.buffers(): The function returns a Generator over module persistent buffers.

- torch.nn.Module.named_buffers(): The function returns a Generator over module persistent buffers, yielding both the names of the buffers as well as the buffers themselves.

MindSpore: The function returns a list of all untrainable parameters. `Parameter` has an attribute `name` in MindSpore, names of parameters can be obtained after getting parameters by using the `untrainable_params` method.

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

# The following implements mindspore.nn.Cell.untrainable_params() with MindSpore.
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
