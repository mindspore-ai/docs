# Comparing the function difference with torch.nn.Module.buffers

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/pytorch_diff/UnTrainableParams.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.Module.buffers

```python
torch.nn.Module.buffers(recurse=True)
```

For more information, see [torch.nn.Module.buffers](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Module.buffers).

## mindspore.nn.Cell.untrainable_params

```python
mindspore.nn.Cell.untrainable_params(recurse=True)
```

For more information, see [mindspore.nn.Cell.untrainable_params](https://mindspore.cn/docs/en/r1.7/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.untrainable_params).

## Differences

In PyTorch, the network has three concepts: `parameter`, `buffer`, and `state`, where `state` is the collection of `parameter` and `buffer`. `parameter` can use the `requires_grad` attribute to distinguish whether the `parameter` in the network needs to be optimized; `buffer` is mostly defined as an invariant in the network, for example, when defining the network, the `running_mean` and `running_var` in BN will be automatically register as buffer; users can also register `parameter` and `buffer` through related interfaces.

-`torch.nn.Module.buffers`: Get the buffer in the network, and return a generator.

-`torch.nn.Module.named_buffers`: Get the name of buffer and buffer itself in the network, and return a generator.

In MindSpore, there is only the concept of `parameter` currently. The `requires_grad` attribute is used to distinguish whether the `parameter` in the network needs to be optimized. For example, when defining the network, the `moving_mean` and `moving_var` in BN will be defined as `parameter` with attribute `requires_grad=False`.

-`mindspore.nn.Cell.untrainable_params`: The function returns a list of all untrainable parameters. `Parameter` has an attribute `name` in MindSpore, names of parameters can be obtained after getting parameters by using the `untrainable_params` method.

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
